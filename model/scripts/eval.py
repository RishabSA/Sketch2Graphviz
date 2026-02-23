import os
import uuid
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.amp import autocast
from tqdm.auto import tqdm
from huggingface_hub import login
from skimage.metrics import structural_similarity
import lpips
import networkx as nx

from scripts.graphviz_renderer import (
    render_graphviz_dot_code_pil,
    convert_graphviz_dot_to_networkx,
)
from scripts.model import Sketch2GraphvizVLM, load_sketch2graphviz_vlm
from scripts.psql_vector_db import get_top_k_similar_vectors_from_db
from scripts.data import (
    get_json_graphviz_json_dataloaders,
    make_inputs_and_labels_vlm,
)
from scripts.prompts import graphviz_code_from_image_instruction


def get_attribute_items(attributes: dict, keys: tuple) -> tuple:
    return tuple(sorted((key, attributes[key]) for key in keys if key in attributes))


def build_node_set(graph: nx.Graph) -> set[str]:
    return {node_id for node_id in graph.nodes()}


def build_edge_set(graph: nx.Graph) -> set[tuple[str, str]]:
    return {
        (edge[0], edge[1]) if graph.is_directed() else tuple(sorted((edge[0], edge[1])))
        for edge in graph.edges()
    }


def build_node_attribute_set(
    graph: nx.Graph,
    attribute_keys: tuple,
) -> set[tuple[str, tuple]]:
    return {
        (node_id, get_attribute_items(attributes, attribute_keys))
        for node_id, attributes in graph.nodes(data=True)
    }


def build_edge_attribute_set(
    graph: nx.Graph,
    attribute_keys: tuple,
) -> set[tuple[str, str, tuple]]:
    is_directed = graph.is_directed()
    edge_attribute_set = set()

    if graph.is_multigraph():
        for u, v, _, attributes in graph.edges(keys=True, data=True):
            if is_directed:
                u_key, v_key = u, v
            else:
                u_key, v_key = sorted((u, v))

            edge_attribute_set.add(
                (u_key, v_key, get_attribute_items(attributes, attribute_keys))
            )
    else:
        for u, v, attributes in graph.edges(data=True):
            if is_directed:
                u_key, v_key = u, v
            else:
                u_key, v_key = sorted((u, v))

            edge_attribute_set.add(
                (u_key, v_key, get_attribute_items(attributes, attribute_keys))
            )

    return edge_attribute_set


def compute_precision_recall_f1(
    predicted_set: set,
    ground_truth_set: set,
) -> dict[str, float]:
    true_positive = len(predicted_set & ground_truth_set)
    false_positive = len(predicted_set - ground_truth_set)
    false_negative = len(ground_truth_set - predicted_set)

    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)
    f1 = 2 * precision * recall / (precision + recall)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def compute_networkx_graph_f1_metrics(
    original_graph: nx.Graph,
    generated_graph: nx.Graph,
    node_attribute_keys: tuple = ("label", "shape", "style", "fillcolor", "color"),
    edge_attribute_keys: tuple = (
        "label",
        "style",
        "color",
        "dir",
        "arrowhead",
        "arrowtail",
        "weight",
        "penwidth",
    ),
) -> dict[str, float]:
    original_node_set = build_node_set(original_graph)
    generated_node_set = build_node_set(generated_graph)
    node_metrics = compute_precision_recall_f1(generated_node_set, original_node_set)

    original_edge_set = build_edge_set(original_graph)
    generated_edge_set = build_edge_set(generated_graph)
    edge_metrics = compute_precision_recall_f1(generated_edge_set, original_edge_set)

    original_node_attribute_set = build_node_attribute_set(
        original_graph, node_attribute_keys
    )
    generated_node_attribute_set = build_node_attribute_set(
        generated_graph, node_attribute_keys
    )
    node_attribute_metrics = compute_precision_recall_f1(
        generated_node_attribute_set, original_node_attribute_set
    )

    original_edge_attribute_set = build_edge_attribute_set(
        original_graph, edge_attribute_keys
    )
    generated_edge_attribute_set = build_edge_attribute_set(
        generated_graph, edge_attribute_keys
    )
    edge_attribute_metrics = compute_precision_recall_f1(
        generated_edge_attribute_set, original_edge_attribute_set
    )

    return {
        "node_precision": node_metrics["precision"],
        "node_recall": node_metrics["recall"],
        "node_f1": node_metrics["f1"],
        "edge_precision": edge_metrics["precision"],
        "edge_recall": edge_metrics["recall"],
        "edge_f1": edge_metrics["f1"],
        "node_attribute_precision": node_attribute_metrics["precision"],
        "node_attribute_recall": node_attribute_metrics["recall"],
        "node_attribute_f1": node_attribute_metrics["f1"],
        "edge_attribute_precision": edge_attribute_metrics["precision"],
        "edge_attribute_recall": edge_attribute_metrics["recall"],
        "edge_attribute_f1": edge_attribute_metrics["f1"],
    }


def evaluate_vlm(
    model: Sketch2GraphvizVLM,
    iterator: DataLoader,
    instruction: str,
    description: str = "Testing",
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
) -> float:
    model.eval()
    model.llama_model.eval()
    model.device = device

    test_loss = 0.0

    progress_bar = tqdm(iterator, desc=description)

    for batch in progress_bar:
        images, graphviz_code = batch["images"], batch["graphviz_code"]

        if isinstance(images, torch.Tensor):
            images = images.to(device)  # shape: (batch_size, 3, H, W)

        # Inputs are token ids of full instruction + Graphviz code text
        # Labels are token ids of just Graphviz code text
        inputs, labels = make_inputs_and_labels_vlm(
            model=model,
            images=images,
            graphviz_code=graphviz_code,
            instruction=instruction,
        )

        with autocast(
            device_type="cuda", dtype=torch.float16, enabled=(device.type == "cuda")
        ), torch.inference_mode():
            outputs = model.llama_model(
                **inputs,
                labels=labels,
            )

            loss = outputs.loss

        loss_val = loss.item()
        test_loss += loss_val

        progress_bar.set_postfix(
            loss=f"{loss_val:.6f}",
        )

        # Save GPU VRAM during model evaluation
        del (
            images,
            graphviz_code,
            inputs,
            labels,
            outputs,
            loss,
            loss_val,
        )

        # torch.cuda.empty_cache()

    return test_loss / len(iterator)


def generate_vlm_outputs(
    model: Sketch2GraphvizVLM,
    iterator: DataLoader,
    instruction: str,
    use_rag: bool = True,
    top_K_rag: int = 5,
    max_new_tokens: int = 1024,
    do_sample: bool = True,
    temperature: float = 1.0,
    skip_special_tokens: bool = True,
    description: str = "Testing",
    outputs_save_path: str = "testing_outputs.jsonl",
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
) -> list[dict]:
    model.eval()
    model.llama_model.eval()
    model.device = device

    progress_bar = tqdm(iterator, desc=description)

    model_outputs = []

    for batch in progress_bar:
        images, graphviz_codes = batch["images"], batch["graphviz_code"]

        if isinstance(images, torch.Tensor):
            images = images.to(device)  # shape: (batch_size, 3, H, W)

        for i in range(len(graphviz_codes)):
            img, graphviz_code = images[i], graphviz_codes[i]

            augmented_instruction = instruction

            if use_rag:
                with autocast(
                    device_type="cuda",
                    dtype=torch.float16,
                    enabled=(device.type == "cuda"),
                ), torch.inference_mode():
                    embedding_vector = model.embed_images(
                        images=(img if isinstance(img, torch.Tensor) else [img])
                    )  # shape: (1, d_model)

                embedding_query_vector = (
                    embedding_vector.cpu().squeeze(dim=0).numpy().astype("float32")
                )

                # Obtain the top-K most similar Graphviz DOT codes by Euclidean (L2) distance between embedding vectors
                vector_similarity_results = get_top_k_similar_vectors_from_db(
                    embedding_vector=embedding_query_vector,
                    top_K=top_K_rag,
                    dbname="sketch2graphvizdb",
                    user="root",
                    table_name="graphviz_embeddings",
                )

                rag_examples = []

                for i, (row_id, rag_graphviz_code, embedding_distance) in enumerate(
                    vector_similarity_results
                ):
                    rag_examples.append(
                        f"Similar Graph Example {i + 1}, distance={embedding_distance:.4f}:\n{rag_graphviz_code}"
                    )

                if rag_examples:
                    rag_instruction_block = "\n\n".join(rag_examples)

                    # Add RAG instructions and top-K RAG examples to the instruction prompt
                    augmented_instruction = f"""{instruction}
                    
Below are DOT code examples for graphs that are similar to the current image.
Use them ONLY as references for structure, style, and typical Graphviz patterns.
Do NOT copy them literally. Instead, carefully inspect the CURRENT image and write new DOT code that exactly matches this image.

{rag_instruction_block}

Based ONLY on the CURRENT image (using the examples only as hints), output the EXACT Graphviz DOT code for this image.
Output ONLY valid DOT code, starting with 'digraph' or 'graph', with no explanations or extra text.
                    """

            with autocast(
                device_type="cuda", dtype=torch.float16, enabled=(device.type == "cuda")
            ), torch.inference_mode():
                sequences = model.generate(
                    images=img,
                    prompts=[augmented_instruction],
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    temperature=temperature,
                    skip_special_tokens=skip_special_tokens,
                )

            model_outputs.append(
                {
                    "id": str(uuid.uuid4()),
                    "original_graphviz_code": graphviz_code,
                    "generated_graphviz_code": sequences[0],
                }
            )

    # Save model outputs to .jsonl file
    pd.DataFrame(model_outputs).to_json(
        outputs_save_path, orient="records", lines=True, force_ascii=False
    )

    print(f"Outputted {len(model_outputs)} model generations")

    return model_outputs


def evaluate_vlm_outputs(
    description: str = "Evaluating Outputs",
    outputs_load_path: str = "testing_outputs.jsonl",
) -> dict:
    # Load the model outputs
    outputs_df = pd.read_json(outputs_load_path, lines=True)
    model_outputs = outputs_df.to_dict(orient="records")

    progress_bar = tqdm(model_outputs, desc=description)

    render_success = 0
    render_fail = 0

    ssim_scores = []
    lpips_distances = []
    networkx_isomorphisms = []

    node_f1_scores = []
    edge_f1_scores = []
    node_attribute_f1_scores = []
    edge_attribute_f1_scores = []

    lpips_loss_fn = lpips.LPIPS(net="alex").eval()

    for item in progress_bar:
        original_graphviz_code, generated_graphviz_code = (
            item["original_graphviz_code"],
            item["generated_graphviz_code"],
        )

        try:
            original_graph = render_graphviz_dot_code_pil(original_graphviz_code)
            generated_graph = render_graphviz_dot_code_pil(generated_graphviz_code)

            render_success += 1
        except Exception:
            render_fail += 1
            progress_bar.set_postfix(render_fail=render_fail)
            continue

        original_graph_np = np.asarray(original_graph, dtype=np.uint8)
        generated_graph_np = np.asarray(generated_graph, dtype=np.uint8)

        # Learned Perceptual Image Patch Similarity (LPIPS)

        # Normalize graph imgages to [-1, 1]
        original_lpips_tensor = (
            torch.from_numpy(original_graph_np)
            .permute(2, 0, 1)
            .unsqueeze(dim=0)
            .float()
            / (255.0 / 2)
            - 1.0
        )
        generated_lpips_tensor = (
            torch.from_numpy(generated_graph_np)
            .permute(2, 0, 1)
            .unsqueeze(dim=0)
            .float()
            / (255.0 / 2)
            - 1.0
        )

        lpips_distance = lpips_loss_fn(original_lpips_tensor, generated_lpips_tensor)
        lpips_distances.append(lpips_distance.item())

        # Structural Similarity Index (SSIM)
        ssim_score = structural_similarity(
            original_graph_np,
            generated_graph_np,
            channel_axis=-1,
            data_range=255,
        )
        ssim_scores.append(float(ssim_score))

        # Graph comparisions with Networkx
        try:
            original_networkx_graph = convert_graphviz_dot_to_networkx(
                original_graphviz_code
            )
            generated_networkx_graph = convert_graphviz_dot_to_networkx(
                generated_graphviz_code
            )

            networkx_graph_isomorphism = nx.is_isomorphic(
                original_networkx_graph, generated_networkx_graph
            )
            networkx_isomorphisms.append(int(networkx_graph_isomorphism))

            graph_f1_metrics = compute_networkx_graph_f1_metrics(
                original_graph=original_networkx_graph,
                generated_graph=generated_networkx_graph,
            )

            node_f1_scores.append(graph_f1_metrics["node_f1"])
            edge_f1_scores.append(graph_f1_metrics["edge_f1"])
            node_attribute_f1_scores.append(graph_f1_metrics["node_attribute_f1"])
            edge_attribute_f1_scores.append(graph_f1_metrics["edge_attribute_f1"])
        except Exception:
            networkx_isomorphisms.append(0)
            node_f1_scores.append(0.0)
            edge_f1_scores.append(0.0)
            node_attribute_f1_scores.append(0.0)
            edge_attribute_f1_scores.append(0.0)

        progress_bar.set_postfix(
            ssim=(sum(ssim_scores) / len(ssim_scores)),
            lpips=(sum(lpips_distances) / len(lpips_distances)),
            isomorphism=(sum(networkx_isomorphisms) / len(networkx_isomorphisms)),
            node_f1=(sum(node_f1_scores) / len(node_f1_scores)),
            edge_f1=(sum(edge_f1_scores) / len(edge_f1_scores)),
            render_fail=render_fail,
        )

    print(f"Evaluated {len(model_outputs)} model generations")

    mean_ssim = float(sum(ssim_scores) / len(ssim_scores))
    mean_lpips = float(sum(lpips_distances) / len(lpips_distances))
    mean_isomorphism = float(sum(networkx_isomorphisms) / len(networkx_isomorphisms))

    mean_node_f1 = float(sum(node_f1_scores) / len(node_f1_scores))
    mean_edge_f1 = float(sum(edge_f1_scores) / len(edge_f1_scores))
    mean_node_attribute_f1 = float(
        sum(node_attribute_f1_scores) / len(node_attribute_f1_scores)
    )
    mean_edge_attribute_f1 = float(
        sum(edge_attribute_f1_scores) / len(edge_attribute_f1_scores)
    )

    render_success_rate = float(render_success / len(model_outputs))

    return {
        "render_success": render_success,
        "render_fail": render_fail,
        "render_success_rate": render_success_rate,
        "mean_ssim": mean_ssim,
        "mean_lpips": mean_lpips,
        "mean_isomorphism": mean_isomorphism,
        "mean_node_f1": mean_node_f1,
        "mean_edge_f1": mean_edge_f1,
        "mean_node_attribute_f1": mean_node_attribute_f1,
        "mean_edge_attribute_f1": mean_edge_attribute_f1,
        "ssim_scores": ssim_scores,
        "lpips_distances": lpips_distances,
        "networkx_isomorphisms": networkx_isomorphisms,
        "node_f1_scores": node_f1_scores,
        "edge_f1_scores": edge_f1_scores,
        "node_attribute_f1_scores": node_attribute_f1_scores,
        "edge_attribute_f1_scores": edge_attribute_f1_scores,
    }


if __name__ == "__main__":
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")
    login(token=hf_token)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size = 1

    train_dataloader, test_dataloader = get_json_graphviz_json_dataloaders(
        json_path="simple_synthetic_data_gen.json",
        batch_size=batch_size,
        root_dir="graphviz_rendered_json",
        image_size=(768, 768),
        return_tensor=False,
    )

    model = load_sketch2graphviz_vlm(
        model_load_dir="checkpoints",
        epoch_load=1,
        quantization="16-bit",
        is_training=False,
        device=device,
    )

    test_loss = evaluate_vlm(
        model=model,
        iterator=test_dataloader,
        instruction=graphviz_code_from_image_instruction,
        description="Testing",
        device=device,
    )

    print(f"Test loss: {test_loss:.6f}")
