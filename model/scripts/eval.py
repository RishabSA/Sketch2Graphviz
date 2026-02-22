import os
import uuid
from dotenv import load_dotenv
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.amp import autocast
from tqdm.auto import tqdm
from huggingface_hub import login

from scripts.graphviz_renderer import render_graphviz_dot_code_pil
from scripts.model import Sketch2GraphvizVLM, load_sketch2graphviz_vlm
from scripts.psql_vector_db import get_top_k_similar_vectors_from_db
from scripts.data import (
    get_json_graphviz_json_dataloaders,
    make_inputs_and_labels_vlm,
)
from scripts.prompts import graphviz_code_from_image_instruction


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

        torch.cuda.empty_cache()

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
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
) -> dict:
    # Load the model outputs
    outputs_df = pd.read_json(outputs_load_path, lines=True)
    model_outputs = outputs_df.to_dict(orient="records")

    progress_bar = tqdm(model_outputs, desc=description)

    for item in progress_bar:
        original_graphviz_code, generated_graphviz_code = (
            item["original_graphviz_code"],
            item["generated_graphviz_code"],
        )

    print(f"Evaluated {len(model_outputs)} model generations")

    return {}


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
