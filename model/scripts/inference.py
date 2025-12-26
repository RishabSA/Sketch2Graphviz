import os
from dotenv import load_dotenv
from PIL import Image
import torch
import torch.nn as nn
from torch.amp import autocast
from huggingface_hub import login

from scripts.model import Sketch2GraphvizVLM, load_sketch2graphviz_vlm_local
from scripts.psql_vector_db import get_top_k_similar_vectors_from_db


def predict_graphviz_dot(
    model: nn.Module,
    image: str | Image.Image | torch.Tensor,
    instruction: str,
    should_print_instruction: bool = False,
    use_rag: bool = True,
    top_K_rag: int = 5,
    max_new_tokens: int = 1024,
    do_sample: bool = True,
    temperature: float = 1.0,
    skip_special_tokens: bool = True,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
) -> str:
    model.eval()
    model.llama_model.eval()
    model.device = device

    if isinstance(image, str):
        # image is a file path
        img = Image.open(image).convert("RGB")
    elif isinstance(image, Image.Image):
        img = image.convert("RGB")
    elif isinstance(image, torch.Tensor):
        if image.dim() == 3:
            img = image.unsqueeze(dim=0)
        elif image.dim() == 4:
            img = image
        else:
            raise ValueError("Image must have shape (3, H, W) or (1, 3, H, W)")
    else:
        raise TypeError("image must be a file path, Image.Image, or torch.Tensor")

    augmented_instruction = instruction

    if use_rag:
        with autocast(
            device_type="cuda", dtype=torch.float16, enabled=(device.type == "cuda")
        ), torch.inference_mode():
            embedding_vector = model.embed_images(
                images=(img if isinstance(img, torch.Tensor) else [img])
            )  # shape: (1, d_model)

        embedding_query_vector = (
            embedding_vector.cpu().squeeze(dim=0).numpy().astype("float32")
        )

        vector_similarity_results = get_top_k_similar_vectors_from_db(
            embedding_vector=embedding_query_vector,
            top_K=top_K_rag,
            dbname="sketch2graphvizdb",
            table_name="graphviz_embeddings",
        )

        rag_examples = []

        for i, (row_id, graphviz_code, embedding_distance) in enumerate(
            vector_similarity_results
        ):
            rag_examples.append(
                f"Similar Graph Example {i + 1}, distance={embedding_distance:.4f}:\n{graphviz_code}"
            )

        if rag_examples:
            rag_instruction_block = "\n\n".join(rag_examples)

            augmented_instruction = f"""{instruction}
            
Below are DOT code examples for graphs that are similar to the current image.
Use them ONLY as references for structure, style, and typical Graphviz patterns.
Do NOT copy them literally. Instead, carefully inspect the CURRENT image and write new DOT code that exactly matches this image.

{rag_instruction_block}

Based ONLY on the CURRENT image (using the examples only as hints), output the EXACT Graphviz DOT code for this image.
Output ONLY valid DOT code, starting with 'digraph' or 'graph', with no explanations or extra text.
            """

    if should_print_instruction:
        print(f"Instruction: {augmented_instruction}\n")

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

    raw_output = sequences[0]
    return raw_output


if __name__ == "__main__":
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")
    login(token=hf_token)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Sketch2GraphvizVLM(
        llama_model_id="meta-llama/Llama-3.2-11B-Vision-Instruct",
        quantization="16-bit",
        device=device,
    )

    if model.quantization != "16-bit":
        model.llama_model.gradient_checkpointing_enable()
        model.llama_model.config.use_cache = False
        model.llama_model.enable_input_require_grads()

    model = load_sketch2graphviz_vlm_local(
        model_load_dir="checkpoints",
        epoch_load=10,
        quantization="16-bit",
        device=device,
    )

    instruction = (
        "You are a compiler that converts images of Graphviz diagrams into their exact Graphviz DOT code. "
        "Given an image of a graph, using only the image, output only the DOT code, starting with either 'digraph' or 'graph', with no explanations, no markdown, and no extra text.\n"
    )

    predicted_graphviz_output = predict_graphviz_dot(
        model=model,
        image="testing_graphs/graph_1.png",
        instruction=instruction,
        should_print_instruction=False,
        use_rag=False,
        top_K_rag=5,
        max_new_tokens=2048,
        do_sample=True,
        temperature=1.0,
        device=device,
    )

    print(predicted_graphviz_output)
