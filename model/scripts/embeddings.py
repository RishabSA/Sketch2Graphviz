import os
import numpy as np
from dotenv import load_dotenv
import torch
from torch.utils.data import DataLoader
from torch.amp import autocast
from tqdm.auto import tqdm
from huggingface_hub import login

from scripts.model import Sketch2GraphvizVLM
from scripts.data import get_json_graphviz_json_dataloaders


def get_graphviz_image_embeddings(
    model: Sketch2GraphvizVLM,
    dataloader: DataLoader,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
) -> list[tuple[str, np.ndarray]]:

    model.eval()
    model.llama_model.eval()

    all_codes_and_embedding_vectors = []

    for batch in tqdm(dataloader, desc="Collecting Graphviz Image Embeddings"):
        images = batch["images"]
        graphviz_codes = batch["graphviz_code"]

        if isinstance(images, torch.Tensor):
            images = images.to(device)  # shape: (batch_size, 3, H, W)

        with autocast(
            device_type="cuda", dtype=torch.float16, enabled=(device.type == "cuda")
        ), torch.inference_mode():
            embedding_vectors = model.embed_images(
                images=images
            )  # shape: (batch_size, d_model)

        embedding_vectors = embedding_vectors.cpu().numpy().astype("float32")

        for graphviz_code, embedding_vector in zip(graphviz_codes, embedding_vectors):
            all_codes_and_embedding_vectors.append((graphviz_code, embedding_vector))

        del (
            images,
            graphviz_codes,
            embedding_vectors,
        )

        torch.cuda.empty_cache()

    print(
        f"Collected {len(all_codes_and_embedding_vectors)} Graphviz code and embedding vector pairs"
    )

    return all_codes_and_embedding_vectors


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
        image_size=(768, 768),  # (1024, 1024)
    )

    model = Sketch2GraphvizVLM(
        llama_model_id="meta-llama/Llama-3.2-11B-Vision-Instruct",
        quantization="16-bit",
        device=device,
    )

    if model.quantization != "16-bit":
        model.llama_model.gradient_checkpointing_enable()
        model.llama_model.config.use_cache = False
        model.llama_model.enable_input_require_grads()

    all_codes_and_embedding_vectors = get_graphviz_image_embeddings(
        model=model,
        dataloader=train_dataloader,
        device=device,
    )
