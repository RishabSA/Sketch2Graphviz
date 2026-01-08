import os
from dotenv import load_dotenv
import torch
from torch.utils.data import DataLoader
from torch.amp import autocast
from tqdm.auto import tqdm
from huggingface_hub import login

from scripts.model import Sketch2GraphvizVLM, load_sketch2graphviz_vlm
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

    test_loss = 0.0

    progress_bar = tqdm(iterator, desc=description)

    for batch in progress_bar:
        images = batch["images"]
        graphviz_code = batch["graphviz_code"]

        if isinstance(images, torch.Tensor):
            images = images.to(device)  # shape: (batch_size, 3, H, W)

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
