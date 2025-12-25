import os
from dotenv import load_dotenv
import torch
from torch.utils.data import DataLoader
from torch.amp import autocast
from tqdm.auto import tqdm
from peft import PeftModel
from huggingface_hub import login

from scripts.model import Sketch2GraphvizVLM
from scripts.data import (
    get_graphviz_hf_dataloaders,
    make_inputs_and_labels_vlm,
)


def evaluate_vlm(
    model: Sketch2GraphvizVLM,
    iterator: DataLoader,
    instruction: str,
    description: str = "Testing",
    model_load_dir: str | None = None,
    epoch_load: int | None = None,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
) -> float:
    if model_load_dir is not None and epoch_load is not None:
        vlm_lora_dir = os.path.join(model_load_dir, f"epoch_{epoch_load}_vlm_lora")
        model.llama_model = PeftModel.from_pretrained(
            model.llama_model,
            vlm_lora_dir,
            device_map="auto",
            torch_dtype=torch.float16,
        )

        model.llama_model.to(device)
        model.device = device

        print(f"Loaded VLM LoRA from: {vlm_lora_dir}")

    model.eval()

    test_loss = 0.0

    progress_bar = tqdm(iterator, desc=description)

    for batch in progress_bar:
        images = batch["images"]
        graphviz_code = batch["graphviz_code"]

        if isinstance(images, torch.Tensor):
            images = images.to(device)  # shape: (batch_size, 3, H, W)

        inputs, encoded_image_vectors, labels = make_inputs_and_labels_vlm(
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
            encoded_image_vectors,
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

    train_dataloader, val_dataloader, test_dataloader = get_graphviz_hf_dataloaders(
        batch_size=batch_size,
        root_dir="graphviz_rendered",
        image_size=(768, 768),  # (512, 512), (1024, 1024)
    )

    model = Sketch2GraphvizVLM(
        llama_model_id="meta-llama/Llama-3.2-11B-Vision-Instruct",
        quantization="16-bit",
        device=device,
    ).to(device)

    if model.quantization != "16-bit":
        model.llama_model.gradient_checkpointing_enable()
        model.llama_model.config.use_cache = False
        model.llama_model.enable_input_require_grads()

    instruction = (
        "You are a compiler that converts images of Graphviz diagrams into their exact Graphviz DOT code. "
        "Given an image of a graph, using only the image, output only the DOT code, starting with either 'digraph' or 'graph', with no explanations, no markdown, and no extra text.\n"
    )

    test_loss = evaluate_vlm(
        model=model,
        iterator=test_dataloader,
        instruction=instruction,
        description="Testing",
        model_load_dir="checkpoints",
        epoch_load=10,
        device=device,
    )

    print(f"Test loss: {test_loss:.6f}")
