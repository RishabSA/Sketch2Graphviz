import os
from dotenv import load_dotenv
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from peft import PeftModel
from huggingface_hub import login

from scripts.model import Sketch2GraphvizVLM
from scripts.data import get_graphviz_hf_dataloaders, make_inputs_and_labels


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
        # Load projector weights
        proj_path = os.path.join(model_load_dir, f"epoch_{epoch_load}_proj.pt")
        model.vit_to_llama_projection.load_state_dict(
            torch.load(proj_path, map_location=device)
        )

        # Load LoRA adapter
        lora_dir = os.path.join(model_load_dir, f"epoch_{epoch_load}_lora")
        model.llama_model = PeftModel.from_pretrained(model.llama_model, lora_dir)

        model.vit_model.to(device)
        model.vit_to_llama_projection.to(device)
        model.device = device

    model.eval()
    test_loss = 0.0

    progress_bar = tqdm(iterator, desc=description)

    for batch in progress_bar:
        images = batch["images"].to(device)
        graphviz_code = batch["graphviz_code"]

        inputs_embeds, attention_mask, labels = make_inputs_and_labels(
            model=model,
            images=images,
            graphviz_code=graphviz_code,
            instruction=instruction,
        )

        with torch.inference_mode():
            outputs = model.llama_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                labels=labels,
            )

        loss = outputs.loss
        test_loss += loss.item()

    return test_loss / len(iterator)


if __name__ == "__main__":
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")
    login(token=hf_token)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size = 16

    train_dataloader, val_dataloader, test_dataloader = get_graphviz_hf_dataloaders(
        batch_size=batch_size,
        root_dir="graphviz_rendered",
        image_size=(336, 336),
    )

    model = Sketch2GraphvizVLM(
        vit_model_id="openai/clip-vit-large-patch14-336",
        llama_model_id="meta-llama/Llama-3.1-8B",
        quantization="4-bit",
        device=device,
    ).to(device)

    instruction = (
        "You are a compiler that converts images of Graphviz diagrams into their exact Graphviz DOT code. "
        "Given the image, output only the DOT code, starting with either 'digraph' or 'graph', with no explanations, no markdown, and no extra text.\n"
    )

    test_loss = evaluate_vlm(
        model=model,
        iterator=test_dataloader,
        instruction=instruction,
        description="Testing",
        model_load_dir="checkpoints",
        epoch=10,
        device=device,
    )

    print(f"Test loss: {test_loss:.6f}")
