import os
from dotenv import load_dotenv
from PIL import Image
import torch
from torch.amp import autocast
from torchvision import transforms
from huggingface_hub import login

from scripts.model import Sketch2GraphvizVLM, load_sketch2graph_vlm


def predict_graphviz_dot(
    model: Sketch2GraphvizVLM,
    image: str | Image.Image | torch.Tensor,
    instruction: str,
    max_new_tokens: int = 1024,
    do_sample: bool = True,
    temperature: float = 1.0,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
) -> str:
    model.eval()
    model.device = device

    if isinstance(image, str):
        # image is a file path
        img = Image.open(image).convert("RGB")
        img = (
            transforms.ToTensor()(img).unsqueeze(dim=0).to(device, dtype=torch.float16)
        )
    elif isinstance(image, Image.Image):
        img = image.convert("RGB")
        img = (
            transforms.ToTensor()(img).unsqueeze(dim=0).to(device, dtype=torch.float16)
        )
    elif isinstance(image, torch.Tensor):
        if image.dim() == 3:
            img = image.unsqueeze(dim=0).to(device, dtype=torch.float16)
        elif image.dim() == 4:
            img = image.to(device, dtype=torch.float16)
        else:
            raise ValueError("Image must have shape (3, H, W) or (1, 3, H, W)")
    else:
        raise TypeError("image must be a file path, Image.Image, or torch.Tensor")

    with autocast(device_type="cuda", dtype=torch.float16), torch.inference_mode():
        sequences = model.generate(
            images=img,
            prompts=[instruction],
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
        )

    raw_output = sequences[0]
    return raw_output


if __name__ == "__main__":
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")
    login(token=hf_token)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Sketch2GraphvizVLM(
        vit_model_id="openai/clip-vit-large-patch14-336",
        llama_model_id="meta-llama/Llama-3.1-8B",
        quantization="4-bit",
        device=device,
    )

    model.llama_model.gradient_checkpointing_enable()
    model.llama_model.config.use_cache = False
    model.llama_model.enable_input_require_grads()

    model = load_sketch2graph_vlm(
        model=model,
        model_load_dir="checkpoints",
        epoch_load=10,
        device=device,
    )

    instruction = (
        "You are a compiler that converts images of Graphviz diagrams into their exact Graphviz DOT code. "
        "Given the image, output only the DOT code, starting with either 'digraph' or 'graph', with no explanations, no markdown, and no extra text.\n"
    )

    predicted_graphviz_output = predict_graphviz_dot(
        model=model,
        image="graphs/graph_1.png",
        instruction=instruction,
        max_new_tokens=1024,
        do_sample=True,
        temperature=1.0,
        device=device,
    )

    print(predicted_graphviz_output)
