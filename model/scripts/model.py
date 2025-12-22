import os
from dotenv import load_dotenv
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from transformers import (
    BitsAndBytesConfig,
    MllamaForConditionalGeneration,
    AutoProcessor,
)
from peft import PeftModel
from huggingface_hub import login


class Sketch2GraphvizVLM(nn.Module):
    def __init__(
        self,
        llama_model_id: str = "meta-llama/Llama-3.2-11B-Vision-Instruct",
        quantization: str = "4-bit",  # "4-bit", "8-bit", or "16-bit"
        device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
    ):
        super().__init__()

        assert quantization in [
            "4-bit",
            "8-bit",
            "16-bit",
        ], "quantization must be set to either 4-bit, 8-bit, or 16-bit"

        self.device = device
        self.quantization = quantization

        self.processor = AutoProcessor.from_pretrained(llama_model_id)
        self.tokenizer = self.processor.tokenizer

        if quantization == "4-bit":
            # 4-bit Bits and Bytes config
            bnb_config_4bit = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )

            self.llama_model = MllamaForConditionalGeneration.from_pretrained(
                llama_model_id,
                device_map="auto",
                quantization_config=bnb_config_4bit,
                low_cpu_mem_usage=True,
            )
        elif quantization == "8-bit":
            self.llama_model = MllamaForConditionalGeneration.from_pretrained(
                llama_model_id,
                device_map="auto",
                load_in_8bit=True,
                low_cpu_mem_usage=True,
            )
        elif quantization == "16-bit":
            self.llama_model = MllamaForConditionalGeneration.from_pretrained(
                llama_model_id,
                device_map="auto",
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
            )

    def encode_images(self, images: torch.Tensor | list) -> torch.Tensor:
        inputs = self.processor(
            text=[""] * images.shape[0],
            images=images,
            return_tensors="pt",
            padding=True,
        ).to(self.device)

        print(inputs)

        outputs = self.llama_model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            pixel_values=inputs["pixel_values"],
            output_hidden_states=True,
            use_cache=False,
        )

        encoded_image_vectors = outputs.hidden_states[
            -1
        ]  # shape: (batch_size, seq_len, d_model)
        attention_mask = inputs["attention_mask"].bool()  # shape (batch_size, seq_len)

        masked_image_vectors = encoded_image_vectors * attention_mask.unsqueeze(
            dim=-1
        )  # shape: (batch_size, seq_len, d_model)

        masked_image_vectors = masked_image_vectors.reshape(
            masked_image_vectors.shape[0], -1
        )  # shape: (batch_size, seq_len * d_model)

        # L2 normaliztion
        masked_image_vectors = F.normalize(masked_image_vectors, p=2, dim=-1)

        return (
            masked_image_vectors.detach().cpu()
        )  # shape: (batch_size, seq_len * d_model)

    def forward(
        self,
        images: torch.Tensor,
        prompts: list[str],
    ):
        # Use forward for training

        inputs = self.processor(
            text=prompts,
            images=images,
            return_tensors="pt",
            padding=True,
        ).to(self.device)

        print(inputs)

        outputs = self.llama_model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            pixel_values=inputs["pixel_values"],
        )

        return outputs

    def generate(
        self,
        images: torch.Tensor,
        prompts: list[str],
        max_new_tokens: int = 1024,
        do_sample: bool = True,
        temperature: float = 1.0,
        skip_special_tokens: bool = True,
    ) -> list[str]:
        # Use generate for inference

        inputs = self.processor(
            text=prompts,
            images=images,
            return_tensors="pt",
            padding=True,
        ).to(self.device)

        print(inputs)

        eot_id = self.tokenizer.convert_tokens_to_ids("<|end_of_text|>")

        with torch.inference_mode():
            out = self.llama_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=[self.tokenizer.eos_token_id, eot_id],
            )

            sequences = self.tokenizer.batch_decode(
                out,
                skip_special_tokens=skip_special_tokens,
                clean_up_tokenization_spaces=False,
            )

        return sequences


def print_num_params(model: nn.Module) -> None:
    total = sum(param.numel() for param in model.parameters())
    trainable = sum(
        param.numel() for param in model.parameters() if param.requires_grad
    )

    print(
        f"The model has {total:,} total parameters | {trainable:,} trainable parameters | Percent trainable: ({((trainable / total) * 100):.2f}%)"
    )


def save_sketch2graphviz_vlm_local(
    model: Sketch2GraphvizVLM,
    model_save_dir: str = "checkpoints",
    epoch_save: int | None = None,
) -> None:
    assert (
        model_save_dir is not None and epoch_save is not None
    ), "You must pass in values for model_save_dir and epoch_save"

    os.makedirs(model_save_dir, exist_ok=True)

    vlm_lora_dir = os.path.join(model_save_dir, f"epoch_{epoch_save + 1}_vlm_lora")
    model.llama_model.save_pretrained(vlm_lora_dir)

    print(f"Saved VLM LoRA to: {vlm_lora_dir}")


def load_sketch2graphviz_vlm_local(
    model_load_dir: str = "checkpoints",
    epoch_load: int | None = None,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
) -> Sketch2GraphvizVLM:
    assert (
        model_load_dir is not None and epoch_load is not None
    ), "You must pass in values for model_load_dir and epoch_load"

    model = Sketch2GraphvizVLM(
        llama_model_id="meta-llama/Llama-3.2-11B-Vision-Instruct",
        quantization="4-bit",
        device=device,
    ).to(device)

    model.llama_model.gradient_checkpointing_enable()
    model.llama_model.config.use_cache = False
    model.llama_model.enable_input_require_grads()

    vlm_lora_dir = os.path.join(model_load_dir, f"epoch_{epoch_load}_vlm_lora")
    model.llama_model = PeftModel.from_pretrained(
        model.llama_model,
        vlm_lora_dir,
        device_map="auto",
        torch_dtype=torch.float16,
    )

    model.llama_model.to(device)
    model.device = device
    model.eval()

    print(f"Loaded VLM LoRA from: {vlm_lora_dir}")

    return model


if __name__ == "__main__":
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")
    login(token=hf_token)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    instruction = (
        "You are a compiler that converts images of Graphviz diagrams into their exact Graphviz DOT code. "
        "Given an image of a graph, using only the image, output only the DOT code, starting with either 'digraph' or 'graph', with no explanations, no markdown, and no extra text.\n"
    )

    model = Sketch2GraphvizVLM(
        llama_model_id="meta-llama/Llama-3.2-11B-Vision-Instruct",
        quantization="4-bit",
        device=device,
    ).to(device)

    model.llama_model.gradient_checkpointing_enable()
    model.llama_model.config.use_cache = False
    model.llama_model.enable_input_require_grads()

    print_num_params(model)

    graphviz_image = Image.open("testing_graphs/graph_1.png").convert("RGB")
    graphviz_image_tensor = (
        transforms.ToTensor()(graphviz_image).unsqueeze(dim=0).to(device)
    )

    sequences = model.generate(
        images=graphviz_image_tensor,
        prompts=[instruction],
        max_new_tokens=1024,
        do_sample=False,
        temperature=1.0,
        skip_special_tokens=True,
    )

    print(sequences[0])
