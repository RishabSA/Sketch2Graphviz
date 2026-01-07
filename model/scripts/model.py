import os
from dotenv import load_dotenv
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
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
            bnb_config_8bit = BitsAndBytesConfig(load_in_8bit=True)

            self.llama_model = MllamaForConditionalGeneration.from_pretrained(
                llama_model_id,
                device_map="auto",
                quantization_config=bnb_config_8bit,
                low_cpu_mem_usage=True,
            )
        elif quantization == "16-bit":
            self.llama_model = MllamaForConditionalGeneration.from_pretrained(
                llama_model_id,
                device_map="auto",
                dtype=torch.float16,
                low_cpu_mem_usage=True,
            )

    def load_lora_adapter(
        self,
        adapter_dir: str,
        name: str,
        is_trainable: bool = False,
        make_active: bool = True,
    ) -> None:
        if isinstance(self.llama_model, PeftModel):
            self.llama_model.load_adapter(
                model_id=adapter_dir,
                adapter_name=name,
                is_trainable=is_trainable,
                device_map="auto",
                torch_dtype=torch.float16,
            )
        else:
            self.llama_model = PeftModel.from_pretrained(
                model=self.llama_model,
                model_id=adapter_dir,
                adapter_name=name,
                is_trainable=is_trainable,
                device_map="auto",
                torch_dtype=torch.float16,
            )

        if make_active:
            self.llama_model.set_adapter(name)

    def set_active_adapter(self, name: str):
        assert isinstance(
            self.llama_model, PeftModel
        ), "llama_model is not a PeftModel so a LoRA adapter cannot be loaded."

        self.llama_model.set_adapter(name)

    def embed_images(self, images: torch.Tensor | list) -> torch.Tensor:
        batch_size = len(images) if isinstance(images, list) else images.shape[0]

        prompt = "You are an image encoder. Embed this Graphviz diagram for retrieval."

        input_texts = []
        for i in range(batch_size):
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]

            input_text = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=False,
            )

            input_texts.append(input_text)

        with torch.no_grad():
            inputs = self.processor(
                images=images,
                text=input_texts,
                add_special_tokens=False,
                padding=True,
                return_tensors="pt",
            ).to(self.device)

            outputs = self.llama_model(
                **inputs,
                output_hidden_states=True,
                use_cache=False,
            )

            hidden_state = outputs.hidden_states[
                -1
            ]  # shape: (batch_size, seq_len, d_model)

            attention_mask = (
                inputs["attention_mask"].unsqueeze(dim=-1).to(hidden_state.dtype)
            )

            masked_hidden_state = hidden_state * attention_mask

            mean_pooled_vectors = masked_hidden_state.sum(dim=1) / attention_mask.sum(
                dim=1
            )  # shape: (batch_size, d_model)

            # L2 normalization
            mean_pooled_vectors = F.normalize(mean_pooled_vectors, p=2, dim=-1)

        return mean_pooled_vectors.detach().cpu()

    def forward(
        self,
        images: torch.Tensor,
        prompts: list[str],
    ):
        # Use forward for training

        input_texts = []
        for prompt in prompts:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]

            input_text = self.processor.apply_chat_template(
                messages, add_generation_prompt=True
            )
            input_texts.append(input_text)

        inputs = self.processor(
            images=images,
            text=input_texts,
            add_special_tokens=False,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        outputs = self.llama_model(**inputs)

        return outputs

    def generate(
        self,
        images: torch.Tensor | Image.Image | None,
        prompts: list[str],
        max_new_tokens: int = 1024,
        do_sample: bool = True,
        temperature: float = 1.0,
        skip_special_tokens: bool = True,
    ) -> list[str]:
        # Use generate for inference

        input_texts = []
        for prompt in prompts:
            messages = [
                {
                    "role": "user",
                    "content": (
                        [
                            {"type": "image"},
                            {"type": "text", "text": prompt},
                        ]
                        if images
                        else [
                            {"type": "text", "text": prompt},
                        ]
                    ),
                }
            ]

            input_text = self.processor.apply_chat_template(
                messages, add_generation_prompt=True
            )
            input_texts.append(input_text)

        inputs = self.processor(
            images=images,
            text=input_texts,
            add_special_tokens=False,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        prompt_lengths = inputs["attention_mask"].sum(dim=1)  # shape: (batch_size)

        eot_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")

        with torch.inference_mode():
            out = self.llama_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=[self.tokenizer.eos_token_id, eot_id],
            )

            response_only = []
            for i in range(out.size(0)):
                prompt_len = int(prompt_lengths[i].item())
                response_list = out[i, prompt_len:].tolist()

                response_only.append(
                    [token for token in response_list if token != eot_id]
                )

            sequences = self.processor.batch_decode(
                response_only,
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


def save_sketch2graphviz_vlm(
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


def load_sketch2graphviz_vlm(
    model_load_dir: str = "checkpoints",
    epoch_load: int = None,
    quantization: str = "16-bit",
    is_training: bool = False,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
) -> Sketch2GraphvizVLM:
    model = Sketch2GraphvizVLM(
        llama_model_id="meta-llama/Llama-3.2-11B-Vision-Instruct",
        quantization=quantization,
        device=device,
    ).to(device)

    if model.quantization != "16-bit" and is_training:
        model.llama_model.gradient_checkpointing_enable()
        model.llama_model.config.use_cache = False
        model.llama_model.enable_input_require_grads()

    if model_load_dir is not None and epoch_load is not None:
        adapter_dir = os.path.join(model_load_dir, f"epoch_{epoch_load}_vlm_lora")

        model.load_lora_adapter(
            adapter_dir=adapter_dir,
            name=f"epoch_{epoch_load}",
            is_trainable=False,
            make_active=True,
        )

        print(f"Loaded LoRA adapter weights from: {adapter_dir}")

    model.device = device
    model.eval()

    print(f"Successfully loaded Sketch2Graphviz model")

    return model


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

    print_num_params(model)
