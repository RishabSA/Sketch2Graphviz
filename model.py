import time
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
from transformers import (
    AutoTokenizer,
    LlamaForCausalLM,
    PreTrainedTokenizerBase,
    CLIPVisionModel,
    CLIPImageProcessor,
    BitsAndBytesConfig,
)


class Sketch2GraphvizVLM(nn.Module):
    def __init__(
        self,
        clip_model_id: str = "openai/clip-vit-large-patch14-336",
        llama_model_id: str = "meta-llama/Llama-3.1-8B",
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

        # CLIP Vision Tower
        self.clip_processor = CLIPImageProcessor.from_pretrained(clip_model_id)
        self.clip_model = CLIPVisionModel.from_pretrained(clip_model_id)
        self.clip_model.to(self.device)
        # self.clip_model.eval()

        # LLama Decoder
        self.llama_tokenizer = AutoTokenizer.from_pretrained(llama_model_id)
        if self.llama_tokenizer.pad_token_id is None:
            self.llama_tokenizer.pad_token = self.llama_tokenizer.eos_token

        if quantization == "4-bit":
            # 4-bit Bits and Bytes config
            bnb_config_4bit = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )

            self.llama_model = LlamaForCausalLM.from_pretrained(
                llama_model_id,
                device_map="auto",
                quantization_config=bnb_config_4bit,
                low_cpu_mem_usage=True,
            )
        elif quantization == "8-bit":
            self.llama_model = LlamaForCausalLM.from_pretrained(
                llama_model_id,
                device_map="auto",
                load_in_8bit=True,
                low_cpu_mem_usage=True,
            )
        elif quantization == "16-bit":
            self.llama_model = LlamaForCausalLM.from_pretrained(
                llama_model_id,
                device_map="auto",
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
            )

        clip_hidden_size = self.clip_model.config.hidden_size  # 1024
        llama_hidden_size = self.llama_model.config.hidden_size  # 4096

        self.clip_to_llama_projection = nn.Linear(
            in_features=clip_hidden_size, out_features=llama_hidden_size
        )

    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        # images shape: (batch_size, 3, 336, 336)

        clip_inputs = self.clip_processor(
            images=images,
            return_tensors="pt",
        )
        pixel_values = clip_inputs["pixel_values"].to(
            self.device
        )  # shape: (batch_size, 3, 336, 336)

        # Freeze CLIP
        with torch.no_grad():
            clip_outputs = self.clip_model(pixel_values=pixel_values)

            clip_last_hidden_state = (
                clip_outputs.last_hidden_state
            )  # shape: (batch_size, 1 + num_pathces, d_clip)

        # Only keep patch tokens
        clip_patch_tokens = clip_last_hidden_state[
            :, 1:, :
        ]  # shape: (batch_size, num_pathces, d_clip)

        clip_tokens = self.clip_to_llama_projection(
            clip_patch_tokens
        )  # shape: (batch_size, num_patches, d_llama)

        return clip_tokens

    def forward(
        self,
        images: torch.Tensor,
        prompts: list[str],
    ):
        # images shape: (batch_size, 3, 336, 336)
        # Use forward for training (only returns single Llama output)

        clip_tokens = self.encode_images(
            images
        )  # shape: (batch_size, num_patches, d_llama)

        llama_inputs = self.llama_tokenizer(
            prompts,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

        llama_input_ids = llama_inputs.input_ids  # shape: (batch_size, seq_len)
        llama_attention_mask = (
            llama_inputs.attention_mask
        )  # shape: (batch_size, seq_len)

        # Get text embeddings from Llama
        text_embeds = self.llama_model.get_input_embeddings()(
            llama_input_ids
        )  # shape: (batch_size, seq_len, d_llama)

        # Concatenate CLIP and Llama embeddings along sequence dim
        inputs_embeds = torch.cat(
            [clip_tokens, text_embeds], dim=1
        )  # shape: (batch_size, num_patches + seq_len, d_llama)

        batch_size, num_patches, d_llama = clip_tokens.shape

        clip_attention_mask = torch.ones(
            batch_size,
            num_patches,
            dtype=llama_attention_mask.dtype,
            device=self.device,
        )  # shape: (batch_size, num_patches)

        full_attention_mask = torch.cat(
            [clip_attention_mask, llama_attention_mask], dim=1
        )  # shape: (batch_size, num_patches + seq_len)

        outputs = self.llama_model(
            inputs_embeds=inputs_embeds,
            attention_mask=full_attention_mask,
            # for training you'd also pass labels aligned to text tokens
        )

        return outputs

    def generate(
        self,
        images: torch.Tensor,
        prompts: list[str],
        max_new_tokens: int = 1000,
        do_sample: bool = True,
        temperature: float = 1.0,
    ) -> list[str]:
        # images shape: (batch_size, 3, 336, 336)
        # Use generate for inference

        with torch.inference_mode():
            clip_tokens = self.encode_images(
                images
            )  # shape: (batch_size, num_patches, d_llama)

            llama_inputs = self.llama_tokenizer(
                prompts,
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to(self.device)

            llama_input_ids = llama_inputs.input_ids  # shape: (batch_size, seq_len)
            llama_attention_mask = (
                llama_inputs.attention_mask
            )  # shape: (batch_size, seq_len)

            # Get text embeddings from Llama
            text_embeds = self.llama_model.get_input_embeddings()(
                llama_input_ids
            )  # shape: (batch_size, seq_len, d_llama)

            # Concatenate CLIP and Llama embeddings along sequence dim
            inputs_embeds = torch.cat(
                [clip_tokens, text_embeds], dim=1
            )  # shape: (batch_size, num_patches + seq_len, d_llama)

            batch_size, num_patches, d_llama = clip_tokens.shape

            clip_attention_mask = torch.ones(
                batch_size,
                num_patches,
                dtype=llama_attention_mask.dtype,
                device=self.device,
            )  # shape: (batch_size, num_patches)

            full_attention_mask = torch.cat(
                [clip_attention_mask, llama_attention_mask], dim=1
            )  # shape: (batch_size, num_patches + seq_len)

            # Generate from combined embeddings
            generated = self.llama_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=full_attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                pad_token_id=self.llama_tokenizer.eos_token_id,
            )

            sequences = self.llama_tokenizer.batch_decode(
                generated,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

        return sequences


def get_clip_patch_tokens(
    clip_model: CLIPVisionModel,
    clip_processor: CLIPImageProcessor,
    image_path: str,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
):
    image = Image.open(image_path).convert("RGB")
    inputs = clip_processor(images=image, return_tensors="pt")
    pixel_values = inputs.pixel_values.to(device)  # shape: (1, 3, 336, 336)

    # Forward pass through CLIP vision tower
    with torch.inference_mode():
        outputs = clip_model(pixel_values=pixel_values)
        # last_hidden_state: [batch, 1 + num_patches, hidden_dim]
        last_hidden = (
            outputs.last_hidden_state
        )  # shape: (batch_size, 1 + num_patches, d_model) = (1, 577, 1024)

    # First token is the CLS token, the others are all patch tokens
    cls_token = last_hidden[:, 0, :]  # shape: (batch_size, d_model), d_model = 1024
    patch_tokens = last_hidden[
        :, 1:, :
    ]  # shape: (batch_size, num_patches, d_model), num_patches = (336 * 336) / (14 * 14) = 576, d_model = 1024

    return cls_token, patch_tokens


def load_decoder_model(
    model_id: str = "meta-llama/Llama-3.1-8B",
) -> tuple[LlamaForCausalLM, PreTrainedTokenizerBase]:
    start_time = time.time()

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    model = LlamaForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    model.eval()
    # model.to(device)

    end_time = time.time()
    print(f"Model download time: {(end_time - start_time):.4f} seconds")

    return model, tokenizer


def generate_model_response(
    model: LlamaForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    stop_tokens: list[str] = [],
    max_new_tokens: int = 512,
    do_sample: bool = True,
    temperature: float = 1.0,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
) -> str:
    stop_ids = [tokenizer.eos_token_id]
    stop_ids.extend([tokenizer.convert_tokens_to_ids(token) for token in stop_tokens])

    inputs = tokenizer(prompt, padding=True, truncation=True, return_tensors="pt").to(
        device
    )

    out = model.generate(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=stop_ids,
    )

    sequences = tokenizer.batch_decode(
        out,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )

    return sequences[0]


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    prompt = (
        "You are a compiler that converts images of Graphviz diagrams into their exact Graphviz DOT code. "
        "Given the image, output only the DOT code, starting with either 'digraph' or 'graph', with no explanations, no markdown, and no extra text.\n"
    )

    model = Sketch2GraphvizVLM(
        clip_model_id="openai/clip-vit-large-patch14-336",
        llama_model_id="meta-llama/Llama-3.1-8B",
        quantization="4-bit",
        device=device,
    )

    graphviz_image = Image.open("graphs/graph_1.png").convert("RGB")
    graphviz_image_tensor = (
        transforms.ToTensor()(graphviz_image).unsqueeze(dim=0).to(device)
    )  # shape: (1, 3, 336, 336)

    sequences = model.generate(
        images=graphviz_image_tensor,
        prompts=[prompt],
        max_new_tokens=1000,
        do_sample=True,
        temperature=1.0,
    )

    print(sequences[0])

    # clip_model = CLIPVisionModel.from_pretrained(
    #     "openai/clip-vit-large-patch14-336"
    # ).to(device)
    # clip_model.eval()

    # clip_processor = CLIPImageProcessor.from_pretrained(
    #     "openai/clip-vit-large-patch14-336"
    # )

    # cls, patches = get_clip_patch_tokens(
    #     clip_model=clip_model,
    #     clip_processor=clip_processor,
    #     image_path="graphs/graph_1.png",
    #     device=device,
    # )

    # print("CLS token shape:", cls.shape)
    # print("Patch tokens shape:", patches.shape)
