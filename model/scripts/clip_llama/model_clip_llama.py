import os
import math
from dotenv import load_dotenv
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from transformers import (
    AutoTokenizer,
    LlamaForCausalLM,
    AutoModel,
    AutoImageProcessor,
    CLIPVisionModel,
    BitsAndBytesConfig,
)
from peft import PeftModel
from huggingface_hub import login

from scripts.prompts import graphviz_code_from_image_instruction


def get_image_tiles(
    image: torch.Tensor, tile_size: int = 336, stride: int | None = None
) -> list:
    if stride is None:
        stride = tile_size

    C, H, W = image.shape

    # Pad to multiple of tile_size so we don't miss edges
    pad_h = (math.ceil(H / tile_size) * tile_size) - H
    pad_w = (math.ceil(W / tile_size) * tile_size) - W

    # pad = (left, right, top, bottom)
    image_padded = F.pad(image, (0, pad_w, 0, pad_h), value=1.0)
    C_pad, H_pad, W_pad = image_padded.shape

    tiles = []
    for top in range(0, H_pad - tile_size + 1, stride):
        for left in range(0, W_pad - tile_size + 1, stride):
            crop = image_padded[:, top : top + tile_size, left : left + tile_size]
            tiles.append(crop)

    return tiles


class ImageTextCrossAttention(nn.Module):
    def __init__(self, d_model: int = 4096, num_heads: int = 8):
        super().__init__()

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            batch_first=True,
        )

        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=0.1)

        # self.vision_scalar = nn.Parameter(torch.zeros(1))
        self.vision_scalar = nn.Parameter(torch.tensor(-5.0))

    def forward(
        self,
        text_embeds: torch.Tensor,
        img_tokens: torch.Tensor,
        img_mask: torch.Tensor = None,
    ):
        # text_embeds shape: (batch_size, seq_len, d_model)
        # vit_tokens shape: (batch_size, num_vit_tokens, d_model)
        # vit_attention_mask shape: (batch_size, num_vit_tokens)

        key_padding_mask = (img_mask == 0) if img_mask is not None else None

        query = self.layer_norm(text_embeds)
        attn_out, _ = self.cross_attn(
            query=query,
            key=img_tokens,
            value=img_tokens,
            key_padding_mask=key_padding_mask,
        )
        attn_out = self.dropout(attn_out)

        # Residual connection
        return text_embeds + torch.sigmoid(self.vision_scalar) * attn_out


class VisionToLlamaProjection(nn.Module):
    def __init__(
        self,
        vit_hidden_size: int = 1024,
        llama_hidden_size: int = 4096,
        dropout_p: float = 0.1,
    ):
        super().__init__()

        self.layer_norm = nn.LayerNorm(vit_hidden_size)

        self.layer_1 = nn.Linear(
            in_features=vit_hidden_size, out_features=llama_hidden_size
        )

        self.activation = nn.GELU()
        self.dropout = nn.Dropout(p=dropout_p)

        self.layer_2 = nn.Linear(
            in_features=llama_hidden_size, out_features=llama_hidden_size
        )

        self.layer_residual = nn.Linear(
            in_features=vit_hidden_size, out_features=llama_hidden_size
        )

    def forward(self, vit_tokens: torch.Tensor):
        # x shape: (batch_size, num_patches, d_vit)

        vit_tokens_norm = self.layer_norm(vit_tokens)

        x = self.layer_1(vit_tokens_norm)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.layer_2(x)

        return x + self.layer_residual(
            vit_tokens
        )  # shape: (batch_size, num_patches, d_llama)


class CLIPLlamaSketch2GraphvizVLM(nn.Module):
    def __init__(
        self,
        vit_model_id: str = "openai/clip-vit-large-patch14-336",
        llama_model_id: str = "meta-llama/Llama-3.1-8B-Instruct",
        quantization: str = "4-bit",  # "4-bit", "8-bit", or "16-bit"
        tile_images: bool = True,
        use_cross_attention: bool = True,
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
        self.tile_images = tile_images
        self.use_cross_attention = use_cross_attention

        # ViT Vision Tower
        self.vit_processor = AutoImageProcessor.from_pretrained(vit_model_id)

        if "clip" in vit_model_id:
            self.vit_model = CLIPVisionModel.from_pretrained(vit_model_id)
        else:
            self.vit_model = AutoModel.from_pretrained(vit_model_id)

        self.vit_model.to(self.device, dtype=torch.float16)
        # self.vit_model.eval()

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

        vit_hidden_size = self.vit_model.config.hidden_size  # 1024
        llama_hidden_size = self.llama_model.config.hidden_size  # 4096

        # self.vit_to_llama_projection = nn.Linear(
        #     in_features=vit_hidden_size, out_features=llama_hidden_size
        # )

        # self.vit_to_llama_projection = nn.Sequential(
        #     nn.LayerNorm(vit_hidden_size),
        #     nn.Linear(in_features=vit_hidden_size, out_features=llama_hidden_size),
        #     nn.GELU(),
        #     nn.Dropout(p=0.1),
        #     nn.Linear(in_features=llama_hidden_size, out_features=llama_hidden_size),
        # )

        self.vit_to_llama_projection = VisionToLlamaProjection(
            vit_hidden_size=vit_hidden_size,
            llama_hidden_size=llama_hidden_size,
            dropout_p=0.1,
        )

        self.vit_to_llama_projection.to(device)

        self.image_text_adapter = nn.Identity()

        if use_cross_attention:
            self.image_text_adapter = ImageTextCrossAttention(
                d_model=llama_hidden_size,
                num_heads=self.llama_model.config.num_attention_heads,
            )
            self.image_text_adapter.to(device)

    def embed_images(
        self, images: torch.Tensor | list
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # images shape: (batch_size, 3, 336, 336) OR list of tensors
        llama_dtype = next(self.llama_model.parameters()).dtype

        if self.tile_images:
            batch_size = images.shape[0]
            tile_size = 336

            all_vit_tokens = []
            seq_lens = []

            for batch in range(batch_size):
                img = images[batch]  # shape: (3, H, W)

                # Tile the image into (3, tile_size, tile_size)
                tiles = get_image_tiles(image=img, tile_size=tile_size, stride=None)
                num_tiles = len(tiles)

                # Run all tiles through the ViT processor and ViT model
                vit_inputs = self.vit_processor(images=tiles, return_tensors="pt")

                pixel_values = vit_inputs["pixel_values"].to(
                    self.device, dtype=llama_dtype
                )  # shape: (num_tiles, 3, 336, 336)

                vit_outputs = self.vit_model(pixel_values=pixel_values)

                vit_last_hidden_state = (
                    vit_outputs.last_hidden_state
                )  # shape: (num_tiles, 1 + num_patches, d_vit)

                # Only keep patch tokens (drop CLS token)
                vit_patch_tokens = vit_last_hidden_state[
                    :, 1:, :
                ]  # shape: (num_tiles, num_patches, d_vit)

                # Flatten tiles and patches
                num_tiles, num_patches, d_vit = vit_patch_tokens.shape

                vit_patch_tokens = vit_patch_tokens.reshape(
                    num_tiles * num_patches, d_vit
                )  # (num_tiles * num_patches, d_vit)

                # project to LLaMA hidden size
                vit_tokens = self.vit_to_llama_projection(
                    vit_patch_tokens.to(dtype=llama_dtype)
                )  # (num_tiles * num_patches, d_llama)

                all_vit_tokens.append(vit_tokens)
                seq_lens.append(vit_tokens.shape[0])

            # Pad to max sequence length across batch so we can form a proper tensor
            max_seq_len = max(seq_lens)
            d_llama = all_vit_tokens[0].shape[1]

            vit_tokens = torch.zeros(
                batch_size,
                max_seq_len,
                d_llama,
                device=self.device,
                dtype=llama_dtype,
            )

            vit_attention_mask = torch.zeros(
                batch_size,
                max_seq_len,
                device=self.device,
                dtype=torch.long,
            )

            for batch in range(batch_size):
                seq_len = all_vit_tokens[batch].shape[0]  # num_tiles * num_patches

                vit_tokens[batch, :seq_len, :] = all_vit_tokens[batch]
                vit_attention_mask[batch, :seq_len] = 1

            return (
                vit_tokens,
                vit_attention_mask,
            )  # shapes: (batch_size, max_seq_len, d_llama), (batch_size, max_seq_len)

        else:
            vit_inputs = self.vit_processor(
                images=images,
                return_tensors="pt",
            )

            pixel_values = vit_inputs["pixel_values"].to(
                self.device, dtype=llama_dtype
            )  # shape: (batch_size, 3, 336, 336)

            vit_outputs = self.vit_model(pixel_values=pixel_values)

            vit_last_hidden_state = (
                vit_outputs.last_hidden_state
            )  # shape: (batch_size, 1 + num_patches, d_vit)

            # Only keep patch tokens (drop CLS token)
            vit_patch_tokens = vit_last_hidden_state[
                :, 1:, :
            ]  # shape: (batch_size, num_patches, d_vit)

            vit_tokens = self.vit_to_llama_projection(
                vit_patch_tokens
            )  # shape: (batch_size, num_patches, d_llama)

            batch_size, num_patches, d_llama = vit_tokens.shape

            vit_attention_mask = torch.ones(
                batch_size,
                num_patches,
                device=self.device,
                dtype=torch.long,
            )

            return (
                vit_tokens,
                vit_attention_mask,
            )  # shapes: (batch_size, num_patches, d_llama), (batch_size, num_patches)

    def forward(
        self,
        images: torch.Tensor,
        prompts: list[str],
    ):
        # images shape: (batch_size, 3, 336, 336)
        # Use forward for training (only returns single Llama output)

        vit_tokens, vit_attention_mask = self.embed_images(images)
        # shapes: (batch_size, num_patches, d_llama), (batch_size, num_patches)
        # OR (depending on tiling)
        # shapes: (batch_size, max_seq_len, d_llama), (batch_size, max_seq_len)

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

        if self.use_cross_attention:
            # Fuse ViT and Llama embeddings with Cross-Attention
            inputs_embeds = self.image_text_adapter(
                text_embeds,  # Query
                vit_tokens,  # Key/Value
                vit_attention_mask,
            )  # shape: (batch_size, seq_len, d_model)

            full_attention_mask = llama_attention_mask  # shape: (batch_size, seq_len)
        else:
            # Concatenate ViT and Llama embeddings along sequence dim
            inputs_embeds = torch.cat(
                [vit_tokens, text_embeds], dim=1
            )  # shape: (batch_size, num_vit_tokens + seq_len, d_llama)

            full_attention_mask = torch.cat(
                [vit_attention_mask, llama_attention_mask], dim=1
            )  # shape: (batch_size, num_vit_tokens + seq_len)

        outputs = self.llama_model(
            inputs_embeds=inputs_embeds,
            attention_mask=full_attention_mask,
            # labels=labels,
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
        # images shape: (batch_size, 3, 336, 336)
        # Use generate for inference

        with torch.inference_mode():
            vit_tokens, vit_attention_mask = self.embed_images(images)
            # shapes: (batch_size, num_patches, d_llama), (batch_size, num_patches)
            # OR (depending on tiling)
            # shapes: (batch_size, max_seq_len, d_llama), (batch_size, max_seq_len)

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

            if self.use_cross_attention:
                # Fuse ViT and Llama embeddings with Cross-Attention
                inputs_embeds = self.image_text_adapter(
                    text_embeds,  # Query
                    vit_tokens,  # Key/Value
                    vit_attention_mask,
                )  # shape: (batch_size, seq_len, d_model)

                full_attention_mask = (
                    llama_attention_mask  # shape: (batch_size, seq_len)
                )
            else:
                # Concatenate ViT and Llama embeddings along sequence dim
                inputs_embeds = torch.cat(
                    [vit_tokens, text_embeds], dim=1
                )  # shape: (batch_size, num_vit_tokens + seq_len, d_llama)

                full_attention_mask = torch.cat(
                    [vit_attention_mask, llama_attention_mask], dim=1
                )  # shape: (batch_size, num_vit_tokens + seq_len)

            eot_id = self.llama_tokenizer.convert_tokens_to_ids("<|end_of_text|>")

            # Generate from combined embeddings
            generated = self.llama_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=full_attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                pad_token_id=self.llama_tokenizer.eos_token_id,
                eos_token_id=[self.llama_tokenizer.eos_token_id, eot_id],
            )

            sequences = self.llama_tokenizer.batch_decode(
                generated,
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


def save_sketch2graphviz_clip_llama_vlm_local(
    model: CLIPLlamaSketch2GraphvizVLM,
    model_save_dir: str = "checkpoints",
    epoch_save: int | None = None,
) -> None:
    assert (
        model_save_dir is not None and epoch_save is not None
    ), "You must pass in values for model_save_dir and epoch_save"

    os.makedirs(model_save_dir, exist_ok=True)

    # Save LoRA for LLaMA
    llama_lora_dir = os.path.join(model_save_dir, f"epoch_{epoch_save + 1}_llama_lora")
    model.llama_model.save_pretrained(llama_lora_dir)

    print(f"Saved Llama LoRA to: {llama_lora_dir}")

    # Save LoRA for ViT
    vit_lora_dir = os.path.join(model_save_dir, f"epoch_{epoch_save + 1}_vit_lora")
    model.vit_model.save_pretrained(vit_lora_dir)

    print(f"Saved ViT LoRA to: {vit_lora_dir}")

    # Save ViT to Llama projector
    projector_dir = os.path.join(model_save_dir, f"epoch_{epoch_save + 1}_proj.pt")
    torch.save(model.vit_to_llama_projection.state_dict(), projector_dir)

    print(f"Saved Projector to: {projector_dir}")


def load_sketch2graphviz_clip_llama_vlm_local(
    model_load_dir: str = "checkpoints",
    epoch_load: int | None = None,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
) -> CLIPLlamaSketch2GraphvizVLM:
    assert (
        model_load_dir is not None and epoch_load is not None
    ), "You must pass in values for model_load_dir and epoch_load"

    model = CLIPLlamaSketch2GraphvizVLM(
        vit_model_id="openai/clip-vit-large-patch14-336",
        llama_model_id="meta-llama/Llama-3.1-8B-Instruct",
        quantization="4-bit",
        tile_images=False,
        use_cross_attention=False,
        device=device,
    ).to(device)

    if model.quantization != "16-bit":
        model.llama_model.gradient_checkpointing_enable()
        model.llama_model.config.use_cache = False
        model.llama_model.enable_input_require_grads()

    # Load projector weights
    projector_path = os.path.join(model_load_dir, f"epoch_{epoch_load}_proj.pt")
    model.vit_to_llama_projection.load_state_dict(
        torch.load(projector_path, map_location=device)
    )

    # Load Llama LoRA adapter
    llama_lora_dir = os.path.join(model_load_dir, f"epoch_{epoch_load}_llama_lora")
    model.llama_model = PeftModel.from_pretrained(
        model.llama_model,
        llama_lora_dir,
        device_map="auto",
        torch_dtype=torch.float16,
    )

    # Load ViT LoRA adapter
    vit_lora_dir = os.path.join(model_load_dir, f"epoch_{epoch_load}_vit_lora")
    model.vit_model = PeftModel.from_pretrained(
        model_load_dir="checkpoints",
        epoch_load=6,
        device=device,
    )

    model.vit_to_llama_projection.to(device)
    model.llama_model.to(device)
    model.vit_model.to(device)
    model.device = device
    model.eval()

    print(f"Loaded Llama LoRA from: {llama_lora_dir}")
    print(f"Loaded ViT LoRA from: {vit_lora_dir}")
    print(f"Loaded Projector from: {projector_path}")

    return model


if __name__ == "__main__":
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")
    login(token=hf_token)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CLIPLlamaSketch2GraphvizVLM(
        vit_model_id="openai/clip-vit-large-patch14-336",
        llama_model_id="meta-llama/Llama-3.1-8B-Instruct",
        quantization="4-bit",
        tile_images=False,
        use_cross_attention=False,
        device=device,
    )

    model.llama_model.gradient_checkpointing_enable()
    model.llama_model.config.use_cache = False
    model.llama_model.enable_input_require_grads()

    print_num_params(model)

    graphviz_image = Image.open("testing_graphs/graph_1.png").convert("RGB")
    graphviz_image_tensor = (
        transforms.ToTensor()(graphviz_image).unsqueeze(dim=0).to(device)
    )  # shape: (1, 3, 336, 336)

    sequences = model.generate(
        images=graphviz_image_tensor,
        prompts=[graphviz_code_from_image_instruction],
        max_new_tokens=2048,
        do_sample=False,
        temperature=1.0,
        skip_special_tokens=True,
    )

    print(sequences[0])
