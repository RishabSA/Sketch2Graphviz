import os
from dotenv import load_dotenv
import torch
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from tqdm.auto import tqdm
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
    PeftModel,
)
from huggingface_hub import login

from scripts.model_clip_llama import (
    CLIPLlamaSketch2GraphvizVLM,
    print_num_params,
)
from scripts.data import (
    get_graphviz_hf_dataloaders,
    make_inputs_and_labels_vlm,
    make_inputs_and_labels_clip_llama_vlm,
)
from scripts.eval import evaluate_vlm


def add_lora_to_llama(
    model: CLIPLlamaSketch2GraphvizVLM,
    rank: int = 32,
    alpha: int = 64,
    dropout: float = 0.05,
) -> CLIPLlamaSketch2GraphvizVLM:
    if isinstance(model.llama_model, PeftModel):
        print("Llama model already has LoRA attached...")
        return model

    llama_model = model.llama_model

    if model.quantization != "16-bit":
        # k-bit training (QLoRA)
        llama_model = prepare_model_for_kbit_training(llama_model)

    target_modules = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]

    llama_lora_config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=target_modules,
        lora_dropout=dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    llama_model = get_peft_model(llama_model, llama_lora_config)
    model.llama_model = llama_model

    print("Llama Model LoRA Trainable Parameters:")
    model.llama_model.print_trainable_parameters()
    print("\n")

    return model


def add_lora_to_vit(
    model: CLIPLlamaSketch2GraphvizVLM,
    rank: int = 32,
    alpha: int = 64,
    dropout: float = 0.05,
) -> CLIPLlamaSketch2GraphvizVLM:
    if isinstance(model.vit_model, PeftModel):
        print("ViT model already has LoRA attached...")
        return model

    vit_model = model.vit_model

    target_modules = ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]

    vit_lora_config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=target_modules,
        lora_dropout=dropout,
        bias="none",
    )

    vit_model = get_peft_model(vit_model, vit_lora_config)

    for name, param in vit_model.named_parameters():
        if "lora_" in name:
            param.requires_grad = True

    model.vit_model = vit_model

    print("ViT Model LoRA Trainable Parameters:")
    model.vit_model.print_trainable_parameters()
    print("\n")

    return model


def train_projection_cross_attention(
    model: CLIPLlamaSketch2GraphvizVLM,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    instruction: str,
    lr_proj: float = 1e-4,
    lr_cross_attention: float = 1e-4,
    weight_decay_proj: float = 5e-2,
    weight_decay_cross_attention: float = 5e-2,
    num_epochs: int = 10,
    use_val_early_stopping: bool = True,
    max_grad_norm: float = 1.0,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
) -> tuple[CLIPLlamaSketch2GraphvizVLM, list[float], list[float]]:
    # Freeze original ViT
    for param in model.vit_model.parameters():
        param.requires_grad = False

    # Freeze original Llama
    for param in model.llama_model.parameters():
        param.requires_grad = False

    # Train projection module
    for param in model.vit_to_llama_projection.parameters():
        param.requires_grad = True

    if model.use_cross_attention:
        # Train Cross-Attention Vision and Text Adapter
        for param in model.image_text_adapter.parameters():
            param.requires_grad = True

    print_num_params(model)

    # Optimizer over trainable params
    trainable_params = [param for param in model.parameters() if param.requires_grad]

    if model.use_cross_attention:
        optimizer = torch.optim.AdamW(
            [
                {
                    "params": model.vit_to_llama_projection.parameters(),
                    "lr": lr_proj,
                    "weight_decay": weight_decay_proj,
                },
                {
                    "params": model.image_text_adapter.parameters(),
                    "lr": lr_cross_attention,
                    "weight_decay": weight_decay_cross_attention,
                },
            ],
        )
    else:
        optimizer = torch.optim.AdamW(
            [
                {
                    "params": model.vit_to_llama_projection.parameters(),
                    "lr": lr_proj,
                    "weight_decay": weight_decay_proj,
                },
            ],
        )

    scaler = GradScaler(enabled=True)

    model.train()

    model.vit_model.eval()
    model.llama_model.eval()
    model.vit_to_llama_projection.train()
    if model.use_cross_attention:
        model.image_text_adapter.train()

    train_losses = []
    val_losses = []

    best_val_loss = float("inf")

    for epoch in range(num_epochs):
        train_loss = 0.0
        progress_bar = tqdm(train_dataloader, desc=f"Training epoch {epoch + 1}")

        for batch in progress_bar:
            images = batch["images"].to(device)  # shape: (batch_size, 3, H, W)
            graphviz_code = batch["graphviz_code"]

            optimizer.zero_grad()

            with autocast(device_type="cuda", dtype=torch.float16):
                inputs_embeds, full_attention_mask, labels = (
                    make_inputs_and_labels_clip_llama_vlm(
                        model=model,
                        images=images,
                        graphviz_code=graphviz_code,
                        instruction=instruction,
                    )
                )

                outputs = model.llama_model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=full_attention_mask,
                    labels=labels,
                )
                loss = outputs.loss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(trainable_params, max_grad_norm)

            scaler.step(optimizer)
            scaler.update()

            loss_val = loss.item()
            train_loss += loss_val

            progress_bar.set_postfix(
                loss=f"{loss_val:.6f}",
            )

            del (
                images,
                graphviz_code,
                inputs_embeds,
                full_attention_mask,
                labels,
                outputs,
                loss,
                loss_val,
            )

            torch.cuda.empty_cache()

        epoch_train_loss = train_loss / len(train_dataloader)

        epoch_val_loss = evaluate_vlm(
            model,
            iterator=val_dataloader,
            instruction=instruction,
            description="Validating",
            model_load_dir=None,
            epoch_load=epoch,
            device=device,
        )

        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)

        print(
            f"Epoch {epoch + 1} | Train loss: {epoch_train_loss:.6f} | Val loss: {epoch_val_loss:.6f}"
        )

        if use_val_early_stopping:
            if epoch_val_loss <= best_val_loss:
                best_val_loss = epoch_val_loss
            else:
                print(f"Early stopping at epoch {epoch + 1}...\n")
                break

    return model, train_losses, val_losses


def finetune_clip_llama_vlm_lora(
    model: CLIPLlamaSketch2GraphvizVLM,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    instruction: str,
    rank: int = 32,
    lora_dropout: float = 0.1,
    lr_vit: float = 1e-5,
    lr_proj: float = 1e-4,
    lr_cross_attention: float = 1e-4,
    lr_lora: float = 2e-4,
    weight_decay_vit: float = 5e-2,
    weight_decay_proj: float = 5e-2,
    weight_decay_cross_attention: float = 5e-2,
    weight_decay_lora: float = 0.0,
    num_epochs: int = 10,
    use_val_early_stopping: bool = True,
    max_grad_norm: float = 1.0,
    model_save_dir: str = "checkpoints",
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
) -> tuple[CLIPLlamaSketch2GraphvizVLM, list[float], list[float]]:
    # Freeze original ViT for LoRA
    for param in model.vit_model.parameters():
        param.requires_grad = False

    # Train projection module
    for param in model.vit_to_llama_projection.parameters():
        param.requires_grad = True

    if model.use_cross_attention:
        # Train Cross-Attention Vision and Text Adapter
        for param in model.image_text_adapter.parameters():
            param.requires_grad = True

    model = add_lora_to_llama(model, rank=rank, alpha=(2 * rank), dropout=lora_dropout)
    model = add_lora_to_vit(model, rank=rank, alpha=(2 * rank), dropout=lora_dropout)
    print_num_params(model)

    # Optimizer over trainable params
    trainable_params = [param for param in model.parameters() if param.requires_grad]

    if model.use_cross_attention:
        optimizer = torch.optim.AdamW(
            [
                {
                    "params": model.vit_model.parameters(),
                    "lr": lr_vit,
                    "weight_decay": weight_decay_vit,
                },
                {
                    "params": model.vit_to_llama_projection.parameters(),
                    "lr": lr_proj,
                    "weight_decay": weight_decay_proj,
                },
                {
                    "params": model.image_text_adapter.parameters(),
                    "lr": lr_cross_attention,
                    "weight_decay": weight_decay_cross_attention,
                },
                {
                    "params": [
                        param
                        for name, param in model.named_parameters()
                        if param.requires_grad
                        and "llama_model" in name
                        and "lora_" in name
                    ],
                    "lr": lr_lora,
                    "weight_decay": weight_decay_lora,
                },
            ],
        )
    else:
        optimizer = torch.optim.AdamW(
            [
                {
                    "params": model.vit_model.parameters(),
                    "lr": lr_vit,
                    "weight_decay": weight_decay_vit,
                },
                {
                    "params": model.vit_to_llama_projection.parameters(),
                    "lr": lr_proj,
                    "weight_decay": weight_decay_proj,
                },
                {
                    "params": [
                        param
                        for name, param in model.named_parameters()
                        if param.requires_grad
                        and "llama_model" in name
                        and "lora_" in name
                    ],
                    "lr": lr_lora,
                    "weight_decay": weight_decay_lora,
                },
            ],
        )

    scaler = GradScaler(enabled=True)

    model.train()

    model.vit_model.train()
    model.llama_model.train()
    model.vit_to_llama_projection.train()
    if model.use_cross_attention:
        model.image_text_adapter.train()

    train_losses = []
    val_losses = []

    best_val_loss = float("inf")

    for epoch in range(num_epochs):
        train_loss = 0.0
        progress_bar = tqdm(train_dataloader, desc=f"Training epoch {epoch + 1}")

        for batch in progress_bar:
            images = batch["images"].to(device)  # shape: (batch_size, 3, H, W)
            graphviz_code = batch["graphviz_code"]

            optimizer.zero_grad()

            with autocast(device_type="cuda", dtype=torch.float16):
                inputs_embeds, full_attention_mask, labels = (
                    make_inputs_and_labels_clip_llama_vlm(
                        model=model,
                        images=images,
                        graphviz_code=graphviz_code,
                        instruction=instruction,
                    )
                )

                outputs = model.llama_model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=full_attention_mask,
                    labels=labels,
                )
                loss = outputs.loss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(trainable_params, max_grad_norm)

            scaler.step(optimizer)
            scaler.update()

            loss_val = loss.item()
            train_loss += loss_val

            progress_bar.set_postfix(
                loss=f"{loss_val:.6f}",
            )

            del (
                images,
                graphviz_code,
                inputs_embeds,
                full_attention_mask,
                labels,
                outputs,
                loss,
                loss_val,
            )

            torch.cuda.empty_cache()

        os.makedirs(model_save_dir, exist_ok=True)

        # Save LoRA for Llama
        model.llama_model.save_pretrained(
            os.path.join(model_save_dir, f"epoch_{epoch + 1}_llama_lora")
        )

        # Save LoRA for ViT
        model.vit_model.save_pretrained(
            os.path.join(model_save_dir, f"epoch_{epoch + 1}_vit_lora")
        )

        # Save ViT to Llama projector
        torch.save(
            model.vit_to_llama_projection.state_dict(),
            os.path.join(model_save_dir, f"epoch_{epoch + 1}_proj.pt"),
        )

        epoch_train_loss = train_loss / len(train_dataloader)

        epoch_val_loss = evaluate_vlm(
            model,
            iterator=val_dataloader,
            instruction=instruction,
            description="Validating",
            model_load_dir=None,
            epoch_load=epoch,
            device=device,
        )

        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)

        print(
            f"Epoch {epoch + 1} | Train loss: {epoch_train_loss:.6f} | Val loss: {epoch_val_loss:.6f}"
        )

        if use_val_early_stopping:
            if epoch_val_loss <= best_val_loss:
                best_val_loss = epoch_val_loss
            else:
                print(f"Early stopping at epoch {epoch + 1}...\n")
                break

    return model, train_losses, val_losses


if __name__ == "__main__":
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")
    login(token=hf_token)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size = 1

    train_dataloader, val_dataloader, test_dataloader = get_graphviz_hf_dataloaders(
        batch_size=batch_size,
        root_dir="graphviz_rendered",
        image_size=(336, 336),  # (672, 672), (1008, 1008), None
    )

    model = CLIPLlamaSketch2GraphvizVLM(
        vit_model_id="openai/clip-vit-large-patch14-336",
        llama_model_id="meta-llama/Llama-3.1-8B-Instruct",
        quantization="4-bit",
        tile_images=False,
        use_cross_attention=False,
        device=device,
    )

    instruction = (
        "You are a compiler that converts images of Graphviz diagrams into their exact Graphviz DOT code. "
        "Given an image of a graph, using only the image, output only the DOT code, starting with either 'digraph' or 'graph', with no explanations, no markdown, and no extra text.\n"
    )

    lora_rank = 16
    lora_dropout = 0.1

    lr_vit = 1e-5  # 5e-6
    lr_proj = 1e-4  # 5e-5
    lr_cross_attention = 1e-4  # 5e-5
    lr_lora = 2e-4  # 1e-4

    weight_decay_vit = 1e-2  # 5e-2
    weight_decay_proj = 1e-2  # 5e-2
    weight_decay_cross_attention = 1e-2  # 5e-2
    weight_decay_lora = 1e-3  # 1e-2

    max_grad_norm = 1.0

    num_epochs = 5

    model, train_losses, val_losses = finetune_clip_llama_vlm_lora(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        instruction=instruction,
        rank=lora_rank,
        lora_dropout=lora_dropout,
        lr_vit=lr_vit,
        lr_proj=lr_proj,
        lr_cross_attention=lr_cross_attention,
        lr_lora=lr_lora,
        weight_decay_vit=weight_decay_vit,
        weight_decay_proj=weight_decay_proj,
        weight_decay_cross_attention=weight_decay_cross_attention,
        weight_decay_lora=weight_decay_lora,
        num_epochs=num_epochs,
        use_val_early_stopping=True,
        max_grad_norm=max_grad_norm,
        model_save_dir="checkpoints",
        device=device,
    )
