import os
from dotenv import load_dotenv
import torch
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from tqdm.auto import tqdm
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from huggingface_hub import login

from scripts.model import Sketch2GraphvizVLM, print_num_params
from scripts.data import get_graphviz_hf_dataloaders, make_inputs_and_labels
from scripts.eval import evaluate_vlm


def add_lora_to_llama(
    model: Sketch2GraphvizVLM, rank: int = 16, alpha: int = 32, dropout: float = 0.05
) -> Sketch2GraphvizVLM:
    llama_model = model.llama_model

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

    return model


def add_lora_to_vit(
    model: Sketch2GraphvizVLM, rank: int = 16, alpha: int = 32, dropout: float = 0.05
) -> Sketch2GraphvizVLM:
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
    model.vit_model = vit_model

    return model


def finetune_vlm_lora(
    model: Sketch2GraphvizVLM,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    instruction: str,
    rank: int = 16,
    lr_vit: float = 1e-5,
    lr_lora: float = 2e-4,
    lr_proj: float = 1e-4,
    weight_decay: float = 1e-2,
    num_epochs: int = 1,
    max_grad_norm: float = 1.0,
    model_save_dir: str = "checkpoints",
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
) -> tuple[Sketch2GraphvizVLM, list[float], list[float]]:
    # Train ViT
    for param in model.vit_model.parameters():
        param.requires_grad = True

    # Train MLP projection
    for param in model.vit_to_llama_projection.parameters():
        param.requires_grad = True

    model = add_lora_to_llama(model, rank=rank, alpha=(2 * rank), dropout=0.05)
    model = add_lora_to_vit(model, rank=rank, alpha=(2 * rank), dropout=0.05)
    print_num_params(model)

    # Optimizer over trainable params
    trainable_params = [param for param in model.parameters() if param.requires_grad]

    optimizer = torch.optim.AdamW(
        [
            {"params": model.vit_model.parameters(), "lr": lr_vit},
            {"params": model.vit_to_llama_projection.parameters(), "lr": lr_proj},
            {
                "params": [
                    param
                    for name, param in model.named_parameters()
                    if param.requires_grad and "llama_model" in name and "lora_" in name
                ],
                "lr": lr_lora,
            },
        ],
        weight_decay=weight_decay,
    )

    scaler = GradScaler(enabled=True)

    model.train()
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        train_loss = 0.0
        progress_bar = tqdm(train_dataloader, desc=f"Training epoch {epoch + 1}")

        for batch in progress_bar:
            images = batch["images"].to(device)  # shape: (batch_size, 3, H, W)
            graphviz_code = batch["graphviz_code"]

            optimizer.zero_grad()

            with autocast(device_type="cuda", dtype=torch.float16):
                inputs_embeds, attention_mask, labels = make_inputs_and_labels(
                    model=model,
                    images=images,
                    graphviz_code=graphviz_code,
                    instruction=instruction,
                )

                outputs = model.llama_model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
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
                attention_mask,
                labels,
                outputs,
                loss,
            )

            torch.cuda.empty_cache()

        os.makedirs(model_save_dir, exist_ok=True)

        # Save LoRA for LLaMA
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

    return model, train_losses, val_losses


if __name__ == "__main__":
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")
    login(token=hf_token)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size = 4

    lr_vit = 1e-5
    lr_lora = 2e-4
    lr_proj = 1e-4
    weight_decay = 1e-2
    num_epochs = 1

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

    model.llama_model.gradient_checkpointing_enable()
    model.llama_model.config.use_cache = False
    model.llama_model.enable_input_require_grads()

    instruction = (
        "You are a compiler that converts images of Graphviz diagrams into their exact Graphviz DOT code. "
        "Given the image, output only the DOT code, starting with either 'digraph' or 'graph', with no explanations, no markdown, and no extra text.\n"
    )

    lora_rank = 16

    model, train_losses, val_losses = finetune_vlm_lora(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        instruction=instruction,
        rank=lora_rank,
        lr_vit=lr_vit,
        lr_lora=lr_lora,
        lr_proj=lr_proj,
        weight_decay=weight_decay,
        num_epochs=num_epochs,
        model_save_dir="checkpoints",
        device=device,
    )
