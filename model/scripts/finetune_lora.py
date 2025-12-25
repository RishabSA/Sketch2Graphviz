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

from scripts.model import (
    Sketch2GraphvizVLM,
    print_num_params,
)
from scripts.data import (
    get_graphviz_hf_dataloaders,
    make_inputs_and_labels_vlm,
)
from scripts.eval import evaluate_vlm


def add_lora_to_VLM(
    model: Sketch2GraphvizVLM, rank: int = 32, alpha: int = 64, dropout: float = 0.05
) -> Sketch2GraphvizVLM:
    if isinstance(model.llama_model, PeftModel):
        print("Llama model already has LoRA attached...")
        return model

    llama_model = model.llama_model

    # k-bit training (QLoRA)
    if model.quantization != "16-bit":
        llama_model = prepare_model_for_kbit_training(llama_model)

    target_modules = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "fc1",
        "fc2",
        "multi_modal_projector",
    ]

    lora_config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=target_modules,
        lora_dropout=dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    llama_model = get_peft_model(llama_model, lora_config)
    model.llama_model = llama_model

    print("Llama Model LoRA Trainable Parameters:")
    model.llama_model.print_trainable_parameters()
    print("\n")

    return model


def finetune_vlm_lora(
    model: Sketch2GraphvizVLM,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    instruction: str,
    rank: int = 32,
    lora_dropout: float = 0.1,
    lr_lora: float = 2e-4,
    weight_decay_lora: float = 0.0,
    num_epochs: int = 10,
    use_val_early_stopping: bool = True,
    max_grad_norm: float = 1.0,
    model_save_dir: str = "checkpoints",
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
) -> tuple[Sketch2GraphvizVLM, list[float], list[float]]:
    model = add_lora_to_VLM(model, rank=rank, alpha=(2 * rank), dropout=lora_dropout)
    print_num_params(model)

    trainable_params = [
        param
        for name, param in model.named_parameters()
        if param.requires_grad and "lora_" in name
    ]

    optimizer = torch.optim.AdamW(
        params=trainable_params,
        lr=lr_lora,
        weight_decay=weight_decay_lora,
    )

    scaler = GradScaler(enabled=True)

    model.train()
    model.llama_model.train()

    train_losses = []
    val_losses = []

    best_val_loss = float("inf")

    for epoch in range(num_epochs):
        train_loss = 0.0
        progress_bar = tqdm(train_dataloader, desc=f"Training epoch {epoch + 1}")

        for batch in progress_bar:
            images = batch["images"]
            graphviz_code = batch["graphviz_code"]

            if isinstance(images, torch.Tensor):
                images = images.to(device)  # shape: (batch_size, 3, H, W)

            optimizer.zero_grad()

            inputs, encoded_image_vectors, labels = make_inputs_and_labels_vlm(
                model=model,
                images=images,
                graphviz_code=graphviz_code,
                instruction=instruction,
            )

            with autocast(
                device_type="cuda", dtype=torch.float16, enabled=(device.type == "cuda")
            ):
                outputs = model.llama_model(
                    **inputs,
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
                inputs,
                encoded_image_vectors,
                labels,
                outputs,
                loss,
                loss_val,
            )

            torch.cuda.empty_cache()

        os.makedirs(model_save_dir, exist_ok=True)

        # Save LoRA for Llama
        model.llama_model.save_pretrained(
            os.path.join(model_save_dir, f"epoch_{epoch + 1}_vlm_lora")
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

    lora_rank = 32
    lora_dropout = 0.1

    lr_lora = 2e-4  # 1e-4
    weight_decay_lora = 1e-3  # 1e-2
    max_grad_norm = 1.0

    num_epochs = 10

    model, train_losses, val_losses = finetune_vlm_lora(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=test_dataloader,
        instruction=instruction,
        rank=lora_rank,
        lora_dropout=lora_dropout,
        lr_lora=lr_lora,
        weight_decay_lora=weight_decay_lora,
        num_epochs=num_epochs,
        use_val_early_stopping=True,
        max_grad_norm=max_grad_norm,
        model_save_dir="checkpoints",
        device=device,
    )

    print(f"Train Losses: {train_losses}")
    print(f"Val Losses: {val_losses}")
