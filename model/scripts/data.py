import os
import time
from typing import Callable
from PIL import Image
from tqdm.auto import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from datasets import load_dataset, concatenate_datasets

from scripts.graphviz_renderer import render_graphviz_dot_code
from scripts.model import Sketch2GraphvizVLM


class GraphvizImageCodeDataset(Dataset):
    def __init__(
        self,
        hf_split,
        split_name: str,
        root_dir: str = "graphviz_rendered",
        image_size: tuple[int, int] = (336, 336),
        transform: Callable | None = None,
    ):
        self.hf_split = hf_split
        self.split_name = split_name
        self.root_dir = root_dir
        self.image_size = image_size
        self.transform = transform or transforms.ToTensor()

        self.split_dir = os.path.join(root_dir, split_name)
        os.makedirs(self.split_dir, exist_ok=True)

        self.valid_indices = []
        self.image_paths = []

        skipped = 0

        for i in tqdm(
            range(len(self.hf_split)),
            desc=f"Rendering Graphviz graphs for split {self.split_name}",
        ):
            filename = f"{split_name}_{i}.png"
            path = os.path.join(self.split_dir, filename)

            # Image exists from a previous run
            if os.path.exists(path):
                self.valid_indices.append(i)
                self.image_paths.append(path)

                continue

            graphviz_dot_code = self.hf_split[i]["graphviz_code"]

            try:
                render_graphviz_dot_code(
                    dot_code=graphviz_dot_code,
                    name=f"{split_name}_{i}",
                    folder=self.split_dir,
                    size=image_size,
                )

                self.valid_indices.append(i)
                self.image_paths.append(path)
            except Exception as e:
                skipped += 1

                print(f"Skipping index {i} due to Graphviz error: {e}")

        print(
            f"Split {self.split_name} | Total: {len(self.hf_split)} | Valid: {len(self.valid_indices)}, Skipped: {skipped}"
        )

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, i: int) -> dict:
        hf_idx = self.valid_indices[i]
        image_path = self.image_paths[i]

        example = self.hf_split[hf_idx]
        dot_code = example["graphviz_code"]

        image = Image.open(image_path).convert("RGB")
        image_tensor = self.transform(image)  # (3, H, W)

        return {
            "image": image_tensor,
            "graphviz_code": dot_code,
            "image_path": image_path,
        }


def get_graphviz_hf_dataloaders(
    batch_size: int = 8,
    root_dir: str = "graphviz_rendered",
    image_size: tuple[int, int] = (336, 336),
) -> tuple[DataLoader, DataLoader, DataLoader]:
    start_time = time.time()

    legal_viz = load_dataset("mizuumi1/LegalViz")
    legal_viz = legal_viz.rename_column("graphviz", "graphviz_code")
    legal_viz_train = legal_viz["train"].filter(
        lambda x: "digraph" in x["graphviz_code"]
    )
    legal_viz_val = legal_viz["validation"].filter(
        lambda x: "digraph" in x["graphviz_code"]
    )
    legal_viz_test = legal_viz["test"].filter(lambda x: "digraph" in x["graphviz_code"])

    v_gen = load_dataset("vgbench/VGen")
    v_gen = v_gen.rename_column("code", "graphviz_code")
    v_gen_train = v_gen["train"].filter(lambda x: "digraph" in x["graphviz_code"])

    diagram_gen_coding = load_dataset(
        "DiagramAgent/DiagramGenBenchmark", "DiagramCoding"
    )
    diagram_gen_coding = diagram_gen_coding.rename_column(
        "reference_answer", "graphviz_code"
    )
    diagram_gen_coding = diagram_gen_coding["test"].filter(
        lambda x: "digraph" in x["graphviz_code"]
    )

    diagram_gen_editing = load_dataset(
        "DiagramAgent/DiagramGenBenchmark", "DiagramEditing"
    )
    diagram_gen_editing = diagram_gen_editing.rename_column(
        "reference_answer", "graphviz_code"
    )
    diagram_gen_editing = diagram_gen_editing["test"].filter(
        lambda x: "digraph" in x["graphviz_code"]
    )

    diagram_gen_generation = load_dataset(
        "DiagramAgent/DiagramGenBenchmark", "DiagramGeneration"
    )
    diagram_gen_generation = diagram_gen_generation.rename_column(
        "reference", "graphviz_code"
    )
    diagram_gen_generation = diagram_gen_generation["test"].filter(
        lambda x: "digraph" in x["graphviz_code"]
    )

    dataset = concatenate_datasets(
        [
            # legal_viz_train,
            v_gen_train,
            diagram_gen_coding,
            diagram_gen_editing,
            diagram_gen_generation,
        ]
    )

    dataset_split = dataset.train_test_split(test_size=0.1, seed=42)
    train_split = dataset_split["train"]
    test_split = dataset_split["test"]

    val_split = concatenate_datasets([legal_viz_val])
    # test_split = concatenate_datasets([legal_viz_test])

    # Image Transforms
    transform = transforms.Compose(
        [
            transforms.ToTensor(),  # shape: (3, H, W) with pixels between [0, 1]
            # Add normalization, random rotation, random color grading, etc
        ]
    )

    train_dataset = GraphvizImageCodeDataset(
        hf_split=train_split,
        split_name="train",
        root_dir=root_dir,
        image_size=image_size,
        transform=transform,
    )

    val_dataset = GraphvizImageCodeDataset(
        hf_split=val_split,
        split_name="validation",
        root_dir=root_dir,
        image_size=image_size,
        transform=transform,
    )

    test_dataset = GraphvizImageCodeDataset(
        hf_split=test_split,
        split_name="test",
        root_dir=root_dir,
        image_size=image_size,
        transform=transform,
    )

    def collate_fn(batch: list[dict]) -> dict:
        images = torch.stack(
            [item["image"] for item in batch], dim=0
        )  # shape: (batch_size, 3, H, W)
        codes = [item["graphviz_code"] for item in batch]
        paths = [item["image_path"] for item in batch]

        return {
            "images": images,
            "graphviz_code": codes,
            "image_path": paths,
        }

    # def collate_fn(batch: list[dict]) -> dict:
    #     images = [item["image"] for item in batch]
    #     codes = [item["graphviz_code"] for item in batch]
    #     paths = [item["image_path"] for item in batch]

    #     return {
    #         "images": images,  # list[Tensor]
    #         "graphviz_code": codes,
    #         "image_path": paths,
    #     }

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=os.cpu_count(),
        pin_memory=True,
        collate_fn=collate_fn,
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=os.cpu_count(),
        pin_memory=True,
        collate_fn=collate_fn,
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=os.cpu_count(),
        pin_memory=True,
        collate_fn=collate_fn,
    )

    end_time = time.time()
    print(f"Data download and setup time: {(end_time - start_time):.4f} seconds")

    return train_dataloader, val_dataloader, test_dataloader


def make_inputs_and_labels(
    model: Sketch2GraphvizVLM,
    images: torch.Tensor,
    graphviz_code: list[str],
    instruction: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    vit_tokens, vit_attention_mask = model.encode_images(images)
    # shapes: (batch_size, num_patches, d_llama), (batch_size, num_patches)
    # OR (depending on tiling)
    # shapes: (batch_size, max_seq_len, d_llama), (batch_size, max_seq_len)

    batch_size, num_vit_tokens, d_llama = vit_tokens.shape

    eot_token = "<|eot_id|>"
    prompts = [instruction + code + eot_token for code in graphviz_code]

    llama_inputs = model.llama_tokenizer(
        prompts,
        padding=True,
        truncation=True,
        max_length=1024,
        return_tensors="pt",
    ).to(model.device)

    llama_input_ids = llama_inputs.input_ids  # shape: (batch_size, seq_len)
    llama_attention_mask = llama_inputs.attention_mask  # shape: (batch_size, seq_len)

    llama_output_labels = llama_input_ids.clone()  # shape: (batch_size, seq_len)

    # Get text embeddings from Llama
    text_embeds = model.llama_model.get_input_embeddings()(
        llama_input_ids
    )  # shape: (batch_size, seq_len, d_llama)

    # Concatenate ViT and Llama embeddings along sequence dim
    inputs_embeds = torch.cat(
        [vit_tokens, text_embeds], dim=1
    )  # shape: (batch_size, num_vit_tokens + seq_len, d_llama)

    full_attention_mask = torch.cat(
        [vit_attention_mask, llama_attention_mask], dim=1
    )  # shape: (batch_size, num_vit_tokens + seq_len)

    # Use -100 for visual tokens
    labels = torch.full(
        (batch_size, num_vit_tokens + llama_input_ids.shape[1]),
        -100,
        dtype=torch.long,
        device=model.device,
    )  # shape: (batch_size, num_vit_tokens + seq_len)

    # Fill with text tokens (ViT tokens are -100)
    labels[:, num_vit_tokens:] = (
        llama_output_labels  # shape: (batch_size, num_vit_tokens + seq_len)
    )

    # Mask instruction tokens so loss is only on code
    instruction_ids = model.llama_tokenizer(
        [instruction], add_special_tokens=False, return_tensors="pt"
    ).input_ids.to(model.device)
    instruction_len = instruction_ids.shape[1]

    labels[:, num_vit_tokens : num_vit_tokens + instruction_len] = -100

    return inputs_embeds, full_attention_mask, labels


if __name__ == "__main__":
    batch_size = 4

    train_dataloader, val_dataloader, test_dataloader = get_graphviz_hf_dataloaders(
        batch_size=batch_size,
        root_dir="graphviz_rendered",
        image_size=None,  # (336, 336)
    )
    print(len(train_dataloader), len(val_dataloader), len(test_dataloader))
