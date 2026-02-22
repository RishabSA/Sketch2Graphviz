import os
import json
import time
from typing import Callable
from PIL import Image
from tqdm.auto import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from datasets import Dataset as HFDataset

from scripts.graphviz_renderer import render_graphviz_dot_code
from scripts.model import Sketch2GraphvizVLM


class GraphvizImageCodeDataset(Dataset):
    def __init__(
        self,
        hf_split,
        split_name: str,
        root_dir: str = "graphviz_rendered",
        image_size: tuple[int, int] = (768, 768),
        transform: Callable | None = None,
    ):
        self.hf_split = hf_split
        self.split_name = split_name
        self.root_dir = root_dir
        self.image_size = image_size
        self.transform = transform

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

            # Skip image if it exists from a previous render
            if os.path.exists(path):
                self.valid_indices.append(i)
                self.image_paths.append(path)

                continue

            graphviz_dot_code = self.hf_split[i]["graphviz_code"]

            try:
                # Render and resize Graphviz image
                render_graphviz_dot_code(
                    dot_code=graphviz_dot_code,
                    name=f"{split_name}_{i}",
                    folder=self.split_dir,
                    size=image_size,
                )

                # Only store the valid/successfully rendered Graphviz images
                self.valid_indices.append(i)
                self.image_paths.append(path)
            except Exception as e:
                # If an error occurs while rendering a Graphviz image, skip it
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

        if self.transform:
            image = self.transform(image)

        return {
            "image": image,
            "graphviz_code": dot_code,
            "image_path": image_path,
        }


def get_json_graphviz_json_dataloaders(
    json_path: str = "simple_synthetic_data_gen.json",
    batch_size: int = 4,
    root_dir: str = "graphviz_rendered_json",
    image_size: tuple[int, int] = (768, 768),
    return_tensor: bool = False,
) -> tuple[DataLoader, DataLoader]:
    start_time = time.time()

    # Load JSON
    with open(json_path, "r") as file:
        dot_code_list = json.load(file)

    # Wrap data in a HuggingFace dataset
    hf_dataset = HFDataset.from_dict({"graphviz_code": dot_code_list})

    hf_dataset_split = hf_dataset.train_test_split(test_size=0.1, seed=42)
    train_split, test_split = hf_dataset_split["train"], hf_dataset_split["test"]

    # Image Transforms for training
    train_transform = transforms.Compose(
        [
            transforms.RandomApply(
                [
                    # Slight random brightness and contrast changes
                    transforms.ColorJitter(brightness=0.2, contrast=0.2)
                ],
                p=0.3,
            ),
            # Slight random translations and scaling
            transforms.RandomAffine(
                degrees=0,
                translate=(0.03, 0.03),
                scale=(0.97, 1.03),
                fill=(255, 255, 255),
            ),
            transforms.RandomApply(
                # Slight random blurring
                [transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5))],
                p=0.1,
            ),
        ]
        + ([transforms.ToTensor()] if return_tensor else [])
    )

    test_transform = transforms.ToTensor() if return_tensor else None

    # Construct train and test datasets with rendered and saved Graphviz iamges
    train_dataset = GraphvizImageCodeDataset(
        hf_split=train_split,
        split_name="train",
        root_dir=root_dir,
        image_size=image_size,
        transform=train_transform,
    )

    test_dataset = GraphvizImageCodeDataset(
        hf_split=test_split,
        split_name="test",
        root_dir=root_dir,
        image_size=image_size,
        transform=test_transform,
    )

    def collate_fn(batch: list[dict]) -> dict:
        if isinstance(batch[0]["image"], torch.Tensor):
            images = torch.stack(
                [item["image"] for item in batch], dim=0
            )  # shape: (batch_size, 3, H, W)
        else:
            images = [item["image"] for item in batch]

        codes = [item["graphviz_code"] for item in batch]
        paths = [item["image_path"] for item in batch]

        return {
            "images": images,
            "graphviz_code": codes,
            "image_path": paths,
        }

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=os.cpu_count(),
        persistent_workers=True,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=os.cpu_count(),
        persistent_workers=True,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    end_time = time.time()
    print(f"JSON data load and render time: {(end_time - start_time):.4f} seconds")

    return train_dataloader, test_dataloader


def make_inputs_and_labels_vlm(
    model: Sketch2GraphvizVLM,
    images: torch.Tensor,
    graphviz_code: list[str],
    instruction: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    eot_token = "<|eot_id|>"

    prefix_message = [
        {
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": instruction}],
        }
    ]

    prefix = model.processor.apply_chat_template(
        prefix_message, add_generation_prompt=True
    )

    # Simulate full instruction + generated graphviz code for full text output
    full_texts = [prefix + code + eot_token for code in graphviz_code]

    # Process full text sequences and images
    inputs = model.processor(
        images=images,
        text=full_texts,
        max_length=2048,
        padding=True,
        truncation=True,
        add_special_tokens=False,
        return_tensors="pt",
    ).to(model.device)

    labels = inputs["input_ids"].clone()

    prefix_ids = model.tokenizer(prefix, add_special_tokens=False).input_ids
    prefix_len = len(prefix_ids)

    # Mask the prefix for every sample and only keep generated tokens for labels
    seq_len = labels.shape[1]
    mask_len = min(prefix_len, seq_len)
    labels[:, :mask_len] = -100

    # Mask all padding tokens
    labels[inputs["attention_mask"] == 0] = -100

    return inputs, labels


if __name__ == "__main__":
    batch_size = 1

    train_dataloader, test_dataloader = get_json_graphviz_json_dataloaders(
        json_path="simple_synthetic_data_gen.json",
        batch_size=batch_size,
        root_dir="graphviz_rendered_json",
        image_size=(768, 768),
        return_tensor=False,
    )
