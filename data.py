import os
from typing import Callable
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from datasets import load_dataset, concatenate_datasets

from graphviz_renderer import render_graphviz_dot_code


class GraphvizImageCodeDataset(Dataset):
    def __init__(
        self,
        hf_split,
        split_name: str,
        root_dir: str = "graphviz_rendered",
        image_size: tuple[int, int] = (336, 336),
        transform: Callable = None,
    ):
        self.hf_split = hf_split
        self.split_name = split_name
        self.root_dir = root_dir
        self.image_size = image_size
        self.transform = transform or transforms.ToTensor()

        self.split_dir = os.path.join(root_dir, split_name)
        os.makedirs(self.split_dir, exist_ok=True)

        # Precompute image paths
        self.image_paths = []
        for i in range(len(self.hf_split)):
            filename = f"{split_name}_{i}.png"
            path = os.path.join(self.split_dir, filename)

            self.image_paths.append(path)

            if not os.path.exists(path):
                graphviz_dot_code = self.hf_split[i]["graphviz_code"]

                render_graphviz_dot_code(
                    dot_code=graphviz_dot_code,
                    name=f"{split_name}_{i}",
                    folder=self.split_dir,
                    size=self.image_size,
                )

    def __len__(self) -> int:
        return len(self.hf_split)

    def __getitem__(self, i: int) -> dict:
        example = self.hf_split[i]
        dot_code = example["graphviz_code"]
        image_path = self.image_paths[i]

        # Load image
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.transform(image)  # shape: (3, H, W)

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
    legal_viz = load_dataset("mizuumi1/LegalViz")
    legal_viz = legal_viz.rename_column("graphviz", "graphviz_code")

    v_gen = load_dataset("vgbench/VGen")
    v_gen = v_gen.rename_column("code", "graphviz_code")

    train_split = concatenate_datasets([legal_viz["train"], v_gen["train"]])
    val_split = concatenate_datasets([legal_viz["validation"]])
    test_split = concatenate_datasets([legal_viz["test"]])

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

    return train_dataloader, val_dataloader, test_dataloader
