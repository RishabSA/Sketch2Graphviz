import os
import random
import json
import time
from typing import Callable
from PIL import Image
from tqdm.auto import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn.functional as F
from datasets import Dataset as HFDataset

from scripts.graphviz_renderer import render_graphviz_dot_code
from scripts.model import Sketch2GraphvizVLM


class HandDrawnWhiteBoardTransform:
    def __init__(
        self,
        p: float = 1.0,
        edge_threshold: tuple[float, float] = (0.005, 0.015),
        line_strength: tuple[float, float] = (1.25, 1.75),
    ):
        self.p = p
        self.edge_threshold = edge_threshold
        self.line_strength = line_strength

        transform = [
            transforms.RandomApply(
                [transforms.GaussianBlur(kernel_size=3, sigma=(0.2, 0.7))], p=1.0
            ),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=(1.0, 1.25), contrast=(0.75, 0.95))],
                p=1.0,
            ),
            transforms.RandomAutocontrast(p=0.9),
            transforms.RandomAdjustSharpness(sharpness_factor=2.0, p=0.75),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.01, 0.01),
                scale=(0.995, 1.005),
                fill=(255, 255, 255),
            ),
            transforms.ToTensor(),
            transforms.Lambda(self.overlay_marker_edges),
            transforms.Lambda(self.add_grain),
            transforms.ToPILImage(),
            transforms.RandomApply(
                [transforms.GaussianBlur(kernel_size=3, sigma=(0.05, 0.2))], p=0.5
            ),
        ]

        self.pipeline = transforms.Compose(transform)

    @staticmethod
    def soft_edge_map(gray: torch.Tensor) -> torch.Tensor:
        # gray shape: (1, H, W)

        sobel_convolution_x = torch.tensor(
            [
                [
                    [-1.0, 0.0, 1.0],
                    [-2.0, 0.0, 2.0],
                    [-1.0, 0.0, 1.0],
                ]
            ],
            dtype=gray.dtype,
            device=gray.device,
        ).unsqueeze(dim=0)

        sobel_convolution_y = torch.tensor(
            [
                [
                    [-1.0, -2.0, -1.0],
                    [0.0, 0.0, 0.0],
                    [1.0, 2.0, 1.0],
                ]
            ],
            dtype=gray.dtype,
            device=gray.device,
        ).unsqueeze(dim=0)

        image_gradient_x = F.conv2d(
            gray.unsqueeze(dim=0), sobel_convolution_x, padding=1
        )
        image_gradient_y = F.conv2d(
            gray.unsqueeze(dim=0), sobel_convolution_y, padding=1
        )

        mag = torch.sqrt(image_gradient_x**2 + image_gradient_y**2).squeeze(
            dim=0
        )  # shape: (1, H, W)

        return mag / mag.max()

    def overlay_marker_edges(self, tensor_img: torch.Tensor) -> torch.Tensor:
        gray = transforms.functional.rgb_to_grayscale(tensor_img, num_output_channels=1)
        edges = self.soft_edge_map(gray)

        threshold = random.uniform(self.edge_threshold[0], self.edge_threshold[1])
        strength = random.uniform(self.line_strength[0], self.line_strength[1])

        alpha = (
            torch.clamp((edges - threshold) / (1.0 - threshold), 0.0, 1.0) * strength
        )
        alpha = torch.clamp(alpha, 0.0, 1.0)

        out = tensor_img * (1.0 - alpha)

        return torch.clamp(out, 0.0, 1.0)

    @staticmethod
    def add_grain(tensor_img: torch.Tensor) -> torch.Tensor:
        sigma = random.uniform(0.01, 0.02)
        noise = torch.randn_like(tensor_img) * sigma

        return torch.clamp(tensor_img + noise, 0.0, 1.0)

    def __call__(self, image: Image.Image) -> Image.Image:
        return self.pipeline(image.convert("RGB"))


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
    json_path: str = "synthetic_data_gen.json",
    batch_size: int = 4,
    root_dir: str = "graphviz_rendered_json",
    image_size: tuple[int, int] = (768, 768),
    return_tensor: bool = False,
    handdrawn_probability: float = 0.30,
) -> tuple[DataLoader, DataLoader]:
    start_time = time.time()

    # Load JSON
    with open(json_path, "r") as file:
        dot_code_list = json.load(file)

    # Wrap data in a HuggingFace dataset
    hf_dataset = HFDataset.from_dict({"graphviz_code": dot_code_list})

    hf_dataset_split = hf_dataset.train_test_split(test_size=0.1, seed=42)
    train_split, test_split = hf_dataset_split["train"], hf_dataset_split["test"]

    standard_transform = transforms.Compose(
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
    )

    handdrawn_transform = HandDrawnWhiteBoardTransform(
        p=1.0,
        edge_threshold=(0.005, 0.015),
        line_strength=(1.25, 1.75),
    )

    transform = transforms.Compose(
        [
            transforms.RandomChoice(
                transforms=[handdrawn_transform, standard_transform],
                p=[handdrawn_probability, 1.0 - handdrawn_probability],
            )
        ]
        + ([transforms.ToTensor()] if return_tensor else [])
    )

    # Construct train and test datasets with rendered and saved Graphviz iamges
    train_dataset = GraphvizImageCodeDataset(
        hf_split=train_split,
        split_name="train",
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
        json_path="synthetic_data_gen.json",
        batch_size=batch_size,
        root_dir="graphviz_rendered_json",
        image_size=(768, 768),
        return_tensor=False,
    )
