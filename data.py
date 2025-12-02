import os
from torch.utils.data import DataLoader
from datasets import load_dataset, concatenate_datasets


def get_graphviz_hf_data(
    batch_size: int = 8,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    legal_viz = load_dataset("mizuumi1/LegalViz")
    v_gen = load_dataset("vgbench/VGen")

    graphviz_hf_training_data = concatenate_datasets(
        [legal_viz["train"], v_gen["train"]]
    )

    graphviz_hf_validation_data = concatenate_datasets([legal_viz["validation"]])

    graphviz_hf_testing_data = concatenate_datasets([legal_viz["test"]])

    graphviz_hf_training_dataloader = DataLoader(
        graphviz_hf_training_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=os.cpu_count(),
        pin_memory=True,
        collate_fn=None,
    )

    graphviz_hf_validation_dataloader = DataLoader(
        graphviz_hf_validation_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=os.cpu_count(),
        pin_memory=True,
        collate_fn=None,
    )

    graphviz_hf_testing_dataloader = DataLoader(
        graphviz_hf_testing_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=os.cpu_count(),
        pin_memory=True,
        collate_fn=None,
    )

    return (
        graphviz_hf_training_dataloader,
        graphviz_hf_validation_dataloader,
        graphviz_hf_testing_dataloader,
    )
