import os
from dotenv import load_dotenv
from huggingface_hub import login
import torch
from PIL import Image
from transformers import (
    MllamaForConditionalGeneration,
    MllamaProcessor,
    BitsAndBytesConfig,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, PeftModel

from scripts.data import get_json_graphviz_json_dataset_trainer


def load_base_vlm_4bit(
    model_id: str = "meta-llama/Llama-3.2-11B-Vision-Instruct",
) -> tuple[MllamaForConditionalGeneration, MllamaProcessor]:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    model = MllamaForConditionalGeneration.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
    )

    processor = MllamaProcessor.from_pretrained(model_id)

    model.config.use_cache = False

    return model, processor


def add_lora_to_vlm(
    model: MllamaForConditionalGeneration,
    rank: int = 32,
    alpha: int = 64,
    dropout: float = 0.1,
) -> PeftModel:
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
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model


class GraphvizDataCollator:
    def __init__(
        self,
        processor: MllamaProcessor,
        tokenizer: AutoTokenizer,
        instruction: str,
        max_length: int = 1024,
    ):
        self.processor = processor
        self.tokenizer = tokenizer
        self.instruction = instruction
        self.max_length = max_length

    def __call__(self, batch: list[dict]) -> dict[str, torch.Tensor]:
        images = [item["image"] for item in batch]
        graphviz_codes = [item["graphviz_code"] for item in batch]

        eot_token = "<|eot_id|>"
        full_texts = []
        response_texts = []

        for code in graphviz_codes:
            message = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": self.instruction},
                    ],
                }
            ]

            prefix = self.processor.apply_chat_template(
                message, add_generation_prompt=True
            )

            response = code + eot_token
            full_texts.append(prefix + response)
            response_texts.append(response)

        # Process sequences and images
        inputs = self.processor(
            images=images,
            text=full_texts,
            max_length=2048,
            padding=True,
            truncation=True,
            add_special_tokens=False,
            return_tensors="pt",
        )

        labels = inputs["input_ids"].clone()

        for i, response_text in enumerate(response_texts):
            response_ids = self.tokenizer(
                response_text, add_special_tokens=False, return_tensors="pt"
            ).input_ids

            response_len = response_ids.shape[1]
            total_len = inputs["attention_mask"][i].sum().item()

            # Mask everything from the start up to the beginning of the response
            prefix_len = total_len - response_len
            labels[i, :prefix_len] = -100

        # Mask all padding tokens
        labels[inputs["attention_mask"] == 0] = -100

        inputs["labels"] = labels

        return inputs


def load_vlm_with_lora_for_inference(
    model_id: str = "meta-llama/Llama-3.2-11B-Vision-Instruct",
    output_dir: str = "vlm_lora_checkpoints",
    lora_dir: str = "vlm_lora_adapter",
):
    base_model = MllamaForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    processor = MllamaProcessor.from_pretrained(model_id)

    peft_model_dir = os.path.join(output_dir, lora_dir)
    model = PeftModel.from_pretrained(base_model, peft_model_dir)
    model.eval()
    model.config.use_cache = True

    return model, processor


def generate_vlm_model_response(
    model,
    processor,
    images: torch.Tensor | Image.Image,
    prompts: list[str],
    max_new_tokens: int = 1024,
    do_sample: bool = True,
    temperature: float = 1.0,
    skip_special_tokens: bool = True,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
) -> list[str]:
    input_texts = []
    for prompt in prompts:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
        input_texts.append(input_text)

    inputs = processor(
        images=images,
        text=input_texts,
        add_special_tokens=False,
        padding=True,
        return_tensors="pt",
    ).to(device)

    eot_id = processor.tokenizer.convert_tokens_to_ids("<|eot_id|>")

    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            pad_token_id=processor.tokenizer.eos_token_id,
            eos_token_id=[processor.tokenizer.eos_token_id, eot_id],
        )

        sequences = processor.batch_decode(
            out,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=False,
        )

    return sequences


if __name__ == "__main__":
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")
    login(token=hf_token)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_model, processor = load_base_vlm_4bit(
        model_id="meta-llama/Llama-3.2-11B-Vision-Instruct"
    )

    lora_rank = 32
    lora_dropout = 0.1
    model = add_lora_to_vlm(
        base_model,
        rank=lora_rank,
        alpha=2 * lora_rank,
        dropout=lora_dropout,
    )

    train_dataset, test_dataset = get_json_graphviz_json_dataset_trainer(
        json_path="simple_synthetic_data_gen.json",
        root_dir="graphviz_rendered_json",
        image_size=(768, 768),  # (512, 512), (1024, 1024)
    )

    instruction = (
        "You are a compiler that converts images of Graphviz diagrams into their exact Graphviz DOT code. "
        "Given an image of a graph, using only the image, output only the DOT code, starting with either 'digraph' or 'graph', with no explanations, no markdown, and no extra text.\n"
    )

    data_collator = GraphvizDataCollator(
        processor=processor,
        tokenizer=model.tokenizer,
        instruction=instruction,
        max_length=1024,
    )

    output_dir = "vlm_lora_checkpoints"

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,  # effective batch size = 4
        num_train_epochs=3,
        learning_rate=2e-4,
        weight_decay=0.01,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=50,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        save_total_limit=2,
        bf16=False,
        fp16=True,
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=data_collator,
    )

    trainer.train()

    # Save LoRA adapter weights
    peft_output_dir = os.path.join(output_dir, "vlm_lora_adapter")
    os.makedirs(peft_output_dir, exist_ok=True)
    model.save_pretrained(peft_output_dir)
    processor.save_pretrained(peft_output_dir)

    print(f"LoRA adapter and processor saved to: {peft_output_dir}")
