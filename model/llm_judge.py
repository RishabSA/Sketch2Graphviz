import os
import json
import time
from dotenv import load_dotenv
import pandas as pd
from tqdm.auto import tqdm

from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential
from pydantic import BaseModel, Field

llm_judge_system_prompt = """
You are a fair evaluator of a batch of Graphviz DOT code examples.

You will receive ground-truth DOT graphs and model-generated DOT graphs.
Your goal is to judge generated graphs against ground truth graphs based on general graph structure, whether the generated graph is broadly correct in meaning, and whether the major structure matches.
Prefer giving credit when the prediction is mostly right.

You will receive a JSON object with this shape:
{
    "examples": [
        {
            "id": <string>,
            "original_graphviz_code": <string>,
            "generated_graphviz_code": <string>
        }
    ]
}

Task:
Evaluate each example independently by comparing each generated DOT to each ground-truth DOT.

Rules:
- Ignore whitespace, line breaks, attribute ordering, and harmless quoting/style differences.
- Prioritize overall semantic and structural similarity over exact syntax.
- Count as valid when most key nodes/edges and the main relationships are present, even if some details are missing or extra.
- Minor mismatches in attributes, naming style, layout hints, or small local structure differences should still be valid.
- If directedness differs but the underlying relationship pattern is still clearly the same, you may still count it as valid.
- When uncertain between invalid and valid for a mostly similar graph, choose valid.
- Do not use one sample to influence another.
- Keep result order identical to input order.

Output format (STRICT):
Return ONLY valid JSON with exactly this shape:
{
    "num_examples": <integer>,
    "num_valid_dot": <integer>
}

No extra keys. No markdown. No commentary.
"""


class RefusalCountResponse(BaseModel):
    num_examples: int = Field(ge=0)
    num_valid_dot: int = Field(ge=0)


def make_foundry_client() -> ChatCompletionsClient:
    endpoint = os.environ["AZURE_INFERENCE_ENDPOINT"]
    api_key = os.environ["AZURE_INFERENCE_CREDENTIAL"]

    return ChatCompletionsClient(
        endpoint=endpoint,
        credential=AzureKeyCredential(api_key),
        model="Llama-3.3-70B-Instruct",
    )


def extract_first_json_object(text: str) -> str | None:
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]

    return None


def score_batch_llama_judge(
    client: ChatCompletionsClient,
    batch: list[dict],
    max_tokens: int = 128,
    num_retries: int = 3,
) -> int:
    payload = {
        "examples": [
            {
                "id": str(item.get("id", "")),
                "original_graphviz_code": item.get("original_graphviz_code", ""),
                "generated_graphviz_code": item.get("generated_graphviz_code", ""),
            }
            for item in batch
        ]
    }
    user_payload = json.dumps(payload, ensure_ascii=False)

    num_attempts = num_retries + 1

    for attempt_idx in range(num_attempts):
        attempt = attempt_idx + 1
        try:
            result = client.complete(
                messages=[
                    SystemMessage(content=llm_judge_system_prompt),
                    UserMessage(content=user_payload),
                ],
                temperature=0.0,
                max_tokens=max_tokens,
            )

            text = (result.choices[0].message.content or "").strip()

            # Parse JSON
            try:
                data = json.loads(text)
            except json.JSONDecodeError:
                json_object = extract_first_json_object(text)
                if json_object is None:
                    raise RuntimeError(f"Judge returned non-JSON:\n{text}")

                data = json.loads(json_object)

            parsed = RefusalCountResponse.model_validate(data)
            if parsed.num_examples != len(batch):
                print(
                    f"Judge returned {parsed.num_examples} examples, but expected {len(batch)}"
                )

            return min(parsed.num_valid_dot, parsed.num_examples, len(batch))
        except Exception as e:
            is_last_attempt = attempt == num_attempts
            print(f"Batch judge attempt {attempt}/{num_attempts} failed: {e}")
            if is_last_attempt:
                print("All batch retries exhausted; counting this batch as 0.")
                return 0

            time.sleep(1.0)

    return 0


def evaluate_llm_judge_vlm_outputs(
    client: ChatCompletionsClient,
    batch_size: int = 32,
    num_retries: int = 3,
    description: str = "Evaluating Outputs with LLM-as-a-Judge",
    outputs_load_path: str = "testing_outputs.jsonl",
) -> dict:
    # Load the model outputs
    outputs_df = pd.read_json(outputs_load_path, lines=True)
    model_outputs = outputs_df.to_dict(orient="records")

    llm_judge_correct = 0
    llm_judge_total = 0

    # Batch the categorical outputs
    batched_outputs = [
        model_outputs[i : i + batch_size]
        for i in range(0, len(model_outputs), batch_size)
    ]

    progress_bar = tqdm(batched_outputs, desc=description)

    for batch in progress_bar:
        num_correct = score_batch_llama_judge(
            client=client,
            batch=batch,
            max_tokens=128,
            num_retries=num_retries,
        )

        llm_judge_correct += num_correct
        llm_judge_total += len(batch)

        progress_bar.set_postfix(
            llm_judge_accuracy=(llm_judge_correct / llm_judge_total),
        )

    print(f"Evaluated {len(model_outputs)} model generations")

    return {
        "llm_judge_correct": llm_judge_correct,
        "llm_judge_total": llm_judge_total,
        "llm_judge_accuracy": llm_judge_correct / llm_judge_total,
    }


if __name__ == "__main__":
    load_dotenv()

    client = make_foundry_client()

    results = evaluate_llm_judge_vlm_outputs(
        client=client,
        batch_size=32,
        num_retries=3,
        description="LLM as a Judge Accuracy",
        outputs_load_path="outputs/testing_outputs_rag.jsonl",
    )

    print(results)
