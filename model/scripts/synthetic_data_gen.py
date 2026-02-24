import os
import sys
import json
from openai import OpenAI
from dotenv import load_dotenv
from pydantic import BaseModel

from prompts import (
    get_synthethic_data_gen_simple_system_prompt,
    get_synthethic_data_gen_complex_system_prompt,
    synthethic_data_gen_simple_prompt_suffixes,
    synthethic_data_gen_complex_prompt_suffixes,
)


class GraphvizData(BaseModel):
    dot_codes: list[str]


def generate_graphviz_code(
    openai_client: OpenAI,
    system_prompt: str,
    prompt_suffix: str,
    model_name: str = "gpt-5-mini",
    temperature: float = 1.0,
    batch_size: int = 50,
) -> list[str]:
    try:
        # Pydantic JSON-based outputs for batched Graphviz dot codes
        response = openai_client.responses.parse(
            model=model_name,
            input=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": f"Can you generate {batch_size} Graphviz code samples to be used as labeled data to help train an AI agent to learn how Graphviz works?\n"
                    + prompt_suffix,
                },
            ],
            temperature=temperature,
            text_format=GraphvizData,
        )

        return response.output_parsed.dot_codes

    except Exception as e:
        print(f"An unexpected error occurred while generating synthetic data: {e}")
        return []


if __name__ == "__main__":
    load_dotenv()

    openai_api_key = os.getenv("OPENAI_API_KEY")

    dot_codes = []

    # json_data_file_path = "data/simple_synthetic_data_gen.json"
    json_data_file_path = "data/complex_synthetic_data_gen.json"

    os.makedirs(os.path.dirname(json_data_file_path), exist_ok=True)

    try:
        # Create the OpenAI Client
        openai_client = OpenAI(api_key=openai_api_key)
    except Exception as e:
        print(f"Loading OpenAI model failed: {e}")
        sys.exit(1)

    batch_size = 50

    # system_prompt = get_synthethic_data_gen_simple_system_prompt(batch_size=batch_size)
    system_prompt = get_synthethic_data_gen_complex_system_prompt(batch_size=batch_size)

    # prompt_suffixes = synthethic_data_gen_simple_prompt_suffixes
    prompt_suffixes = synthethic_data_gen_complex_prompt_suffixes

    # Use a new suffix for every batch generation to get different types of generated Graphviz DOT codes
    for i, suffix in enumerate(prompt_suffixes):
        generated_dot_codes = generate_graphviz_code(
            openai_client=openai_client,
            system_prompt=system_prompt,
            prompt_suffix=suffix,
            model_name="gpt-5-mini",
            temperature=1.0,
            batch_size=batch_size,
        )

        if generated_dot_codes:
            dot_codes.extend(generated_dot_codes)

            with open(json_data_file_path, "w") as json_file:
                json.dump(dot_codes, json_file, indent=4)

            print(f"Generated {len(generated_dot_codes)} with suffix: {suffix}")

    print(
        f"Generated {len(dot_codes)} final DOT samples saved to {json_data_file_path}"
    )
