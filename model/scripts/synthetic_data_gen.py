import os
import sys
import json
from openai import OpenAI
from dotenv import load_dotenv
from pydantic import BaseModel


class GraphvizData(BaseModel):
    dot_codes: list[str]


def generate_simple_graphviz_code(
    openai_client: OpenAI,
    model_name: str = "gpt-5-mini",
    prompt_suffix: str = "",
    temperature: float = 1.0,
    batch_size: int = 32,
) -> list[str]:
    try:
        response = openai_client.responses.parse(
            model=model_name,
            input=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert in the Graphviz DOT language. "
                        "You are generating *training data* for a vision-language model that "
                        "learns to map diagram images (PNGs) to their corresponding DOT code.\n\n"
                        "Requirements:\n"
                        "- Each sample must be standalone, valid Graphviz DOT code.\n"
                        "- Use either 'graph' or 'digraph' as appropriate.\n"
                        "- Use only ASCII characters and no comments.\n"
                        "- Do NOT include any explanations, markdown, prose, or backticks—only raw DOT code.\n"
                        "- Keep each graph relatively simple (1 - 5 nodes) but with some variety in structure "
                        "(chains, stars, small DAGs, small undirected graphs, etc.).\n"
                        "- Avoid attributes that depend on a specific layout engine; basic node/edge attributes are fine."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Can you generate {batch_size} simple Graphviz code samples to be used as labeled data to help train an AI agent to learn how Graphviz works?"
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
    json_data_file_path = "data/simple_synthetic_data_gen.json"
    os.makedirs(os.path.dirname(json_data_file_path), exist_ok=True)

    try:
        # Create the OpenAI Client
        openai_client = OpenAI(api_key=openai_api_key)
    except Exception as e:
        print(f"Loading OpenAI model failed: {e}")
        sys.exit(1)

    prompt_suffixes = [
        "",
        "Focus on graphs that use varied node and edge colors, different edge styles, and edge weights.",
        "Focus on graphs that include subgraphs, same-rank node alignment, and orthogonal edges.",
        "Focus on graphs that use HTML-like labels (e.g., tables, formatted text) inside nodes.",
        "Focus on graphs that use a variety of node shapes (e.g., box, ellipse, diamond, record).",
    ]

    num_samples_suffix = 128
    batch_size = 32

    for i, suffix in enumerate(prompt_suffixes):
        for batch_idx in range(num_samples_suffix // batch_size):
            generated_dot_codes = generate_simple_graphviz_code(
                openai_client=openai_client,
                model_name="gpt-5-mini",
                prompt_suffix=suffix,
                temperature=1.0,
                batch_size=batch_size,
            )

            if generated_dot_codes:
                dot_codes.extend(generated_dot_codes)

                with open(json_data_file_path, "w") as json_file:
                    json.dump(dot_codes, json_file, indent=4)

                print(
                    f"Generated {len(generated_dot_codes)} during iteration {batch_idx + i * num_samples_suffix // batch_size} | suffix = {suffix} | batch = {batch_idx + 1}"
                )

    print(
        f"Generated {len(dot_codes)} final DOT samples saved to {json_data_file_path}"
    )
