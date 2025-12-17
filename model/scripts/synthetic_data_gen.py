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
                        f"- Return exactly {batch_size} samples.\n"
                        "- Represent them as a JSON object with a single field `dot_codes`, "
                        "  whose value is a list of strings.\n"
                        "- Each list element must be one complete, standalone, valid and syntactically correct Graphviz DOT program.\n"
                        "- Keep each graph relatively simple, with each DOT program having between 2 and 8 nodes and between 1 and 16 edges, but with some variety in structure.\n"
                        "(chains, stars, small DAGs, small undirected graphs, etc.).\n"
                        "- Use either 'graph' (with --) or 'digraph' (with ->) consistently per sample.\n"
                        "- Do NOT reuse the same node/edge names and structure across all samples.\n"
                        "- Do NOT include comments or blank lines at the top or bottom.\n"
                        "- Do NOT include any explanations, markdown, prose, or backticks, only raw DOT code.\n"
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
        # Styling / colors / weights
        " Focus on graphs that use varied node and edge colors, different edge styles, and edge weights while still keeping graphs small (1–5 nodes).",
        " Focus on graphs where some edges are bold or thicker (using penwidth) and others are thin, combined with different arrowhead styles.",
        " Focus on graphs that use filled nodes with different fillcolors and fontcolors to create strong visual contrast.",
        " Focus on graphs that mix solid, dashed, and dotted edges, sometimes combining these with color and weight attributes.",
        # Layout / rank / direction
        " Focus on graphs that explicitly set rankdir (LR, TB, BT, RL) and show different linear and layered layouts with small node sets.",
        " Focus on graphs that use same-rank constraints and rank attributes to align nodes horizontally or vertically.",
        " Focus on graphs that contain a simple chain or path and one or two extra edges that cross ranks in interesting ways.",
        " Focus on graphs that represent small trees or DAGs with a clear top-to-bottom or left-to-right flow.",
        # Subgraphs and clusters
        " Focus on graphs that include subgraphs and clusters (cluster_0, cluster_1) with different colors and labels.",
        " Focus on graphs that have one main cluster and a few nodes outside of the cluster connected by edges.",
        " Focus on graphs where subgraphs are used only for same-rank grouping instead of visual clustering.",
        " Focus on graphs that combine clusters with different rankdirs inside the same diagram.",
        # HTML-like labels
        " Focus on graphs that use HTML-like labels with simple tables (one or two rows) inside nodes.",
        " Focus on graphs that use HTML-like labels to show multi-line text or simple formatting such as bold or line breaks.",
        " Focus on graphs that mix plain text labels and HTML-like labels within the same graph.",
        # Node shapes / records / ports
        " Focus on graphs that use a variety of node shapes (box, ellipse, diamond, circle, record) in a single small graph.",
        " Focus on graphs that use record-shaped nodes with multiple fields and edges connecting to specific fields using ports.",
        " Focus on graphs that mix normal nodes and record-shaped nodes in one diagram.",
        " Focus on graphs that use doublecircle or Msquare / Mdiamond shapes for special nodes (e.g., start/end).",
        # Edge labels and semantics
        " Focus on graphs where most edges have labels, including numeric labels, short words, and simple phrases.",
        " Focus on graphs that contain bidirectional relationships (A -> B and B -> A) with different edge labels.",
        " Focus on graphs that include self-loops (a node with an edge to itself) along with normal edges.",
        " Focus on graphs that model simple state machines with states as nodes and labeled transitions as directed edges.",
        " Focus on graphs that look like tiny flowcharts with start, decision, and end nodes and labeled edges.",
        # Undirected / mixed structures
        " Focus on undirected graphs (using 'graph' and '--') that include simple paths, cycles, and small complete graphs.",
        " Focus on graphs that have more than one connected component, possibly mixing a path in one component and a star in another.",
        " Focus on graphs that combine both directed and undirected structures across different samples (but each individual sample should be consistent).",
        # Global attributes / graph-level settings
        " Focus on graphs that set global node, edge, and graph attributes at the top (e.g., node [shape=box]; edge [color=gray];).",
        " Focus on graphs that vary fonts, font sizes, and font colors for different nodes and edge labels.",
        " Focus on graphs that use background-like effects via style=filled and different fillcolors on clusters or special nodes.",
        # Constraints / advanced edge usage (still simple sized graphs)
        " Focus on graphs that use edge attributes like minlen and weight to hint at layout distances, while keeping graphs small (1–5 nodes).",
        " Focus on graphs that include parallel edges between the same pair of nodes, with different colors or labels.",
        " Focus on graphs that use constraint=true/false on certain edges to influence layout.",
        " Focus on graphs that use invisible edges (style=invis) to control alignment or spacing between nodes.",
        # Compositional / mixed-feature prompts
        " Focus on graphs that combine clusters, different node shapes, and labeled edges all in the same small diagram.",
        " Focus on graphs that mix HTML-like labels, colored edges, and rankdir changes to create visually complex but small graphs.",
        " Focus on graphs that resemble small network diagrams, with different node roles indicated by shape, color, and edge style.",
        " Focus on graphs that resemble simple dependency graphs or call graphs, using directed edges and a few different node styles.",
    ]

    batch_size = 32

    for i, suffix in enumerate(prompt_suffixes):
        for temperature in [1.0]:  # [0.25, 0.5, 1.0, 1.25]
            generated_dot_codes = generate_simple_graphviz_code(
                openai_client=openai_client,
                model_name="gpt-5-mini",
                prompt_suffix=suffix,
                temperature=temperature,
                batch_size=batch_size,
            )

            if generated_dot_codes:
                dot_codes.extend(generated_dot_codes)

                with open(json_data_file_path, "w") as json_file:
                    json.dump(dot_codes, json_file, indent=4)

                print(
                    f"Generated {len(generated_dot_codes)} with temperature = {temperature} | suffix = {suffix}"
                )

    print(
        f"Generated {len(dot_codes)} final DOT samples saved to {json_data_file_path}"
    )
