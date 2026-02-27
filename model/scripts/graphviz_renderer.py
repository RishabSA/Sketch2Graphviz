import os
from io import BytesIO
from graphviz import Source
from PIL import Image
import networkx as nx
import pydot


def render_graphviz_dot_code(
    dot_code: str,
    name: str,
    folder: str = "graphs",
    size: tuple[int, int] | None = (768, 768),
) -> str:
    os.makedirs(folder, exist_ok=True)
    src = Source(dot_code)

    # Render Graphviz graph from source DOT code
    output_path = src.render(
        filename=name,
        directory=folder,
        format="png",
        view=False,
        cleanup=True,
        quiet=True,
    )

    if size is not None:
        # Resize the rendered Graphviz image if provided with a size by pasting a resized image onto a blank canvas
        with Image.open(output_path) as image:
            width, height = image.size
            target_width, target_height = size

            # Calculate minimum resize ratio
            ratio_width = target_width / width
            ratio_height = target_height / height

            # Add a slight padding from the edge of image
            ratio = min(ratio_width, ratio_height) - 0.05

            new_width = int(width * ratio)
            new_height = int(height * ratio)

            # Resize the content
            resized_content = image.resize(
                (new_width, new_height), Image.Resampling.LANCZOS
            )

            # Paste onto a white canvas, keeping it centered
            background = Image.new(
                "RGB", (target_width, target_height), (255, 255, 255)
            )

            offset_x = (target_width - new_width) // 2
            offset_y = (target_height - new_height) // 2

            background.paste(resized_content, (offset_x, offset_y))

            background.save(output_path)

    return output_path


def render_graphviz_dot_code_pil(
    dot_code: str,
    size: tuple[int, int] | None = (768, 768),
) -> Image.Image:
    src = Source(dot_code)

    # Render to PNG bytes in memory
    png_bytes = src.pipe(format="png")
    image = Image.open(BytesIO(png_bytes)).convert("RGB")

    if size is None:
        return image

    width, height = image.size
    target_width, target_height = size

    # Calculate minimum resize ratio
    ratio_width = target_width / width
    ratio_height = target_height / height

    # Add a slight padding from the edge of image
    ratio = min(ratio_width, ratio_height) - 0.05

    new_width = int(width * ratio)
    new_height = int(height * ratio)

    # Resize the content
    resized_content = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    # Paste onto a white canvas, keeping it centered
    background = Image.new("RGB", (target_width, target_height), (255, 255, 255))

    offset_x = (target_width - new_width) // 2
    offset_y = (target_height - new_height) // 2

    background.paste(resized_content, (offset_x, offset_y))

    return background


def convert_graphviz_dot_to_networkx(dot_code: str) -> nx.Graph:
    pydot_graphs = pydot.graph_from_dot_data(dot_code)
    if not pydot_graphs:
        raise ValueError("Could not parse DOT code into a networkx graph.")

    networkx_graph = nx.nx_pydot.from_pydot(pydot_graphs[0])
    return networkx_graph


if __name__ == "__main__":
    dot_code = """
graph CulturalNetwork48 {
layout=neato;
node [shape=ellipse, style=filled, color=lightpink];
Artist [label="Artist"];
Gallery [label="Gallery"];
Museum [label="Museum"];
Curator [label="Curator"];
Collector [label="Collector"];
Critic [label="Critic"];
Auction [label="Auction_House", shape=folder];
Patron [label="Patron", shape=oval];
Publication [label="Publication", shape=note];
Exhibition [label="Exhibition", shape=box3d];
Artist -- Gallery;
Artist -- Museum;
Gallery -- Curator;
Museum -- Curator;
Curator -- Exhibition;
Exhibition -- Publication;
Collector -- Auction;
Auction -- Gallery;
Patron -- Collector;
Critic -- Publication;
Publication -- Critic;
Gallery -- Collector [style=dashed];
Artist -- Critic [style=dotted];
Museum -- Patron;
Exhibition -- Patron;
Collector -- Museum;
}
    """

    file_path = render_graphviz_dot_code(
        dot_code=dot_code,
        name="complex_graph_3",
        folder="testing_outputs",
        size=(768, 768),
    )
    print(f"Saved graph to: {file_path}")

    # networkx_graph = convert_graphviz_dot_to_networkx(dot_code=dot_code)
