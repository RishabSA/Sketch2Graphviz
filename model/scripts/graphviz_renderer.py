import os
from graphviz import Source
from PIL import Image, ImageChops


def render_graphviz_dot_code(
    dot_code: str,
    name: str,
    folder: str = "graphs",
    size: tuple[int, int] | None = (1024, 1024),
) -> str:
    os.makedirs(folder, exist_ok=True)
    src = Source(dot_code)

    output_path = src.render(
        filename=name,
        directory=folder,
        format="png",
        view=False,
        cleanup=True,
        quiet=True,
    )

    if size is not None:
        with Image.open(output_path) as image:
            width, height = image.size
            target_width, target_height = size

            ratio_w = target_width / width
            ratio_h = target_height / height
            ratio = min(ratio_w, ratio_h) - 0.05

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


if __name__ == "__main__":
    dot_code = """
    digraph G26 { node [shape=hex, style=filled, fillcolor=white]; hA [fillcolor=gold]; hB [fillcolor=lightsteelblue]; hC [fillcolor=lightgreen]; hD [fillcolor=lightpink]; hA -> hB [color=black, style=solid, weight=4, penwidth=2]; hB -> hC [color=blue, style=dotted, weight=1]; hC -> hD [color=gray, style=dashed, weight=2]; }
    """

    file_path = render_graphviz_dot_code(
        dot_code=dot_code, name="graph_4", folder="testing_outputs", size=(768, 768)
    )
    print(f"Saved graph to: {file_path}")
