import os
from graphviz import Source
from PIL import Image


def render_graphviz_dot_code(
    dot_code: str,
    name: str,
    folder: str = "graphs",
    size: tuple[int, int] | None = (336, 336),
) -> str:
    os.makedirs(folder, exist_ok=True)
    src = Source(dot_code)

    output_path = src.render(
        filename=name,
        directory=folder,
        format="png",
        view=False,
        cleanup=True,
    )

    if size is not None:
        # Resize with the same aspect ratio and pad with whitespace
        with Image.open(output_path) as image:
            target_w, target_h = size

            # Scale to fit inside target size while preserving aspect ratio
            image.thumbnail((target_w, target_h), Image.Resampling.LANCZOS)

            # Create a white background canvas
            background = Image.new("RGB", size, (255, 255, 255))

            # Center the graph on the canvas
            offset_x = (target_w - image.width) // 2
            offset_y = (target_h - image.height) // 2
            background.paste(image, (offset_x, offset_y))

            background.save(output_path)

    return output_path


if __name__ == "__main__":
    dot_code = """
    digraph G {
        rankdir=LR;
        A -> B;
        B -> C;
        A -> C [label="shortcut"];
    }
    """

    file_path = render_graphviz_dot_code(
        dot_code=dot_code, name="graph_1", folder="graphs", size=(336, 336)
    )
    print(f"Saved graph to: {file_path}")
