# uvicorn main:app --reload
# /docs for API documentation

import os
import io
import random
import numpy as np
import torch
from dotenv import load_dotenv
from huggingface_hub import login
from PIL import Image
import logging
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from scripts.model import load_sketch2graphviz_vlm_local, print_num_params
from scripts.inference import predict_graphviz_dot

load_dotenv()

logger = logging.getLogger("uvicorn.error")

SEED = 42
batch_size = 1

instruction = """
You are an expert compiler that converts images of Graphviz diagrams into their exact Graphviz DOT code.
Given an image of a graph, using only the image, output only the DOT code, starting with either 'digraph' or 'graph', with no explanations, no markdown, and no extra text.
Graphviz DOT is a plain-text language for describing graphs as nodes and edges with optional attributes such as labels, shapes, colors, and styles, for both directed ('digraph') and undirected ('graph') diagrams.

## Core Syntax Rules

1.  **Graph Type:**
    * Use `digraph` (Directed Graph) for hierarchies, flows, or dependencies. Use `->` for edges.
    * Use `graph` (Undirected Graph) for physical networks or mutual connections. Use `--` for edges.
2.  **Identifiers:**
    * Alphanumeric strings (e.g., `A`, `node1`) do not need quotes.
    * Strings with spaces, special characters, or reserved keywords MUST be enclosed in double quotes (e.g., `"User Login"`, `"Data-Base"`).
3.  **Statement Termination:** End all node, edge, and attribute statements with a semicolon `;`.
4.  **Scope:** All code must be enclosed within braces `{ ... }`.

## Attribute Dictionary

Apply attributes using brackets `[key=value]`. If multiple attributes are needed, comma-separate them or use spaces: `[shape=box, color=red]`.

### Node Attributes

  * **`shape`**:
    * Process/Step: `box`
    * Start/End: `ellipse` or `oval`
    * Decision: `diamond`
    * Database: `cylinder`
    * Code/Structure: `record` (use `|` to separate fields in label)
  * **`style`**: `filled`, `rounded`, `dotted`, `invis`
  * **`fillcolor`**: Hex codes (`#FF0000`) or common names (`lightblue`). Only visible if `style=filled`.
  * **`label`**: The visible text. If omitted, the identifier is used.

### Edge Attributes

  * **`label`**: Text displayed along the line.
  * **`style`**: `solid` (default), `dashed` (future/theoretical), `dotted`.
  * **`dir`**: `forward` (default), `back`, `both`, `none`.
  * **`color`**: Edge color.

  * Output **only** the code block.
  * Do not include any explanations.
  * Ensure all braces `{}` are balanced.

"""

quantization = "4-bit"


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        app.state.device = device

        print(f"Currently on device: {app.state.device}")

        hf_token = os.getenv("HF_TOKEN")
        login(token=hf_token)

        model = load_sketch2graphviz_vlm_local(
            model_load_dir="vlm_model",
            epoch_load=2,
            quantization=quantization,
            device=device,
        )
        app.state.model = model

        print_num_params(app.state.model)

        yield
    except Exception as e:
        logger.error(f"An error occured while starting the API: {e}")
        raise


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {"Name": "Sketch2Graphviz FastAPI"}


@app.post("/graphviz_code", response_class=PlainTextResponse)
async def get_graphviz_code(file: UploadFile = File(...)) -> str:
    try:
        if not hasattr(app.state, "model"):
            logger.error("The Sketch2Graphviz VLM model is not loaded")
            raise HTTPException(503, "The Sketch2Graphviz VLM model is not loaded")

        content = await file.read()

        # Open as PIL image
        image = Image.open(io.BytesIO(content)).convert("RGB")

        if image.size != (768, 768):
            image = image.resize((768, 768), resample=Image.NEAREST)

        predicted_graphviz_output = predict_graphviz_dot(
            model=app.state.model,
            image=image,
            instruction=instruction,
            should_print_instruction=False,
            use_rag=True,
            top_K_rag=5,
            max_new_tokens=2048,
            do_sample=False,
            temperature=0.3,
            skip_special_tokens=False,
            device=app.state.device,
        )

        return predicted_graphviz_output
    except Exception as e:
        logger.error(
            f"An error occured while attempting to convert the provided image to Graphviz code: {e}"
        )

        raise HTTPException(
            status_code=500,
            detail=f"An error occured while attempting to convert the provided image to Graphviz code: {e}",
        )
