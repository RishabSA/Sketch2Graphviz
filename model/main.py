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
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException, UploadFile, File, Query
from fastapi.responses import PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from scripts.model import load_sketch2graphviz_vlm, print_num_params
from scripts.inference import predict_graphviz_dot_from_image, edit_graphviz_dot
from scripts.selective_edit import get_numbered_graphviz_code, apply_llm_edit_plan
from scripts.prompts import (
    graphviz_code_from_image_instruction,
    graphviz_code_edit_instruction,
    graphviz_selective_code_edit_instruction,
)

load_dotenv()

logger = logging.getLogger("uvicorn.error")

SEED = 42
batch_size = 1

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

        model = load_sketch2graphviz_vlm(
            model_load_dir="checkpoints",
            epoch_load=1,
            quantization=quantization,
            is_training=False,
            device=device,
        )

        # model = load_sketch2graphviz_vlm(
        #     model_load_dir=None,
        #     epoch_load=None,
        #     quantization=quantization,
        #     is_training=False,
        #     device=device,
        # )

        app.state.model = model

        print_num_params(app.state.model)

        yield
    except Exception as e:
        logger.error(f"An error occurred while starting the API: {e}")
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


@app.post("/graphviz_code_from_image", response_class=PlainTextResponse)
async def get_graphviz_code_from_image(
    file: UploadFile = File(...),
    use_rag: bool = Query(True),
    top_K_rag: int = Query(5, ge=0),
) -> str:
    try:
        if not hasattr(app.state, "model"):
            logger.error("The Sketch2Graphviz VLM model is not loaded")
            raise HTTPException(503, "The Sketch2Graphviz VLM model is not loaded")

        content = await file.read()

        # Open as PIL image
        image = Image.open(io.BytesIO(content)).convert("RGB")

        if image.size != (768, 768):
            image = image.resize((768, 768), resample=Image.LANCZOS)

        predicted_graphviz_output = predict_graphviz_dot_from_image(
            model=app.state.model,
            image=image,
            instruction=graphviz_code_from_image_instruction,
            should_print_instruction=False,
            use_rag=use_rag,
            top_K_rag=top_K_rag,
            max_new_tokens=2048,
            do_sample=False,
            temperature=0.3,
            skip_special_tokens=False,
            device=app.state.device,
        )

        return predicted_graphviz_output
    except Exception as e:
        logger.error(
            f"An error occurred while attempting to convert the provided image to Graphviz code: {e}"
        )

        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while attempting to convert the provided image to Graphviz code: {e}",
        )


class GraphvizEditRequest(BaseModel):
    edit_text: str
    graphviz_code: str
    use_selective_changes: bool


@app.post("/graphviz_code_edit", response_class=PlainTextResponse)
async def get_graphviz_code_edit(request: GraphvizEditRequest) -> str:
    try:
        if not hasattr(app.state, "model"):
            logger.error("The Sketch2Graphviz VLM model is not loaded")
            raise HTTPException(503, "The Sketch2Graphviz VLM model is not loaded")

        if request.use_selective_changes:
            numbered_graphviz_code, graphviz_parts = get_numbered_graphviz_code(
                request.graphviz_code
            )

            predicted_graphviz_output = edit_graphviz_dot(
                model=app.state.model,
                instruction=graphviz_selective_code_edit_instruction,
                edit_text=request.edit_text,
                graphviz_code=numbered_graphviz_code,
                should_print_instruction=False,
                max_new_tokens=2048,
                do_sample=False,
                temperature=0.3,
                skip_special_tokens=False,
                device=app.state.device,
            )

            updated_graphviz_code = apply_llm_edit_plan(
                graphviz_parts=graphviz_parts,
                llm_response_text=predicted_graphviz_output,
            )

            return updated_graphviz_code
        else:
            predicted_graphviz_output = edit_graphviz_dot(
                model=app.state.model,
                instruction=graphviz_code_edit_instruction,
                edit_text=request.edit_text,
                graphviz_code=request.graphviz_code,
                should_print_instruction=False,
                max_new_tokens=2048,
                do_sample=False,
                temperature=0.3,
                skip_special_tokens=False,
                device=app.state.device,
            )

            return predicted_graphviz_output
    except Exception as e:
        logger.error(
            f"An error occurred while attempting to make an edit to the Graphviz DOT code: {e}"
        )

        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while attempting to make an edit to the Graphviz DOT code: {e}",
        )
