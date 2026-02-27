# Sketch2Graphviz

![Sketch2Graphviz Icon](/client/public/assets/icon.svg)

<img src="resources/decision_tree_with_edit_demo.png" alt="Sketch2Graphviz Demo on a Decision Tree with an Edit" width="49%"/> <img src="resources/sketch_demo.png" alt="Sketch2Graphviz Demo on a Sketch of a Diagram" width="49%"/>

<img src="resources/system_architecture_demo.png" alt="Sketch2Graphviz Architecture Diagram Demo" width="49%"/> <img src="resources/flowchart_demo.png" alt="Sketch2Graphviz Flowchart Demo with an Edit" width="49%"/>

**Sketch2Graphviz** allows you to convert sketches or images of graphs and flowcharts to proper Graphviz code using a **LoRA fine-tuned Llama 3.2 11B Vision** and **Retrieval-Augmented Generation (RAG)** through a vector database built with PostgreSQL and PGVector, making a previously tedious and manual task fast, effortless, and automated.

The client-side web application uses React JS, Vite, and Tailwind CSS.
The server uses FastAPI and is deployed with Docker.

## Setup

The server uses Docker and can be deployed to services such as Runpod to be run with FastAPI and have GPU access. The public docker image can be accessed at **[rishabsa/sketch2graphviz:latest](https://hub.docker.com/r/rishabsa/sketch2graphviz)** on DockerHub. Docker containerizes the built model, FastAPI server, and PostgreSQL database. The project Docker image is used on [Runpod](https://www.runpod.io/) to host the FastAPI server with GPU access.

### Runpod Setup

Create a pod with an appropriate GPU (~12-16 GB VRAM for the 4-bit quantized model or ~24 GB VRAM for the 16-bit quantized model). For my purposes, I deployed the 4-bit quantized model on a RTX A5000 with 24 GB VRAM.

- Use the **rishabsa/sketch2graphviz:latest** docker image as the template on Runpod
- Set the container disk and volume disk storage both to ~40 GB
- Expose HTTP port `8000`
- Set an environment variable with the key `HF_TOKEN` and the value being a huggingface token with read permissions
  - Note: Make sure that you have been granted access to `meta-llama/Llama-3.2-11B-Vision-Instruct` on HuggingFace as it is a gated model
- Uncheck "Start Jupyter notebook"
- Deploy the pod on-demand

The FastAPI server will being to start, and in the logs, you should see the `meta-llama/Llama-3.2-11B-Vision-Instruct` model being downloaded and loaded in to VRAM with the fine-tuned LoRA adapters loaded from storage. By default, the base model is loaded with **4-bit quantization** and the LoRA adapters are loaded in **32-bits**. The Dockerfile automatically downloads and sets up PostgreSQL and PGVector, and also loads in the saved `sketch2graphvizdb.sql` file, which contains the vector embeddings and DOT code samples for RAG similarity search.

To test that everything works correctly, you can run the below command on your local machine from the root directory (change the runpod server to the one provided to you) to test out the FastAPI server and the Sketch2Graphviz model:

```bash
curl -X POST https://<RUNPOD_POD>-8000.proxy.runpod.net/graphviz_code_from_image -F "file=@model/testing_graphs/graph_6.png"
```

Run the React JS + Vite frotend client with `npm run dev` from the client directory. Make sure that you set the server URL to the server URL at which the Sketch2Graphviz FastAPI server and model are hosted in the `.env` file.

The Sketch2Graphviz web application allows you to sketch a graph or flowchart with multiple shapes and colors, or upload an image to then be converted to Graphviz code. The website also renders the generated Graphviz code, allowing you to make any necessary tweaks to the generated code.

## Implementation Details

### Generating Synthethic Data

Because there are no publicly available Graphviz datasets that provide DOT code of sufficient quality, I synthesized my own data by generating several Graphviz DOT code samples using the OpenAI API and the `gpt-5-mini` model. Specific types of graphs are generated with prompt suffixes that specify different types of nodes, edges, and attributes for all graphs generated in a batch.

### Model and Low-Rank Adaptation (LoRA) Fine-Tuning

The Sketch2Graphviz Vision-Language model (VLM) uses the 11 billion parameter `meta-llama/Llama-3.2-11B-Vision-Instruct` as a base model. The base model was fine-tuned with LoRA adapters on the linear layers in both the image encoder and text decoder. The model was loaded with **16-bit quantization** for LoRA fine-tuning and **4-bit quantization** for inferencing and deployment on the FastAPI server to reduce VRAM usage and deployment costs, with no noticeable impact on results.

Due to limited compute, the model was trained with a batch size of 1 with gradient accumulation. Backpropagation was performed once per batch to accumulate gradients, but the optimization step and updating parameters was only done after every 16 epochs, leading to an effective batch of size 16.

### Retrieval-Augmented Generation (RAG) and Vector Database

Retrieval-Augmented Generation (RAG) was used to improve the quality of DOT code generations produced. First, images in the training dataset are passed through the VLM with a constant prompt. The prompt and image data are passed through the VLM (image is passed through the image encoder) to get the hidden state output of the model. The hidden state is averaged over all token positions to get a single embedding vector representation of size `d_model` for the image. This is then L2 normalized to get a final image embedding vector.

PostgreSQL and PGVector were used to store Graphviz code and embedding pairs for retrieval at inference-time. The top-K most similar Graphviz codes are retrived by a similarity search by Euclidean L2 vector distance between the provided embedding at inference-time and those stored in the vector DB. The top-K most similar Graphviz codes corresponding to the most similar embeddings were then provided as context to the Sketch2Graphviz model for full generation as few-shot prompting by adding them to the prompt passed to the model.

Utilizing RAG through the vector similarity search significantly improved the quality of results through few-shot prompting, leading to noticable differences in the alignment of the target and generated Graphviz codes.

### Editing Generated Graphviz DOT Code

Sketch2Graphviz also allows users to continually improve generated Graphviz DOT code by requesting the model to make edits. Sketch2Graphviz can make rewrite and selective edits.

Rewrite edits prompt the language model with just the base Graphviz DOT code sample and a user edit request (no image) and ask it to apply the user's request to improve the code. The model regenerates all of the DOT code, applying the user's edit request. However, this does lead to some issues as the model can accidentally modify unrelated parts of the Graphviz code, hurting the response.

Making selective edits utilizes a different methodology in which the base Graphviz DOT code is first split into statements by certain indicators: `'{'`, `'}'`, `';'`. The statements are then indexed, and joined back together with the indexes serving as numbers. The numbered graphviz code and the user's edit request are sent to the language model with a JSON schema detailing the format to write actions.

```json
{
  "actions": [
    {"command": "add", "idx": <int>, "content": "<DOT statement>"},
    {"command": "edit", "idx": <int>, "content": "<DOT statement>"},
    {"command": "delete", "idx": <int>}
  ]
}
```

The language model outputs a set of JSON actions to follow to make the isolated/selective changes. The model is given the option to add, edit, or delete statements following the user's request.
Once the JSON is outputted, it is parsed and iterated through too validate and apply each of the selective actions to the base Graphviz DOT code.
