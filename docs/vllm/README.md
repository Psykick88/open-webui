# Using Open WebUI with vLLM and TensorRT-LLM

This guide shows how to run **MythoMax L2 13B** using the [vLLM](https://github.com/vllm-project/vllm) server with TensorRT acceleration and connect it to Open WebUI.

## Prerequisites

- Docker with NVIDIA container runtime enabled
- CUDA 12.9 drivers on the host
- Two NVIDIA RTX 3090 GPUs connected via NVLink
- Approximately 24GB of GPU memory per GPU (required for the 13B model)

## 1. Start the vLLM server

Use the `vllm` Docker image together with the `nvcr.io/nvidia/tensorrtllm` runtime to enable TensorRT. The container below exposes an OpenAI-compatible HTTP API on port `8000`.

```bash
# Pull the container images (one-time)
docker pull nvcr.io/nvidia/tensorrtllm:v23.10-py3

# Start vLLM with TensorRT backend
# Adjust MODEL_PATH to the location of the HF formatted MythoMax L2 13B weights.
docker run -d --gpus all \
  -p 8000:8000 \
  -v /path/to/models:/models \
  --name vllm-server \
  nvcr.io/nvidia/tensorrtllm:v23.10-py3 \
  python3 -m vllm.entrypoints.openai.api_server \
    --model /models/MythoMax-L2-13B \
    --tensor-parallel-size 2 \
    --download-dir /models/cache
```

This command launches vLLM using both GPUs with TensorRT optimizations. Modify the `MODEL_PATH` if your model files are stored elsewhere.

## 2. Configure Open WebUI

Create the following `docker-compose.vllm.yaml` next to the existing compose files:

```yaml
services:
  open-webui:
    image: ghcr.io/open-webui/open-webui:cuda
    container_name: open-webui
    ports:
      - "3000:8080"
    environment:
      - ENABLE_OPENAI_API=true
      - OPENAI_API_BASE_URLS=http://vllm-server:8000/v1
    volumes:
      - open-webui:/app/backend/data
    depends_on:
      - vllm-server
    extra_hosts:
      - host.docker.internal:host-gateway

  vllm-server:
    image: nvcr.io/nvidia/tensorrtllm:v23.10-py3
    command: >
      python3 -m vllm.entrypoints.openai.api_server
      --model /models/MythoMax-L2-13B
      --tensor-parallel-size 2
      --download-dir /models/cache
    volumes:
      - /path/to/models:/models
    deploy:
      resources:
        reservations:
          devices:
            - capabilities: [gpu]
```

Bring up the stack with:

```bash
docker compose -f docker-compose.vllm.yaml up -d
```

After the containers start, navigate to [http://localhost:3000](http://localhost:3000) to access the WebUI. Open WebUI will route requests to the vLLM server using the OpenAI API-compatible endpoint.

## Tips

- Ensure that CUDA 12.9 drivers are installed on the host for maximum performance.
- Adjust `--tensor-parallel-size` according to the number of GPUs available.
- You can interact with the model via the WebUI, the built-in API, or any OpenAI compatible client such as LM Studio.

