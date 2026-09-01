FROM python:3.13-slim

# Model weights are baked in at build time so a container start does not depend
# on the Hugging Face CDN, and so every replica serves byte-identical weights.
ARG MODEL_NAME=gpt2
ENV MODEL_NAME=${MODEL_NAME} \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/opt/hf \
    HF_HUB_DISABLE_PROGRESS_BARS=1 \
    OMP_NUM_THREADS=1 \
    TORCH_THREADS=1

WORKDIR /app

COPY worker/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY worker/server ./server
COPY proto /proto

RUN python -c "\
from transformers import AutoModelForCausalLM, AutoTokenizer; \
import os; m = os.environ['MODEL_NAME']; \
AutoTokenizer.from_pretrained(m); AutoModelForCausalLM.from_pretrained(m)" \
 && chmod -R a+rX /opt/hf

RUN useradd --create-home --uid 10001 worker && chown -R worker /app
USER worker

EXPOSE 50051 9100
CMD ["python", "-m", "server.worker"]
