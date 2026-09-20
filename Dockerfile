# The Tinfoil Container image. Private: only its SHA256 digest is published,
# in tinfoil-config.yml over in the public garlic-cvm repo. Every line here is
# part of the measurement a client verifies before sending plaintext.
FROM python:3.12-slim

# chunklaya's own source, pinned by commit. Bump this and rebuild to pick up
# harness changes; the digest in the public config changes with it.
ARG CHUNKLAYA_REF=5a1aa1a322d4c4f932fe0f52a9fb83f737b01a3e

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    USE_TF=0 \
    USE_TORCH=1 \
    TOKENIZERS_PARALLELISM=false \
    LAYA_MODEL_DIR=/opt/models/laya \
    PORT=8080

# curl is only here for the healthcheck in tinfoil-config.yml.
RUN apt-get update && apt-get install -y --no-install-recommends curl \
 && rm -rf /var/lib/apt/lists/*

# CPU-only torch. The default PyPI index serves the CUDA build: ~2.5 GB of
# libraries this container can never use, all of it measured into the image.
RUN pip install --no-cache-dir \
      --index-url https://download.pytorch.org/whl/cpu \
      torch

RUN pip install --no-cache-dir laya==0.3.3 "numpy>=2.0"

# Bake the weights in. The enclave has no route to the Hugging Face Hub, so
# laya's own snapshot_download fallback would fail at load time.
# `laya-multilingual` (mmBERT-base, 322M) is the checkpoint every number in
# the chunklaya README was measured on. One checkpoint only: all three would
# add ~4.6 GB. Kept ahead of the source install so a chunklaya bump does not
# invalidate this 647 MB layer.
RUN python -c "\
import os, shutil; \
from huggingface_hub import snapshot_download; \
src = snapshot_download('convaiinnovations/laya', allow_patterns=['multilingual/*']); \
shutil.copytree(os.path.join(src, 'multilingual'), '/opt/models/laya'); \
print('weights:', sorted(os.listdir('/opt/models/laya')))" \
 && rm -rf /root/.cache/huggingface

# --no-deps: torch, laya and numpy are installed above. Letting pip resolve
# them again pulls torch from PyPI, i.e. the CUDA build.
RUN pip install --no-cache-dir --no-deps \
      "https://github.com/myxamediyar/chunklaya/archive/${CHUNKLAYA_REF}.tar.gz"

COPY server.py /opt/server.py

RUN find /usr/local/lib/python3.12 -name '__pycache__' -type d -prune -exec rm -rf {} + 2>/dev/null; \
    find /usr/local/lib/python3.12 -name 'tests' -type d -prune -exec rm -rf {} + 2>/dev/null; \
    rm -rf /root/.cache /tmp/* ; true

EXPOSE 8080

# Absolute path, matching what the Nitro Enclave required: init execs this
# directly rather than resolving it against PATH.
CMD ["/usr/local/bin/python", "/opt/server.py"]
