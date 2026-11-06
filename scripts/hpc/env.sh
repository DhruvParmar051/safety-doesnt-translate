# Environment for the DAU HPC cluster, sourced by scripts/hpc/setup_login.sh and every job.
source ~/envs/sdt/bin/activate
# Your Slurm account, one line: export SDT_ACCOUNT=<account>. Not committed.
[ -f scripts/hpc/local.env ] && source scripts/hpc/local.env
export SDT_PROFILE=hpc
export HF_HOME=/tmp/$USER/hf_cache TMPDIR=/tmp/$USER/tmp PYTHONUNBUFFERED=1
export HF_TOKEN_PATH=$HOME/.cache/huggingface/token   # the token lives in home, not in HF_HOME
# FlashInfer's sampler compiles a kernel at start-up and GPU nodes have no CUDA compiler; decoding
# is greedy, so vLLM's PyTorch sampler gives the same tokens.
export VLLM_USE_FLASHINFER_SAMPLER=0
# Some installed library starts CUDA at import, and a forked engine process cannot reuse it.
export VLLM_WORKER_MULTIPROC_METHOD=spawn
mkdir -p "$HF_HOME/hub" "$TMPDIR" && chmod 700 "/tmp/$USER"

# Home has a 25 GB quota, so model outputs live in /tmp; runs/models points there on every node.
MODELS_DIR=/tmp/$USER/sdt-models
mkdir -p "$MODELS_DIR" runs logs
[ -L runs/models ] || ln -s "$MODELS_DIR" runs/models
