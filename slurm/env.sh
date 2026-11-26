cd ~/adnn
export HF_HOME=~/adnn/models HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
PY=./venv/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
