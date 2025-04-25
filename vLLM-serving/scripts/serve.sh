#!/bin/bash

# Usage: ./serve.sh mistral | llama | deepseek

MODEL_NAME=$1

# API Parameters
HOST=localhost
PORT=8000

# Arguments
GPU_MEMORY_UTILIZATION=0.7
SWAP_SPACE=10 # in GiB -> CPU swap space size (GiB) per GPU.
CPU_OFFLOAD_GB=14 # in GiB -> CPU offload size (GiB) per GPU.
QUANTIZATION=fp8 # Possible choices: aqlm, awq, deepspeedfp, tpu_int8, fp8, ptpc_fp8, fbgemm_fp8, modelopt, nvfp4, marlin, gguf, gptq_marlin_24, gptq_marlin, awq_marlin, gptq, compressed-tensors, bitsandbytes, qqq, hqq, experts_int8, neuron_quant, ipex, quark, moe_wna16, torchao, None
MAX_MODEL_LEN=4K # in tokens -> Model context length
MAX_SEQ_LEN_TO_CAPTURE=4096 # in tokens -> Maximum sequence length covered by CUDA. If longer fall back to eager mode
TOKENIZER_MODE=auto

# Scheduler Config
MAX_NUM_BATCHED_TOKENS=4096 # in tokens -> Maximum number of tokens to be batched together (<= MAX_SEQ_LEN_TO_CAPTURE)
MAX_NUM_SEQS=1 # Maximum number of sequences to be processed in a single iteration.


case $MODEL_NAME in
  mistral)
    HF_MODEL="mistralai/Mistral-7B-Instruct-v0.3"
    TOKENIZER_MODE="mistral"
    ;;
  llama)
    HF_MODEL="meta-llama/Llama-2-7b-chat-hf"
    ;;
  deepseek)
    HF_MODEL="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    ;;
  gemma)
    HF_MODEL="google/gemma-3-1b-it"
    ;;
  phi)
    HF_MODEL="microsoft/Phi-3.5-mini-instruct"
    ;;
  *)
    echo "Usage: $0 mistral | llama | deepseek | gemma | phi"
    exit 1
    ;;
esac

echo "Launching model: $HF_MODEL (alias: $MODEL_NAME)"
echo "Memory: GPU ${GPU_MEMORY_UTILIZATION}, Seqs ${MAX_NUM_SEQS}, Length ${MAX_SEQ_LEN_TO_CAPTURE}"

python3 -m vllm.entrypoints.openai.api_server \
  --model $HF_MODEL \
  --served-model-name $MODEL_NAME \
  --host $HOST \
  --port $PORT \
  --gpu-memory-utilization $GPU_MEMORY_UTILIZATION \
  --swap-space $SWAP_SPACE \
  --cpu-offload-gb $CPU_OFFLOAD_GB \
  --quantization $QUANTIZATION \
  --max-model-len $MAX_MODEL_LEN \
  --max-seq-len-to-capture $MAX_SEQ_LEN_TO_CAPTURE \
  --tokenizer-mode $TOKENIZER_MODE \
  --max-num-batched-tokens $MAX_NUM_BATCHED_TOKENS \
  --max-num-seqs $MAX_NUM_SEQS \
  --disable-log-requests \
  --trust-remote-code \

if $DISABLE_LOGS; then
  CMD="$CMD --disable-log-requests"
fi
