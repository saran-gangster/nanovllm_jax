#!/bin/bash
# RunPod Setup Script for nanovllm_jax
# Optimized for: runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404
#
# Usage: ssh root@<IP> -p <PORT> -i ~/.ssh/id_ed25519 -o StrictHostKeyChecking=no "bash -s" < nanovllm_jax/setup_runpod.sh
#
# Or run remotely:
# ssh root@<IP> -p <PORT> -i ~/.ssh/id_ed25519 -o StrictHostKeyChecking=no "curl -s https://raw.githubusercontent.com/saran-gangster/nanovllm_jax/main/setup_runpod.sh | bash"

set -e  # Exit on error

echo "============================================"
echo "  RunPod Setup for nanovllm_jax"
echo "  (PyTorch 2.8.0 Template)"
echo "============================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

# Step 1: Check GPU
echo ""
echo "Step 1: Checking GPU..."
if command -v nvidia-smi &> /dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -1)
    print_status "GPU detected: $GPU_NAME ($GPU_MEM)"
else
    print_error "No GPU detected! nvidia-smi not found."
    exit 1
fi

# Step 2: Install JAX with CUDA support and dependencies
echo ""
echo "Step 2: Installing JAX with CUDA 12 support and dependencies..."
pip install --break-system-packages --quiet 'jax[cuda12]' flax transformers safetensors xxhash

# Verify JAX installation
JAX_VERSION=$(python3 -c "import jax; print(jax.__version__)")
DEVICES=$(python3 -c "import jax; print(len(jax.devices()))" 2>/dev/null)
print_status "JAX $JAX_VERSION installed with $DEVICES device(s)"

# Step 3: Clone or update repository
echo ""
echo "Step 3: Setting up nanovllm_jax repository..."
cd /workspace

if [ -d "nanovllm_jax" ]; then
    print_warning "Repository exists, pulling latest changes..."
    cd nanovllm_jax
    git pull
else
    git clone https://github.com/saran-gangster/nanovllm_jax.git
    cd nanovllm_jax
    print_status "Repository cloned"
fi
print_status "Repository ready at /workspace/nanovllm_jax"

# Step 4: Download model (optional - Qwen3-0.6B)
echo ""
echo "Step 4: Downloading Qwen3-0.6B model..."
MODEL_PATH="/workspace/models/Qwen3-0.6B"

if [ -d "$MODEL_PATH" ] && [ -f "$MODEL_PATH/model.safetensors" ]; then
    print_warning "Model already exists at $MODEL_PATH"
else
    mkdir -p /workspace/models
    huggingface-cli download Qwen/Qwen3-0.6B --local-dir "$MODEL_PATH" 2>/dev/null || {
        print_warning "huggingface-cli download failed, trying alternative method..."
        python3 -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-0.6B', local_dir='$MODEL_PATH')"
    }
    print_status "Model downloaded to $MODEL_PATH"
fi

# Step 5: Quick test
echo ""
echo "Step 5: Running quick test..."
cd /workspace
export PYTHONPATH=/workspace/nanovllm_jax:$PYTHONPATH

python3 << 'EOF'
import sys
sys.path.insert(0, "/workspace/nanovllm_jax")

# Suppress JAX warnings for cleaner output
import os
os.environ['JAX_PLATFORMS'] = 'cuda'

import jax
print(f"JAX devices: {jax.devices()}")

# Quick import test
from nanovllm_jax import LLM, SamplingParams
print("✓ nanovllm_jax imports successful")
EOF

if [ $? -eq 0 ]; then
    print_status "Quick test passed!"
else
    print_error "Quick test failed!"
    exit 1
fi

# Done!
echo ""
echo "============================================"
echo -e "${GREEN}  Setup Complete!${NC}"
echo "============================================"
echo ""
echo "To use nanovllm_jax:"
echo ""
echo "  cd /workspace"
echo "  export PYTHONPATH=/workspace/nanovllm_jax:\$PYTHONPATH"
echo ""
echo "  python3 -c '"
echo "  from nanovllm_jax import LLM, SamplingParams"
echo "  llm = LLM(model=\"/workspace/models/Qwen3-0.6B\")"
echo "  outputs = llm.generate([\"Hello, my name is\"], SamplingParams(temperature=0.7, max_tokens=50))"
echo "  print(outputs)"
echo "  '"
echo ""
