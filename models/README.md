## Local models (not tracked)

This directory is for **local HuggingFace model artifacts** (weights/tokenizers/configs).

- **Not tracked by git**: large files like `*.safetensors` should never be committed.
- **Recommended**: put your downloaded model here and point scripts to it via
  `NANOVLLM_MODEL_PATH`.

Example:

```bash
export NANOVLLM_MODEL_PATH="$PWD/models/qwen/Qwen-3-0.6B"
python examples/jax_example.py
```

