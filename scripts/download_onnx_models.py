
import os
from huggingface_hub import snapshot_download

MODELS_DIR = "models/onnx"
os.makedirs(MODELS_DIR, exist_ok=True)

def download_repo(repo_id, local_dir):
    print(f"Downloading {repo_id} to {local_dir}...")
    try:
        # Only download config files and the 4-bit quantized ONNX models to save space
        snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            resume_download=True,
            allow_patterns=["*.json", "*.txt", "*.jinja", "onnx/model_q4.onnx", "onnx/model_q4.onnx_data*"]
        )
        print(f"✅ Successfully downloaded {repo_id}")
        return True
    except Exception as e:
        print(f"❌ Error downloading {repo_id}: {e}")
        return False

# 1. Llama-3.2-1B-Instruct ONNX (Community Export)
download_repo("onnx-community/Llama-3.2-1B-Instruct-ONNX", f"{MODELS_DIR}/llama1b")

# 2. Qwen2.5-Coder-1.5B-Instruct ONNX (Community Export)
download_repo("onnx-community/Qwen2.5-Coder-1.5B-Instruct", f"{MODELS_DIR}/qwen15b_coder")

print("\n✅ All downloads complete!")
