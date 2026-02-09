import os
from huggingface_hub import hf_hub_download, list_repo_files

MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

def download_model(repo_id, filename, local_filename):
    print(f"Downloading {filename} from {repo_id}...")
    try:
        path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=MODELS_DIR,
            local_dir_use_symlinks=False
        )
        # Rename to a standard name for easier usage
        standard_path = os.path.join(MODELS_DIR, local_filename)
        if path != standard_path:
            os.rename(path, standard_path)
        print(f"Successfully downloaded to {standard_path}")
        return standard_path
    except Exception as e:
        print(f"Error downloading {filename}: {e}")
        return None

def main():
    # General Model: Llama-3.2-1B-Instruct
    # Using bartowski's quant
    general_repo = "bartowski/Llama-3.2-1B-Instruct-GGUF"
    general_file = "Llama-3.2-1B-Instruct-Q4_K_M.gguf"
    
    # SQL Model: Snowflake Arctic Text2SQL (Quantized)
    # Found: mradermacher/Arctic-Text2SQL-R1-7B-GGUF
    sql_repo = "mradermacher/Arctic-Text2SQL-R1-7B-GGUF"
    sql_file = "Arctic-Text2SQL-R1-7B.Q4_K_M.gguf"
    
    print("Starting model downloads...")
    
    # Download General
    if os.path.exists(os.path.join(MODELS_DIR, "general_model.gguf")):
        print("General model already exists. Skipping.")
        general_path = os.path.join(MODELS_DIR, "general_model.gguf")
    else:
        general_path = download_model(general_repo, general_file, "general_model.gguf")
    
    # Download SQL
    if os.path.exists(os.path.join(MODELS_DIR, "sql_model.gguf")):
         print("SQL model already exists. Skipping.")
         sql_path = os.path.join(MODELS_DIR, "sql_model.gguf")
    else:
        sql_path = download_model(sql_repo, sql_file, "sql_model.gguf")
    
    if general_path and sql_path:
        print("\nAll models downloaded successfully!")
        print(f"General Model: {general_path}")
        print(f"SQL Model: {sql_path}")
    else:
        print("\nSome downloads failed. Check errors above.")

if __name__ == "__main__":
    main()
