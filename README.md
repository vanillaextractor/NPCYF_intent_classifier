# NPCYF Intent Classifier

A robust multi-class intent classifier designed to route user queries related to crop yield forecasting and climatic data. The system distinguishes between **SQL/Database** queries, **General** conversational queries, and **Platform**-related queries.

## 🚀 Overview

The project uses a hybrid approach:

- **Intent Classifier**: A Scikit-learn based multiclass model (`MulticlassIntentClassifier`) that categorizes input.
- **Text-to-SQL**: A dedicated LLM (Qwen2.5-Coder) optimized for generating PostgreSQL queries based on a specific agricultural schema.
- **General LLM**: A general-purpose LLM (Llama 3) for handling non-technical conversational queries.

## 🛠️ Prerequisites

- **Python**: 3.8+ (Recommended 3.14 as per venv)
- **PostgreSQL**: A running instance with the NPCYF schema.
- **Required Libraries**:
  ```bash
  pip install llama-cpp-python psycopg2-binary scikit-learn python-dotenv
  ```

## ⚙️ Setup & Installation

1. **Virtual Environment**:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On macOS/Linux
   ```

2. **Environment Variables**:
   Create a `.env` file in the root directory (refer to `.env.example`):

   ```env
   DB_HOST=localhost
   DB_NAME=postgres
   DB_USER=your_user
   DB_PASSWORD=your_password
   INTENT_THRESHOLD=0.5
   ```

3. **Models**:
   Download the necessary GGUF and Pickle models:
   ```bash
   python scripts/download_models.py
   ```
   Ensure models are placed in the `models/` directory as specified in `config.json`.

## 📖 Usage

### Interactive Inference

The primary entry point for using the system in a chat-like interface:

```bash
python scripts/interactive_inference.py
```

This script handles:

- Intent classification.
- Dynamic prompt generation for SQL based on detected locations (States/Districts).
- SQL validation and automatic retry logic.
- Database execution and result display.

### Training the Classifier

To retrain the intent classifier with new data:

```bash
python scripts/train_multiclass_model.py
```

### Benchmarking

Run the SQL generation benchmark to evaluate model performance:

```bash
python scripts/benchmark_sql_modular.py
```

## 📂 Project Structure

- `src/`: Core logic and classifier classes.
- `scripts/`: Utility scripts for inference, training, and benchmarking.
- `models/`: Storage for LLM GGUF files and classifier pickle files.
- `data/`: Training and testing datasets.
- `config.json`: Configuration for model paths and LLM parameters.


