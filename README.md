# RAG Evaluation

A baseline pipeline for building a Retrieval-Augmented Generation (RAG) system over a custom dataset (e.g. SQuAD v2) and evaluating it with [DeepEval](https://github.com/confident-ai/deepeval) using Amazon Bedrock models.

The pipeline:
1. Ingests a dataset, embeds its passages, and stores them in a FAISS vector store (optionally also building a BM25 index for hybrid retrieval).
2. Retrieves context for each evaluation question (vector and/or BM25 search).
3. Generates an answer for each question with a Bedrock LLM.
4. Scores the generated answers against the ground truth with DeepEval metrics (Answer Relevancy, Contextual Precision, Contextual Recall, Contextual Relevancy, Faithfulness).
5. Exports the results to an Excel spreadsheet.

## Requirements

- Python 3.10+
- An AWS account with access to Amazon Bedrock (for embeddings, generation, and evaluation models)

## Setup

1. Clone the repository and install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Copy the example environment file and fill in your values:

   ```bash
   cp example.env .env
   ```

   The `.env` file configures:
   - AWS credentials and region
   - Bedrock model IDs for embeddings, generation, and evaluation
   - Storage paths for the FAISS index and BM25 cache
   - Retrieval settings (`retrieval_k`, `bm25_k`)

3. Adjust [src/config/config.yaml](src/config/config.yaml) to control the run:

   | Key | Description |
   | --- | --- |
   | `dataset_path` | Hugging Face dataset path (e.g. `rajpurkar/squad_v2`) |
   | `split` | Dataset split to use (e.g. `validation`) |
   | `num_docs` | Number of documents to ingest (`-1` = all) |
   | `num_eval` | Number of samples to evaluate (`-1` = all) |
   | `load_docs` | Whether to (re-)run ingestion before evaluation |
   | `ISBM25` | Whether to build/use a BM25 index alongside vector search |
   | `run_async` | Whether to run DeepEval metrics/evaluation asynchronously |

## Usage

Run the evaluation pipeline:

```bash
python -m src.app
```

Or point it at a different config file:

```bash
python -m src.app --config path/to/config.yaml
```

If `load_docs` is enabled in the config, ingestion runs first to (re)build the FAISS vector store and, optionally, the BM25 index. The pipeline then builds test cases by retrieving context and generating answers for each evaluation question, runs the DeepEval metrics, and writes the scored results to a timestamped `evaluation_results_<timestamp>.xlsx` file in the project root.

You can also run ingestion on its own:

```bash
python -m src.ingestion.loader --num_docs 100 --ISBM25 true
```

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
