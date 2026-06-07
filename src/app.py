import argparse
import os
import pickle
from datetime import datetime

import yaml
import pandas as pd
from dotenv import load_dotenv
from datasets import load_dataset
from tqdm import tqdm
from deepeval import evaluate
from deepeval.evaluate import AsyncConfig
from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
    FaithfulnessMetric,
)
from deepeval.models import AmazonBedrockModel
from deepeval.test_case import LLMTestCase

from ingestion.loader import loader
from retrieval.vector_search import context_retreiver, vector_store_func
from generation.generator import invoking_LLM


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def dataset_evaluator():
    load_dotenv()

    parser = argparse.ArgumentParser(description="RAG evaluation pipeline")
    parser.add_argument("--config", type=str, default="src/config/config.yaml", help="Path to YAML config file")
    args = parser.parse_args()

    cfg = load_config(args.config)

    num_docs = cfg.get("num_docs", -1)
    ISBM25 = cfg.get("ISBM25", False)
    load_docs = cfg.get("load_docs", False)
    num_eval = cfg.get("num_eval", -1)
    run_async = cfg.get("run_async", False)
    dataset_path = cfg.get("dataset_path") or os.getenv("dataset_path")
    split = cfg.get("split") or os.getenv("dataset_split")
    retrieval_k = int(os.getenv("retrieval_k", "3"))
    bm25_cache_path = os.getenv("BM25_CACHE_PATH")

    if load_docs:
        loader(num_docs, ISBM25, cfg.get("dataset_path"), cfg.get("split"))

    slice_str = f"[:{num_eval}]" if num_eval != -1 else ""
    dataset = load_dataset(dataset_path, split=f"{split}{slice_str}")

    vector_store = vector_store_func()
    vector_retriever = vector_store.as_retriever(search_kwargs={"k": retrieval_k})

    bm25_retriever = None
    if ISBM25 and bm25_cache_path and os.path.exists(bm25_cache_path):
        with open(bm25_cache_path, "rb") as f:
            bm25_retriever = pickle.load(f)

    test_cases = []
    no_ans = 0
    for row in tqdm(dataset, desc="Building test cases"):
        if row["answers"]["text"] != []:
            question = row["question"]
            text, ret_context = context_retreiver(
                question, vector_retriever=vector_retriever, bm25_retriever=bm25_retriever
            )
            ai_answer = invoking_LLM(text, question)
            test_cases.append(
                LLMTestCase(
                    input=question,
                    actual_output=ai_answer,
                    retrieval_context=[doc.page_content for doc in ret_context],
                    expected_output=row["answers"]["text"][0],
                )
            )
        else:
            no_ans += 1
    print(f"Skipped {no_ans} questions with no ground-truth answers")

    eval_model = AmazonBedrockModel(
        model=os.getenv("eval_model"),
        region=os.getenv("region"),
        generation_kwargs={"temperature": 0},
    )
    metrics = [
        AnswerRelevancyMetric(threshold=0.8, model=eval_model, async_mode=run_async),
        ContextualPrecisionMetric(threshold=0.8, model=eval_model, async_mode=run_async),
        FaithfulnessMetric(threshold=0.8, model=eval_model, async_mode=run_async),
        ContextualRelevancyMetric(threshold=0.8, model=eval_model, async_mode=run_async),
        ContextualRecallMetric(threshold=0.8, model=eval_model, async_mode=run_async),
    ]

    eval_results = evaluate(test_cases, metrics=metrics, async_config=AsyncConfig(run_async=run_async))

    data = []
    for result in eval_results.test_results:
        md = result.metrics_data
        data.append({
            "input": result.input,
            "actual_output": result.actual_output,
            "expected_output": result.expected_output,
            "answer_relevancy_score": md[0].score if md else None,
            "contextual_precision_score": md[1].score if md else None,
            "faithfulness_score": md[2].score if md else None,
            "context_relevancy_score": md[3].score if md else None,
            "context_recall_score": md[4].score if md else None,
            "answer_relevancy_reason": md[0].reason if md else None,
            "contextual_precision_reason": md[1].reason if md else None,
            "faithfulness_reason": md[2].reason if md else None,
            "context_relevancy_reason": md[3].reason if md else None,
            "context_recall_reason": md[4].reason if md else None,
        })

    df = pd.DataFrame(data)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"evaluation_results_{timestamp}.xlsx"
    df.to_excel(output_path, index=False)
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    dataset_evaluator()
