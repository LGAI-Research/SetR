import os
import json
import argparse
from tqdm import tqdm
from rankify.dataset.dataset import Dataset, Document
from rankify.retrievers.retriever import Retriever

OUTPUT_DIR = "./retrieval_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def save_documents(documents: list[Document], dataset_name: str, retriever_name: str, n_docs: int):
    """
    Save the retrieval results to a JSONL file.
    Each line contains question, answers, and retrieved contexts.
    """
    output_path = os.path.join(OUTPUT_DIR, f"{dataset_name}_{retriever_name}_top{n_docs}.jsonl")
    with open(output_path, "w", encoding="utf-8") as f:
        for doc in documents:
            item = {
                "question": doc.question.text,
                "answers": doc.answers.answer_strings,
                "contexts": [ctx.text for ctx in doc.contexts],
            }
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"[Saved] {output_path}")

def run_retrieval(dataset_name: str, retriever_name: str, n_docs: int):
    """
    Run document retrieval on a given dataset using the specified retriever.
    """
    print(f"\n=== Dataset: {dataset_name} | Retriever: {retriever_name} | Top-{n_docs} ===")
    
    # Load questions from QA file
    dataset = Dataset('bm25', f'{dataset_name}-dev', 20)
    questions: list[Document] = dataset.download(force_download=False)

    # Initialize the retriever
    if retriever_name == "bm25":
        retriever = Retriever(method="bm25", n_docs=n_docs, index_type="wiki")
    elif retriever_name == "bge":
        retriever = Retriever(method="bge", model="BAAI/bge-large-en-v1.5", n_docs=n_docs, index_type="wiki")
    else:
        raise ValueError(f"Unknown retriever: {retriever_name}")

    # Perform retrieval for each question
    for doc in tqdm(questions, desc=f"Retrieving [{retriever_name}] for [{dataset_name}]"):
        doc.contexts = retriever.retrieve(question=doc.question)
        break

    # Save the results
    save_documents(questions, dataset_name, retriever_name, n_docs)

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["musique", "2wikimultihopqa", "hotpotqa"],
                        help="Name of the dataset to process")
    parser.add_argument("--retriever", required=True, choices=["bm25", "bge"],
                        help="Retriever method to use")
    parser.add_argument("--n_docs", type=int, default=20,
                        help="Number of top documents to retrieve")
    args = parser.parse_args()

    run_retrieval(args.dataset, args.retriever, args.n_docs)