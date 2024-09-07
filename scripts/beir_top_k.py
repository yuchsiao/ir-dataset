import argparse
import dataclasses
import json
import logging
import os

import pandas as pd

from beir import LoggingHandler
from beir import util
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval import models


logging.basicConfig(format="%(asctime)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                    level=logging.INFO,
                    handlers=[LoggingHandler()])


def get_args():
    """Return command line args."""
    parser = argparse.ArgumentParser(
        description="Compute top k similar documents for BEIR datasets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataset", type=str, default=None,
                        help="Dataset to use. If not given, all datasets in BEIR with commercial compatible licenses.")
    parser.add_argument("--datasets_dir", type=str, default="..",
                        help="Datasets directory.")
    parser.add_argument("--model_name", type=str, default="msmarco-distilbert-base-tas-b",
                        help="Model name")
    parser.add_argument("--batch_size", type=int, default=1024,
                        help="Batch size for model inference")
    parser.add_argument("--top_k", type=int, default=500,
                        help="Top K similar documents")
    parser.add_argument("--include_text", action="store_true",
                        help="Include query, title, document text for each qid, docid pair")
    return parser.parse_args()


@dataclasses.dataclass
class BeirTopKSimilarityMetadata:
    """Metadata to be stored side by side with the output file."""
    dataset_name: str
    num_queries: int
    num_documents: int
    num_positive_annotations: int
    model_name: str
    top_k: bool
    include_text: bool
    ndcg: dict[str, float]
    mean_ap: dict[str, float]
    recall: dict[str, float]
    precision: dict[str, float]

    def __init__(
            self, dataset_name, queries, corpus, qrels, model_name, top_k, include_text,
            ndcg, mean_ap, recall, precision):
        self.dataset_name = dataset_name
        self.num_queries = len(queries)
        self.num_documents = len(corpus)
        self.num_positive_annotations = sum(len(rel_docids) for qid, rel_docids in qrels.items())
        self.model_name = model_name
        self.top_k = top_k
        self.include_text = include_text
        self.ndcg = ndcg
        self.mean_ap = mean_ap
        self.recall = recall
        self.precision = precision


def download_data(dataset_name: str, datasets_path: str = os.path.join("..", "datasets")) -> str:
    """Download dataset from BEIR official site."""
    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset_name}.zip"
    data_path = util.download_and_unzip(url, datasets_path)
    return data_path


def load_data(data_path: str):
    """Load dataset by path using BEIR utility function, assuming the folder containing corpus.json, qrels/, queries.jsonl."""
    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")
    return corpus, queries, qrels


def compute_similarity(
        model_name, batch_size, corpus, queries, qrels, score_function=None
        ) -> tuple[dict[str, dict[str, float]], dict[str, float], dict[str, float], dict[str, float], dict[str, float]]:
    """Compute embedding similarity between queries and corpus by score_function using model_name by batch_size."""
    if score_function is None:
        score_function = "dot"
    if score_function not in ("dot", "cos_sim"):
        raise ValueError(f"score_function must be either 'dot' or 'cos_sim'. Received: '{score_function}'")

    model = DRES(models.SentenceBERT(model_name, trust_remote_code=True), batch_size=batch_size, trust_remote_code=True)
    retriever = EvaluateRetrieval(model, score_function=score_function)
    q_doc_sim = retriever.retrieve(corpus, queries)
    ndcg, mean_avg_prec, recall, precision = retriever.evaluate(qrels, q_doc_sim, retriever.k_values)

    return q_doc_sim, ndcg, mean_avg_prec, recall, precision


def get_top_k_similar_docs(doc_sim: dict[str, float], k: int) -> dict[str, float]:
    """For a dict of docids to similarity scores, get top-k docs with highest similarity scores."""
    return dict(sorted(doc_sim.items(), key=lambda x: x[1], reverse=True)[:k])


def get_labels_similarities(query, top_k_similar_docs, qrels):
    """"""
    labels = {}
    similarity_scores = {}
    for doc, similarity_score in top_k_similar_docs.items():
        if doc in qrels[query] and qrels[query][doc] == 1:
            labels[doc] = 1
        else:
            labels[doc] = 0
        similarity_scores[doc] = similarity_score
    return labels, similarity_scores
        

def get_top_k_data_frame(q_doc_sim, top_k, corpus, queries, qrels, include_text=False):
    qids = []
    docids = []
    rels = []
    sims = []
    if include_text:
        q_texts = []
        title_texts = []
        doc_texts = []
    
    for qid, doc_sim in q_doc_sim.items():
        top_k_doc_scores = get_top_k_similar_docs(q_doc_sim[qid], top_k)
        docid_labels, similarities = get_labels_similarities(qid, top_k_doc_scores, qrels)
        for docid, rel in docid_labels.items():
            qids.append(qid)
            docids.append(docid)
            rels.append(rel)
            sims.append(similarities[docid])
            if include_text:
                q_texts.append(queries[qid])
                title_texts.append(corpus[docid].get("title", ""))
                doc_texts.append(corpus[docid].get("text", ""))
    data = {
        "qid": qids, 
        "docid": docids, 
        "rel": rels,
        "sim": sims,
    }
    
    if include_text:
        data.update({
            "query": q_texts,
            "title": title_texts,
            "document": doc_texts,
        })
    return pd.DataFrame(data)    


def main(args):
    datasets = [
        "trec-covid",
        "nq",
        "hotpotqa",
        "arguana",
        "webis-touche2020",
        "cqadupstack",
        "dbpedia-entity",
        "scidocs",
        "fever",
    ]
    dataset_name = args.dataset
    datasets_dir = args.datasets_dir
    model_name = args.model_name
    batch_size = args.batch_size
    top_k = args.top_k
    include_text = args.include_text

    def process_sub_dataset(dataset_name: str, data_path: str, dirname: str = ""):
        """Process dataset_name_{dirname} under folder data_path/dirname. Return silently if no needed files found."""
        output_path = data_path
        if dirname:
            output_path = data_path + "_" + dirname
            data_path = os.path.join(data_path, dirname)
            dataset_name += "_" + dirname

        if tuple(sorted(os.listdir(data_path))) != ("corpus.jsonl", "qrels", "queries.jsonl"):
            return

        logging.info(f"## Process {data_path}.")
        corpus, queries, qrels = load_data(data_path)
        # Compute similarity scores.
        logging.info("## Compute similarity scores.")
        q_doc_sim, ndcg, mean_ap, recall, precision = compute_similarity(model_name, batch_size, corpus, queries, qrels)    
        # Get DF.
        logging.info("## Get the dataframe.")
        df = get_top_k_data_frame(q_doc_sim, top_k, corpus, queries, qrels, include_text)
        metadata = BeirTopKSimilarityMetadata(
            dataset_name, queries, corpus, qrels, model_name, top_k, include_text,
            ndcg, mean_ap, recall, precision)

        # Write to files
        logging.info("## Write to files.")
        df.to_csv(output_path + ".tsv", sep="\t", index=False)
        with open(output_path + "_metadata.json", "w") as fout:
            json.dump(dataclasses.asdict(metadata), fout, indent=4)

    def process_dataset(dataset_name):
        # Load dataset.
        logging.info("## Load dataset.")
        data_path = download_data(dataset_name, datasets_dir)

        # Process the current directory.
        process_sub_dataset(dataset_name, data_path)

        # Process one more level down if there exists subtasks such as `cqadupstack` dataset.
        for dirname in os.listdir(data_path):
            if not os.path.isdir(os.path.join(data_path, dirname)):
                continue
            process_sub_dataset(dataset_name, data_path, dirname)
    
    if dataset_name is None:
        for dataset_idx, dataset_name in enumerate(datasets):
            logging.info(f"#### DATASET {dataset_idx}: {dataset_name}")
            process_dataset(dataset_name)
    else:
        # Only process the specified dataset.
        logging.info(f"#### DATASET: {dataset_name}")
        process_dataset(dataset_name)


if __name__ == "__main__":
    args = get_args()
    main(args)
