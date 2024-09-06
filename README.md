# ir-datasets

Compute top-k similar documents for queries.

## Requirements

```python
pip install beir
```

## Usage

```python
python scripts/beir_top_k.py
```

## Results

See `datasets` folder. Use `arguana` dataset as an example.

The `arguana_metadata.json`  file shows the metadata:

```python
{
    "dataset_name": "arguana",                      # dataset name
    "num_queries": 1406,                            # number of queries contained in the dataset
    "num_documents": 8674,                          # number of documents contained in the dataset
    "num_positive_annotations": 1406,               # number of query-document pairs annotated as relevant
    "model_name": "msmarco-distilbert-base-tas-b",  # embedding model used for top-k computation
    "top_k": 500,                                   # top k similar query-doc pairs
    "include_text": false,                          # only query id, document id, relevance annotation, and 
                                                    # similarity score stored in tsv if false. otherwise, 
                                                    # additional query text, title text, and document text 
                                                    # are also included.
    "ndcg": {                                       # trec evaluation scores for ndcg@first-k
        "NDCG@1": 0.19915,
        "NDCG@3": 0.31702,
        "NDCG@5": 0.36434,
        "NDCG@10": 0.427,
        "NDCG@100": 0.4789,
        "NDCG@1000": 0.48601
    },
    "mean_ap": {                                    # trec evaluation scores for map@first-k
        "MAP@1": 0.19915,
        "MAP@3": 0.2871,
        "MAP@5": 0.31328,
        "MAP@10": 0.33946,
        "MAP@100": 0.35126,
        "MAP@1000": 0.35159
    },
    "recall": {                                     # trec evaluation scores for recall@first-k
        "Recall@1": 0.19915,
        "Recall@3": 0.40398,
        "Recall@5": 0.5192,
        "Recall@10": 0.71124,
        "Recall@100": 0.94168,
        "Recall@1000": 0.99431
    },
    "precision": {                                  # trec evaluation scores for precision@first-k
        "P@1": 0.19915,
        "P@3": 0.13466,
        "P@5": 0.10384,
        "P@10": 0.07112,
        "P@100": 0.00942,
        "P@1000": 0.00099
    }
}
```

The `arguana.tsv` in `datasets_top_k.tsv.tar.gz` is a TSV file that contains the following columns

| Column Name | Description                                                |
|-------------|------------------------------------------------------------|
| qid         | Query ID                                                   |
| docid       | Document ID                                                |
| rel         | Relevence Annotation. 1 for a relevant query-document pair |
| sim         | Raw dot-product similarity score                           |
| query       | Query text, if `include_text` is true                      |
| title       | Title text, if `include_text` is true                      |
| document    | Document text, if `include_text` is true                   |

## Caveat

Note that `rel` is derived from single-class IR tasks, meaning that all 0s are soft negatives.