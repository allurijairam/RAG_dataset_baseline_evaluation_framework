from datasets import load_dataset, concatenate_datasets, DatasetDict
import argparse
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import BedrockEmbeddings
from dotenv import load_dotenv
from tqdm import tqdm
import os
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
import warnings
import pickle
def loader(num_docs,ISBM25=False,dataset_path=None,split=None):
    print("num_docs: ",num_docs)
    print("ISBM25: ",ISBM25)

    warnings.filterwarnings("ignore")
    load_dotenv()

    if not dataset_path:
        dataset_path = os.getenv("dataset_path")
    if not split:
        split = os.getenv("dataset_split")

    slice_str = "" if num_docs == -1 else num_docs

    try:
        dataset = load_dataset(dataset_path, split=f"{split}[:{slice_str}]")
    except:
        raw = load_dataset(dataset_path)
        dataset = concatenate_datasets(list(raw.values())) if isinstance(raw, DatasetDict) else raw
        if slice_str:
            dataset = dataset.select(range(min(slice_str, len(dataset))))

    docs = []
    data_set = set()
    for row in dataset:
        if row['context'] not in data_set:
            docs.append(Document(page_content=row['context']))
            data_set.add(row['context'])

    embeddings = BedrockEmbeddings(
        model_id=os.getenv("embedding_model"),
        region_name=os.getenv("region"),
    )
    vector_store = None
    batch_size = 50
    for i in tqdm(range(0,len(docs),batch_size)):
        
        batch = docs[i:i+batch_size]
        if not vector_store:
            vector_store = FAISS.from_documents(batch,embeddings)
        else:

            vector_store.add_documents(batch)
    db_path =  os.getenv("db_save_location")
    vector_store.save_local(db_path)
    print(f"Saved to {db_path}")

    ################## BM25 SEARCH INDEX ##########################
    if ISBM25:
        load_bm25(num_docs,dataset)

def load_bm25(num_docs,dataset):
    #### Test this bm25. never tested it
    print("Loading BM25")
    BM25_CACHE_PATH = os.getenv("BM25_CACHE_PATH")
    if num_docs==-1:
        num_docs = ""

    docs = []
    data_set = set()
    for row in dataset:
        if row['context'] not in data_set:
            docs.append(Document(page_content=row['context']))
            data_set.add(row['context'])
    bm25_k = int(os.getenv("bm25_k", "3"))
    retriever = BM25Retriever.from_documents(docs, k=bm25_k)
    os.makedirs(os.path.dirname(BM25_CACHE_PATH), exist_ok=True)
    with open(BM25_CACHE_PATH, "wb") as f:
        pickle.dump(retriever, f)
    print(f"Saved BM25 to {BM25_CACHE_PATH}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingestion of squad dataset")
    parser.add_argument("--num_docs",type=int,default=-1)
    parser.add_argument("--ISBM25",type=bool,default=False)
    parser.add_argument("--dataset_path",type=str,default=None)
    parser.add_argument("--split",type=str,default=None)
    args = parser.parse_args()
    loader(args.num_docs,args.ISBM25,args.dataset_path,args.split)
