
from langchain_aws import BedrockEmbeddings
from functools import lru_cache
import os 
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
def context_retreiver(question,clean_documents=True,vector_retriever=None,bm25_retriever=None):
    if bm25_retriever and vector_retriever:
        ensemble_retriever = EnsembleRetriever(
                retrievers=[vector_retriever,bm25_retriever],
                weights=[0.5,0.5]
            )
    else:
        ensemble_retriever = vector_retriever
        
    retrieved_docs = ensemble_retriever.invoke(question)
    if clean_documents:
        cleaned_text = clean_docs(retrieved_docs)
        return cleaned_text,retrieved_docs
    return retrieved_docs

@lru_cache(maxsize=1)
def vector_store_func():
    load_dotenv()
    
    # loading variables from env
    region = os.getenv("region")
    embedding_model = os.getenv("embedding_model")
    db_path = os.getenv("db_save_location")

    embeddings = BedrockEmbeddings(
        model_id=embedding_model,
        region_name=region,
    )


    vector_store = FAISS.load_local(
        db_path,
        embeddings,
        allow_dangerous_deserialization=True,
    )

    return vector_store


def clean_docs(retrieved_docs):
    text = ""
    for doc in retrieved_docs:
        text += " doc: " + doc.page_content
    return text

