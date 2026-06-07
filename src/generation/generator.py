from langchain_aws import ChatBedrockConverse
from langchain_core.output_parsers import StrOutputParser
from generation.prompt import template
import boto3
import warnings
import os
from dotenv import load_dotenv

def invoking_LLM(retrieved_text,question):
    warnings.filterwarnings("ignore")

    load_dotenv()
    region = os.getenv("region")
    client = boto3.client("bedrock-runtime", region_name=region)



    llm = ChatBedrockConverse(
        model_id=os.getenv("llm_model"),
        region_name=region,
    )
    chain = template | llm | StrOutputParser()

    ai_msg = chain.invoke({
        "human_text": question,
        "context": retrieved_text
    })
    return ai_msg
