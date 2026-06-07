from langchain_core.prompts import ChatPromptTemplate


template = ChatPromptTemplate([
    ("system","you are a helpful assistant who asnwers in short and you need to answer the user using this context : {context}"),
    ("human", "{human_text}")])