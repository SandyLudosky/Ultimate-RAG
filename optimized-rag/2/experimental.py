import os
from langchain_community.document_loaders import TextLoader
from dotenv import load_dotenv
from langchain_community.chat_models import ChatOpenAI
from langchain_experimental.text_splitter import SemanticChunker
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts.chat import (
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain_community.vectorstores import Chroma
from colorama import Fore
import warnings
warnings.filterwarnings("ignore")

load_dotenv()

template: str = "You are a helpful assistant who answers {question} based on {context}"

system_message_prompt = SystemMessagePromptTemplate.from_template(template)
human_message_prompt = HumanMessagePromptTemplate.from_template(
    input_variables=["question", "context"],
    template="{question}",
)
chat_prompt_template = ChatPromptTemplate.from_messages(
    [system_message_prompt, human_message_prompt]
)

model = ChatOpenAI()


def load_documents():
    """ Load documents and split by semantic chunks"""

    with open("./docs/state_of_the_union.txt") as f:
        state_of_the_union = f.read()
        text_splitter = SemanticChunker(
            OpenAIEmbeddings(), 
            breakpoint_threshold_type="percentile"
        )
        docs = text_splitter.create_documents([state_of_the_union])
        return docs


documents = load_documents()
db = Chroma.from_documents(documents, OpenAIEmbeddings())
retriever = db.as_retriever()


def print_chunks():
    _ = [print(f"{index + 1}/{len(documents)} - {document.page_content}\n") for index, document in enumerate(documents)]
    print(f"{Fore.GREEN}- Number of chunks: {len(documents)}")

 
def main(query):
    print_chunks()
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | chat_prompt_template
        | model
        | StrOutputParser()
    )
    response = chain.invoke(query)
    print(f"{Fore.CYAN}{response}")
   

if __name__ == "__main__":
    query = "What did the president say about Ketanji Brown Jackson"
    # query = "What happened to Heath"
    main(query)
