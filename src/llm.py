import os
from dotenv import load_dotenv

load_dotenv('.env')

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain.memory import ChatMessageHistory
from langchain_community.document_loaders import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.chroma import Chroma
from langchain import hub
from langchain_core.runnables import RunnablePassthrough


class LLM:
    def __init__(self: str, model_name: str):
        self.messages = ChatMessageHistory()
        self.chat = ChatOpenAI(model="gpt-3.5-turbo-1106", api_key=os.getenv("OPENAI_SECRET"), max_tokens=256)
        self.prompt = hub.pull("rlm/rag-prompt")
        self.loader = CSVLoader(file_path="/home/kristian/Documents/img-txt-categorisation-chat/txt_data/data/senticap_dataset-val.csv.multi.topics", source_column="filename", csv_args={
            'delimiter': ',',
            'quotechar': '"',
            'fieldnames': ['index', 'filename', 'is_positive_sentiment', 'caption', 'topic']
        })
        """
        Index: 0
        filename: COCO_XXXX.jpg
        
        """
        self.data = self.loader.load()
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
        self.all_splits = self.text_splitter.split_documents(self.data)
        self.vectorstore = Chroma.from_documents(documents=self.all_splits, embedding=OpenAIEmbeddings(api_key=os.getenv("OPENAI_SECRET")))
        self.retriever = self.vectorstore.as_retriever(k=4)
        # TODO: Figure out if the retriever works. 

        self.chain = (
            {"context": self.parse_retriever_input | self.retriever | self.format_docs, "question": RunnablePassthrough()}
            | self.prompt
            | self.chat
            | StrOutputParser()
        )

    def parse_retriever_input(self, params: dict):
        return params["messages"][-1].content

    def format_docs(self, docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def generate_response(self, input_text: str):
        self.messages.add_user_message(input_text)
        return self.chain.stream({"messages": self.messages.messages})
    
    def add_message(self, msg: str, human: bool = False):
        if human:
            self.messages.add_user_message(msg)
        else:
            self.messages.add_ai_message(msg)