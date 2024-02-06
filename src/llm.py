import time

from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ChatMessageHistory



class LLM:
    def __init__(self: str, model_name: str):
        self.messages = ChatMessageHistory()
        self.chat = ChatOllama(model=model_name)
        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful assistant. Answer all questions to the best of your ability.",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )
        self.chain = self.prompt | self.chat | StrOutputParser()

    def generate_response(self, input_text: str):
        self.messages.add_user_message(input_text)
        return self.chain.stream({"messages": self.messages.messages})
    
    def add_message(self, msg: str, human: bool = False):
        if human:
            self.messages.add_user_message(msg)
        else:
            self.messages.add_ai_message(msg)