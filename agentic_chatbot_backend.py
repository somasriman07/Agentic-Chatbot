from langgraph.graph import START,END,StateGraph
from typing import TypedDict,Annotated
from langchain_core.messages import BaseMessage,HumanMessage
from dotenv import load_dotenv
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3


from langchain_groq import ChatGroq

load_dotenv()

llm = ChatGroq(
    model="qwen/qwen3.8-27b",
    temperature=0.7,
    max_tokens=256,
)


from langgraph.graph.message import add_messages
class ChatState(TypedDict):
    messages : Annotated[list[BaseMessage],add_messages]


def chat_node(state : ChatState):
    # Take the user query from the state
    message = state['messages']

    response = llm.invoke(message)

    return {'messages':[response]}

conn = sqlite3.connect(database='Chatbot.db',check_same_thread=False)
checkpointer = SqliteSaver(conn)
graph = StateGraph(ChatState)

graph.add_node('chat_node',chat_node)

graph.add_edge(START,'chat_node')
graph.add_edge('chat_node',END)

workflow = graph.compile(checkpointer=checkpointer)

def get_all_thread():
    all_threads = set()
    for ckpt in checkpointer.list(None):
        all_threads.add(ckpt.config['configurable']['thread_id'])
    return list(all_threads)

