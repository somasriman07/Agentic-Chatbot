from agentic_chatbot_backend import workflow
from langchain_core.messages import BaseMessage,HumanMessage

config = {'configurable':{'thread_id':'1'}}

for message,metadata in workflow.stream(
    {'messages':[HumanMessage(content = 'Explain RAG in 100 words?')]},
    config=config,
    stream_mode='messages'):
    print(message.content, end="", flush=True)

