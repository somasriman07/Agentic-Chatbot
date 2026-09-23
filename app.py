from agentic_chatbot_backend import workflow
from langchain_core.messages import BaseMessage,HumanMessage
import streamlit as st

if 'message_history' not in st.session_state:
    st.session_state['message_history']= []


config = {'configurable':{'thread_id':"thread_id-1"}}


st.title('Agentic Chatbot with LangGraph')


for i in st.session_state['message_history']:
    with st.chat_message(i['role']):
        st.text(i['content'])

user_input = st.chat_input('Type here ...')

if user_input:
    st.session_state['message_history'].append({'role':'user','content':user_input})
    with st.chat_message('user'):
        st.text(user_input)

    response = workflow.invoke({'messages':[HumanMessage(content=user_input)]},config=config)

    ai_message = response['messages'][-1].content 
    st.session_state['message_history'].append({'role':'ai','content':ai_message})

    with st.chat_message('AI'):
        st.text(ai_message)
