from agentic_chatbot_backend import workflow,get_all_thread
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
import streamlit as st
import uuid


# Generate a unique thread ID for each new conversation
def generate_thread_id():
    return str(uuid.uuid4())


# Add a new thread ID to the conversation list
def add_thread(thread_id):
    # Prevent the same thread from being added multiple times
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)


def generate_thread_name(user_input):
    """
    Generate a short conversation name from the user's first message.
    """
    name = user_input.strip()

    # Keep the name short for the sidebar
    if len(name) > 30:
        name = name[:30].rsplit(" ", 1)[0] + "..."

    return name


# Create a completely new chat conversation
def reset_chat():

    st.session_state["thread_id"] = generate_thread_id()

    st.session_state["message_history"] = []

    st.session_state["thread_names"][
        st.session_state["thread_id"]
    ] = "New Chat"


# Load a previous conversation from the LangGraph checkpointer
def load_conversation(thread_id):

    # Get the saved state for the selected thread
    state = workflow.get_state(
        config={
            "configurable": {
                "thread_id": thread_id
            }
        }
    )

    # Return saved messages
    # Return an empty list if no messages are available
    return state.values.get("messages", [])


# Display the main application title
st.title("Agentic Chatbot with LangGraph")


# Create message_history when the app runs for the first time
if "message_history" not in st.session_state:
    st.session_state["message_history"] = []


# Create a thread ID when the app runs for the first time
if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()


# Create a list for storing all conversation thread IDs
if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = get_all_thread()

if "thread_names" not in st.session_state:
    st.session_state["thread_names"] = {}


# Initialize thread name storage
if st.session_state["thread_id"] not in st.session_state["thread_names"]:
    st.session_state["thread_names"][
        st.session_state["thread_id"]
    ] = "New Chat"


# ========================= Sidebar threading feature =========================

# Display the sidebar title
st.sidebar.title("My Conversations")


# Create a button for starting a new conversation
if st.sidebar.button("New Chat"):

    # Reset the current chat and create a new thread
    reset_chat()

    # Rerun the Streamlit app to update the interface
    st.rerun()


# Display all conversation threads in reverse order
# This shows the newest conversation first
for thread_id in st.session_state["chat_threads"][::-1]:

    thread_name = st.session_state["thread_names"].get(
        thread_id,
        "New Chat"
    )

    if st.sidebar.button(
        thread_name,
        key=thread_id
    ):
        

        # Set the selected thread as the current thread
        st.session_state["thread_id"] = thread_id

        # Load the messages saved under the selected thread
        messages = load_conversation(thread_id)

        # Temporary list for converting LangChain messages
        # into Streamlit's required message format
        temp_messages = []

        # Loop through all saved messages
        for message in messages:

            # Check whether the message was sent by the user
            if isinstance(message, HumanMessage):
                role = "user"

            # Check whether the message was sent by the AI
            elif isinstance(message, AIMessage):
                role = "assistant"

            # Ignore other message types, such as ToolMessage
            else:
                continue

            # Convert the LangChain message into a dictionary
            temp_messages.append({
                "role": role,
                "content": message.content
            })

        # Replace the current UI history with the selected conversation
        st.session_state["message_history"] = temp_messages

        # Rerun the application to display the loaded messages
        st.rerun()


# ========================= Main chat interface =========================

# Display all messages from the currently selected conversation
for message in st.session_state["message_history"]:

    # Create either a user chat bubble or assistant chat bubble
    with st.chat_message(message["role"]):

        # Display the message content
        st.text(message["content"])


# Create the chat input box
user_input = st.chat_input("Type here")


# Run this block after the user submits a message
if user_input:

    # Save the user's message in Streamlit session state
    st.session_state["message_history"].append({
        "role": "user",
        "content": user_input
    })
    # Generate a name for the conversation from the first user message
    thread_id = st.session_state["thread_id"]
    add_thread(thread_id)

    if st.session_state["thread_names"].get(thread_id) == "New Chat":

        st.session_state["thread_names"][thread_id] = generate_thread_name(
            user_input
        )

    # Display the user's message in the chat interface
    with st.chat_message("user"):
        st.text(user_input)

    # Pass the current thread ID to LangGraph
    # LangGraph uses this ID to save and retrieve conversation memory
    CONFIG = {
        "configurable": {
            "thread_id": st.session_state["thread_id"]
        }
    }

    # Create the assistant chat-message container
    with st.chat_message("assistant"):

        # Stream the assistant response token by token
        ai_message = st.write_stream(

            # Return only the content of AI message chunks
            message_chunk.content

            # Stream messages from the LangGraph workflow
            for message_chunk, metadata in workflow.stream(
                {
                    # Send the latest user message to the workflow
                    "messages": [
                        HumanMessage(content=user_input)
                    ]
                },

                # Use the current conversation thread
                config=CONFIG,

                # Stream individual message chunks
                stream_mode="messages"
            )

            # Display only AI messages
            # This prevents tool and user messages from appearing
            if isinstance(message_chunk, AIMessage)
        )

    # Save the complete assistant response in Streamlit session state
    st.session_state["message_history"].append({
        "role": "assistant",
        "content": ai_message
    })