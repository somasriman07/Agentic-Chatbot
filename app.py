from agentic_chatbot_backend import workflow, get_all_thread
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
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
    # Return saved messages, or empty list if none
    return state.values.get("messages", [])


# ========================= App title =========================

st.title("Agentic Chatbot with LangGraph")


# ========================= Session state init =========================

if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = get_all_thread()

if "thread_names" not in st.session_state:
    st.session_state["thread_names"] = {}

if st.session_state["thread_id"] not in st.session_state["thread_names"]:
    st.session_state["thread_names"][
        st.session_state["thread_id"]
    ] = "New Chat"


# ========================= Sidebar threading feature =========================

st.sidebar.title("My Conversations")

if st.sidebar.button("New Chat"):
    reset_chat()
    st.rerun()

# Display all conversation threads in reverse order (newest first)
for thread_id in st.session_state["chat_threads"][::-1]:

    thread_name = st.session_state["thread_names"].get(thread_id, "New Chat")

    if st.sidebar.button(thread_name, key=thread_id):

        st.session_state["thread_id"] = thread_id

        messages = load_conversation(thread_id)

        temp_messages = []

        for message in messages:
            if isinstance(message, HumanMessage):
                role = "user"
            elif isinstance(message, AIMessage):
                role = "assistant"
            else:
                # Ignore ToolMessage and other types
                continue

            temp_messages.append({
                "role": role,
                "content": message.content
            })

        st.session_state["message_history"] = temp_messages
        st.rerun()


# ========================= Main chat interface =========================

# Display all messages from the current conversation
for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.text(message["content"])


# Chat input box
user_input = st.chat_input("Type here")


if user_input:

    # Save the user message
    st.session_state["message_history"].append({
        "role": "user",
        "content": user_input
    })

    thread_id = st.session_state["thread_id"]
    add_thread(thread_id)

    # Name the conversation from the first message
    if st.session_state["thread_names"].get(thread_id) == "New Chat":
        st.session_state["thread_names"][thread_id] = generate_thread_name(
            user_input
        )

    # Show the user bubble
    with st.chat_message("user"):
        st.text(user_input)

    CONFIG = {
        "configurable": {
            "thread_id": st.session_state["thread_id"]
        }
    }

    # ========================= Assistant response =========================
    with st.chat_message("assistant"):

        status_holder = {"box": None}

        def ai_only_stream():
            for message_chunk, metadata in workflow.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode="messages",
            ):
                # When LLM decides to call a tool, show it immediately
                if isinstance(message_chunk, AIMessage):
                    tool_calls = getattr(message_chunk, "tool_calls", [])
                    if tool_calls:
                        for tc in tool_calls:
                            tool_name = tc.get("name", "tool")
                            if status_holder["box"] is None:
                                status_holder["box"] = st.status(
                                    f"🔧 Using `{tool_name}` …",
                                    expanded=True
                                )
                            else:
                                status_holder["box"].update(
                                    label=f"🔧 Using `{tool_name}` …",
                                    state="running",
                                    expanded=True,
                                )

                # Mark tool as done once result arrives
                if isinstance(message_chunk, ToolMessage):
                    tool_name = getattr(message_chunk, "name", "tool")
                    if status_holder["box"] is not None:
                        status_holder["box"].update(
                            label=f"✅ `{tool_name}` done",
                            state="complete",
                            expanded=False,
                        )

                # Stream only AI text tokens
                if isinstance(message_chunk, AIMessage) and message_chunk.content:
                    yield message_chunk.content

        ai_message = st.write_stream(ai_only_stream())

    # Save the assistant response
    st.session_state["message_history"].append({
        "role": "assistant",
        "content": ai_message
    })
