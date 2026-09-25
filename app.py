from agentic_chatbot_backend import workflow, get_all_thread, ingest_rag_document, rag_index_exists
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
import streamlit as st
import tempfile
import os
import uuid


# ========================= Helper functions =========================

def generate_thread_id():
    return str(uuid.uuid4())


def add_thread(thread_id):
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)


def generate_thread_name(user_input):
    name = user_input.strip()
    if len(name) > 30:
        name = name[:30].rsplit(" ", 1)[0] + "..."
    return name


def reset_chat():
    st.session_state["thread_id"] = generate_thread_id()
    st.session_state["message_history"] = []
    st.session_state["thread_names"][st.session_state["thread_id"]] = "New Chat"


def load_conversation(thread_id):
    state = workflow.get_state(
        config={"configurable": {"thread_id": thread_id}}
    )
    return state.values.get("messages", [])


# ========================= Page config =========================

st.set_page_config(
    page_title="Agentic Chatbot",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 Agentic Chatbot with LangGraph")


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
    st.session_state["thread_names"][st.session_state["thread_id"]] = "New Chat"

# Track which documents have been ingested this session
if "ingested_docs" not in st.session_state:
    st.session_state["ingested_docs"] = []


# ========================= Sidebar =========================

st.sidebar.title("My Conversations")

if st.sidebar.button("➕ New Chat", use_container_width=True):
    reset_chat()
    st.rerun()

st.sidebar.divider()

# --------- RAG Document Upload ---------
st.sidebar.markdown("### 📄 Document Upload")

# Show current index status
if rag_index_exists():
    st.sidebar.success("✅ Knowledge base is active", icon="🗂️")
else:
    st.sidebar.info("No documents uploaded yet.", icon="ℹ️")

uploaded_file = st.sidebar.file_uploader(
    "Upload a PDF or TXT file",
    type=["pdf", "txt"],
    help="The document will be chunked, embedded, and added to the knowledge base.",
)

if uploaded_file is not None:
    # Only process if this file hasn't been ingested already
    if uploaded_file.name not in st.session_state["ingested_docs"]:
        if st.sidebar.button("⚙️ Ingest Document", use_container_width=True):
            with st.sidebar.status(f"Ingesting **{uploaded_file.name}** …", expanded=True) as status:
                try:
                    # Save uploaded bytes to a temp file with the correct extension
                    suffix = os.path.splitext(uploaded_file.name)[1]
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                        tmp.write(uploaded_file.read())
                        tmp_path = tmp.name

                    st.write("📂 Loading document…")
                    chunk_count = ingest_rag_document(tmp_path)

                    # Clean up temp file
                    os.unlink(tmp_path)

                    st.write(f"✂️ Split into **{chunk_count}** chunks")
                    st.write("🔢 Embedded and saved to knowledge base")

                    # Track the ingested doc
                    st.session_state["ingested_docs"].append(uploaded_file.name)

                    status.update(
                        label=f"✅ **{uploaded_file.name}** ingested successfully!",
                        state="complete",
                        expanded=False,
                    )

                except Exception as e:
                    status.update(
                        label=f"❌ Ingestion failed: {e}",
                        state="error",
                        expanded=True,
                    )
    else:
        st.sidebar.success(f"✅ **{uploaded_file.name}** is already in the knowledge base.")

# Show list of ingested documents
if st.session_state["ingested_docs"]:
    st.sidebar.markdown("**Ingested documents:**")
    for doc_name in st.session_state["ingested_docs"]:
        st.sidebar.markdown(f"- 📎 {doc_name}")

st.sidebar.divider()

# --------- Conversation list ---------
st.sidebar.markdown("### 💬 Previous Chats")

for thread_id in st.session_state["chat_threads"][::-1]:
    thread_name = st.session_state["thread_names"].get(thread_id, "New Chat")

    if st.sidebar.button(thread_name, key=thread_id, use_container_width=True):
        st.session_state["thread_id"] = thread_id
        messages = load_conversation(thread_id)
        temp_messages = []

        for message in messages:
            if isinstance(message, HumanMessage):
                role = "user"
            elif isinstance(message, AIMessage):
                role = "assistant"
            else:
                continue

            temp_messages.append({"role": role, "content": message.content})

        st.session_state["message_history"] = temp_messages
        st.rerun()


# ========================= Main chat interface =========================

for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input("Ask me anything…")

if user_input:

    st.session_state["message_history"].append({
        "role": "user",
        "content": user_input,
    })

    thread_id = st.session_state["thread_id"]
    add_thread(thread_id)

    if st.session_state["thread_names"].get(thread_id) == "New Chat":
        st.session_state["thread_names"][thread_id] = generate_thread_name(user_input)

    with st.chat_message("user"):
        st.markdown(user_input)

    CONFIG = {"configurable": {"thread_id": st.session_state["thread_id"]}}

    with st.chat_message("assistant"):
        status_holder = {"box": None}

        def ai_only_stream():
            for message_chunk, metadata in workflow.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode="messages",
            ):
                # Show tool name as soon as the LLM decides to call one
                if isinstance(message_chunk, AIMessage):
                    tool_calls = getattr(message_chunk, "tool_calls", [])
                    if tool_calls:
                        for tc in tool_calls:
                            tool_name = tc.get("name", "tool")
                            if status_holder["box"] is None:
                                status_holder["box"] = st.status(
                                    f"🔧 Using `{tool_name}` …",
                                    expanded=True,
                                )
                            else:
                                status_holder["box"].update(
                                    label=f"🔧 Using `{tool_name}` …",
                                    state="running",
                                    expanded=True,
                                )

                # Mark tool complete once the result comes back
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

    st.session_state["message_history"].append({
        "role": "assistant",
        "content": ai_message,
    })
