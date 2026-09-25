# =============================================================================
# Imports
# =============================================================================
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated, Any
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_tavily import TavilySearch
from langchain_core.tools import tool
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever, ParentDocumentRetriever
from langchain_classic.storage import LocalFileStore, create_kv_docstore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from sentence_transformers import CrossEncoder
from dotenv import load_dotenv
import requests
import pickle
import math
import os
import sqlite3

load_dotenv()


# =============================================================================
# LLM & Embeddings
# =============================================================================
llm = ChatGroq(
    model="qwen/qwen3-32b",
    temperature=0.7,
    max_tokens=512,
)

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001"
)


# =============================================================================
# Hybrid RAG — storage paths
# =============================================================================
FAISS_CHILD_PATH  = "rag_store/faiss_children"   # FAISS index for small child chunks
PARENT_STORE_PATH = "rag_store/parent_store"      # LocalFileStore for full parent chunks
BM25_STORE_PATH   = "rag_store/bm25_corpus.pkl"   # Pickled list of all child Documents

# Splitters — match the reference notebooks
PARENT_SPLITTER = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
CHILD_SPLITTER  = RecursiveCharacterTextSplitter(chunk_size=400,  chunk_overlap=50)

# Cross-encoder re-ranker (runs locally, no API key needed)
# ms-marco-MiniLM-L-6-v2 is a fast, accurate model for passage re-ranking
_reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def _ensure_store_dirs():
    os.makedirs(FAISS_CHILD_PATH, exist_ok=True)
    os.makedirs(PARENT_STORE_PATH, exist_ok=True)
    os.makedirs("rag_store", exist_ok=True)


def rag_index_exists() -> bool:
    """Return True when both the FAISS child index and parent store are present."""
    return (
        os.path.exists(os.path.join(FAISS_CHILD_PATH, "index.faiss"))
        and os.path.exists(PARENT_STORE_PATH)
    )


def _load_bm25_corpus() -> list[Document]:
    """Load previously persisted child-chunk corpus for BM25."""
    if os.path.exists(BM25_STORE_PATH):
        with open(BM25_STORE_PATH, "rb") as f:
            return pickle.load(f)
    return []


def _save_bm25_corpus(corpus: list[Document]):
    with open(BM25_STORE_PATH, "wb") as f:
        pickle.dump(corpus, f)


# =============================================================================
# Ingestion
# =============================================================================
def ingest_rag_document(file_path: str) -> int:
    """
    Ingest a PDF or .txt file using parent-document retrieval strategy.

    - Parent chunks (1500 tokens) are stored in a LocalFileStore on disk.
    - Child chunks  (400 tokens)  are embedded and stored in FAISS.
    - All child chunks are also pickled for BM25 retrieval.

    New documents are MERGED into the existing stores — nothing is overwritten.
    Returns the number of child chunks added.
    """
    _ensure_store_dirs()

    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
    elif ext == ".txt":
        loader = TextLoader(file_path, encoding="utf-8")
    else:
        raise ValueError(f"Unsupported file type '{ext}'. Only .pdf and .txt are accepted.")

    raw_docs = loader.load()
    if not raw_docs:
        return 0

    # ── 1. Split into parent chunks ──────────────────────────────────────────
    parent_chunks = PARENT_SPLITTER.split_documents(raw_docs)

    # ── 2. Split parents into child chunks ───────────────────────────────────
    #  Tag each child with the index of its parent so we can look it up later
    child_chunks: list[Document] = []
    for p_idx, parent in enumerate(parent_chunks):
        children = CHILD_SPLITTER.split_documents([parent])
        for child in children:
            child.metadata["parent_id"] = f"doc_{p_idx}_{os.path.basename(file_path)}"
        child_chunks.extend(children)

    if not child_chunks:
        return 0

    # ── 3. Persist parent chunks to LocalFileStore ───────────────────────────
    fs         = LocalFileStore(PARENT_STORE_PATH)
    docstore   = create_kv_docstore(fs)
    parent_map = {}
    for p_idx, parent in enumerate(parent_chunks):
        key = f"doc_{p_idx}_{os.path.basename(file_path)}"
        parent_map[key] = parent
    docstore.mset(list(parent_map.items()))

    # ── 4. Embed child chunks into FAISS (merge if index already exists) ─────
    if os.path.exists(os.path.join(FAISS_CHILD_PATH, "index.faiss")):
        faiss_store = FAISS.load_local(
            folder_path=FAISS_CHILD_PATH,
            embeddings=embeddings,
            allow_dangerous_deserialization=True,
        )
        faiss_store.add_documents(child_chunks)
    else:
        faiss_store = FAISS.from_documents(child_chunks, embeddings)

    faiss_store.save_local(FAISS_CHILD_PATH)

    # ── 5. Append child chunks to BM25 corpus ────────────────────────────────
    existing_corpus = _load_bm25_corpus()
    updated_corpus  = existing_corpus + child_chunks
    _save_bm25_corpus(updated_corpus)

    return len(child_chunks)


# =============================================================================
# Hybrid Retrieval  (semantic + BM25 → RRF → parent lookup → re-rank)
# =============================================================================
def hybrid_retrieve(query: str, top_k: int = 6) -> list[Document]:
    """
    Full hybrid retrieval pipeline:

    1. FAISS semantic search on child chunks  (dense)
    2. BM25Plus keyword search on child chunks (sparse)
    3. Reciprocal Rank Fusion (RRF) to merge both ranked lists
    4. Deduplicate by parent_id and fetch the full parent chunk
    5. Cross-encoder re-ranking of the parent chunks
    6. Return top_k re-ranked parent chunks
    """
    # ── 1. Dense retriever ───────────────────────────────────────────────────
    faiss_store = FAISS.load_local(
        folder_path=FAISS_CHILD_PATH,
        embeddings=embeddings,
        allow_dangerous_deserialization=True,
    )
    dense_retriever = faiss_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": top_k * 2},   # cast a wide net before fusion
    )

    # ── 2. Sparse BM25 retriever ─────────────────────────────────────────────
    corpus = _load_bm25_corpus()
    bm25_retriever = BM25Retriever.from_documents(
        corpus,
        k=top_k * 2,
        bm25_variant="plus",   # BM25Plus: no zero scores for matched terms
    )

    # ── 3. EnsembleRetriever — uses RRF internally ───────────────────────────
    #  Weights: 0.7 semantic / 0.3 keyword  (tunable)
    ensemble = EnsembleRetriever(
        retrievers=[dense_retriever, bm25_retriever],
        weights=[0.7, 0.3],
    )
    fused_child_docs = ensemble.invoke(query)

    # ── 4. Parent lookup — swap each child for its full parent chunk ─────────
    fs       = LocalFileStore(PARENT_STORE_PATH)
    docstore = create_kv_docstore(fs)

    seen_parent_ids: set[str] = set()
    parent_docs: list[Document] = []

    for child in fused_child_docs:
        parent_id = child.metadata.get("parent_id")
        if parent_id and parent_id not in seen_parent_ids:
            result = docstore.mget([parent_id])
            parent = result[0] if result else None
            if parent is not None:
                seen_parent_ids.add(parent_id)
                parent_docs.append(parent)

        # Safety cap — no point re-ranking 50 docs
        if len(parent_docs) >= top_k * 3:
            break

    # Fall back to child docs if parent lookup yields nothing
    if not parent_docs:
        parent_docs = fused_child_docs

    # ── 5. Cross-encoder re-ranking ──────────────────────────────────────────
    if len(parent_docs) == 1:
        return parent_docs

    pairs  = [(query, doc.page_content) for doc in parent_docs]
    scores = _reranker.predict(pairs)

    ranked = sorted(zip(scores, parent_docs), key=lambda x: x[0], reverse=True)
    return [doc for _, doc in ranked[:top_k]]


# =============================================================================
# RAG Tool (used by the LangGraph agent)
# =============================================================================
@tool
def rag_tool(query: str) -> str:
    """
    Retrieve relevant information from uploaded PDF or text documents.

    Uses a hybrid pipeline: BM25 keyword search + semantic vector search,
    fused with Reciprocal Rank Fusion (RRF), parent-document expansion,
    and cross-encoder re-ranking for the most accurate results.

    Use this tool when the user asks factual or conceptual questions that
    can be answered from the uploaded documents.

    Args:
        query: The question or search phrase to look up in the documents.
    """
    if not rag_index_exists():
        return (
            "No documents have been uploaded yet. "
            "Please upload a PDF or text file using the 📄 Document Upload panel in the sidebar."
        )

    try:
        documents = hybrid_retrieve(query, top_k=4)
    except Exception as e:
        return f"Retrieval error: {e}"

    if not documents:
        return "No relevant information was found in the uploaded documents."

    formatted = []
    for idx, doc in enumerate(documents, start=1):
        source = doc.metadata.get("source", "Unknown source")
        page   = doc.metadata.get("page", "N/A")
        formatted.append(
            f"[{idx}] Source: {source} | Page: {page}\n"
            f"{doc.page_content}"
        )

    return "\n\n---\n\n".join(formatted)


# =============================================================================
# Other Tools
# =============================================================================
search_tool = TavilySearch(
    max_results=5,
    topic="general",
    search_depth="advanced",
    time_range="day",
)


@tool
def calculator(expression: str) -> str:
    """
    Evaluate a math expression.
    Examples: '2 + 2', 'math.sqrt(16)', '10 * 5 / 2'
    """
    try:
        allowed = {
            "math": math, "abs": abs, "round": round,
            "min": min, "max": max, "sum": sum,
        }
        return str(eval(expression, {"__builtins__": {}}, allowed))
    except Exception as e:
        return f"Calculation error: {e}"


@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch the latest stock price for a ticker symbol (e.g. 'AAPL', 'TSLA').
    """
    url = (
        f"https://www.alphavantage.co/query"
        f"?function=GLOBAL_QUOTE&symbol={symbol}&apikey=IGWD6PDN8J26RPOW"
    )
    return requests.get(url).json()


@tool
def get_current_weather(location: str) -> str:
    """
    Get real-time weather for a city or location.
    Examples: 'Dhaka', 'London, UK', 'New York, US'
    """
    api_key = os.getenv("OPENWEATHER_API_KEY")
    if not api_key:
        return "Weather API key is missing. Set OPENWEATHER_API_KEY in your .env."

    try:
        geo = requests.get(
            "https://api.openweathermap.org/geo/1.0/direct",
            params={"q": location, "limit": 1, "appid": api_key},
            timeout=10,
        )
        geo.raise_for_status()
        locs: list[dict[str, Any]] = geo.json()
        if not locs:
            return f"Could not find location: {location}"

        lat, lon = locs[0]["lat"], locs[0]["lon"]
        name     = locs[0].get("name", location)
        state    = locs[0].get("state", "")
        country  = locs[0].get("country", "")

        wx = requests.get(
            "https://api.openweathermap.org/data/2.5/weather",
            params={"lat": lat, "lon": lon, "appid": api_key, "units": "metric"},
            timeout=10,
        )
        wx.raise_for_status()
        d = wx.json()

        vis_m  = d.get("visibility")
        vis_km = round(vis_m / 1000, 1) if vis_m else "N/A"

        display = ", ".join(p for p in [name, state, country] if p)
        return (
            f"Current weather in {display}:\n"
            f"- Condition   : {d['weather'][0]['description'].title()}\n"
            f"- Temperature : {d['main']['temp']}°C\n"
            f"- Feels like  : {d['main']['feels_like']}°C\n"
            f"- Humidity    : {d['main']['humidity']}%\n"
            f"- Pressure    : {d['main']['pressure']} hPa\n"
            f"- Wind speed  : {d.get('wind', {}).get('speed', 'N/A')} m/s\n"
            f"- Visibility  : {vis_km} km"
        )

    except requests.Timeout:
        return "Weather service request timed out."
    except requests.HTTPError as e:
        code = e.response.status_code if e.response else "unknown"
        return ("Invalid API key." if code == 401 else f"Weather API HTTP error: {code}")
    except requests.RequestException as e:
        return f"Could not connect to weather service: {e}"
    except (KeyError, TypeError, ValueError) as e:
        return f"Unexpected weather API response: {e}"


# =============================================================================
# LangGraph agent setup
# =============================================================================
tools         = [search_tool, calculator, get_stock_price, get_current_weather, rag_tool]
llm_with_tools = llm.bind_tools(tools)


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def chat_node(state: ChatState):
    """LLM node — answers directly or delegates to a tool."""
    system_message = SystemMessage(
        content=(
            "You are a helpful Agentic Chatbot with access to several tools.\n\n"
            "Tool usage guidelines:\n"
            "- `rag_tool`: Use for ANY question about uploaded PDFs or text files. "
            "Always call it before answering document-related questions. "
            "It uses hybrid search (BM25 + semantic + re-ranking) internally.\n"
            "- `tavily_search`: Use for current events, recent news, or anything "
            "requiring live internet data.\n"
            "- `calculator`: Use for all mathematical calculations.\n"
            "- `get_stock_price`: Use when the user asks for a stock's current price.\n"
            "- `get_current_weather`: Use when the user asks about current weather.\n\n"
            "Answer general knowledge questions directly without using a tool. "
            "Do not fabricate document content — if no document is uploaded, say so. "
            "After receiving a tool result, give a clear, concise final answer."
        )
    )
    response = llm_with_tools.invoke([system_message, *state["messages"]])
    return {"messages": [response]}


tool_node  = ToolNode(tools)
conn       = sqlite3.connect(database="Chatbot.db", check_same_thread=False)
checkpointer = SqliteSaver(conn)

graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tools",     tool_node)
graph.add_edge(START,       "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools",     "chat_node")

workflow = graph.compile(checkpointer=checkpointer)


def get_all_thread() -> list[str]:
    all_threads: set[str] = set()
    for ckpt in checkpointer.list(None):
        all_threads.add(ckpt.config["configurable"]["thread_id"])
    return list(all_threads)


__all__ = [
    "workflow",
    "get_all_thread",
    "ingest_rag_document",
    "rag_index_exists",
]
