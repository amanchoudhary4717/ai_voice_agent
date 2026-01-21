from typing import List, Dict
import os
import uuid
import tempfile
import asyncio
import time
from datetime import datetime
import streamlit as st
from firecrawl import FirecrawlApp
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams, Filter, FieldCondition, MatchValue
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import CrossEncoder
from openai import OpenAI, AsyncOpenAI, RateLimitError
from openai_agents import Agent, Runner  # Correct import for OpenAI Agents SDK

# ===============================
# CONFIG
# ===============================
COLLECTION_NAME = "docs_embeddings"
CHUNK_SIZE = 512  # Token-based sizing (recommended)
CHUNK_OVERLAP = 100
MAX_CONTEXT_CHARS = 5000
QUERY_COOLDOWN_SECONDS = 15
EMBEDDING_MODEL = "text-embedding-3-small"  # 1536 dim, good balance of quality & speed
EMBEDDING_DIM = 1536
RERANK_MODEL = 'cross-encoder/ms-marco-MiniLM-L-6-v2'
CRAWL_LIMIT = 30  # Increased for better doc coverage
MAX_RETRIEVE = 10  # Candidates for reranking

# ===============================
# HELPERS
# ===============================
async def retry_openai(call, retries=3, base_delay=2):
    for attempt in range(retries):
        try:
            return await call()
        except RateLimitError:
            if attempt == retries - 1:
                return None
            await asyncio.sleep(base_delay * (attempt + 1))

def chunk_text(text: str):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", "!", "?", ","]  # More semantic splits
    )
    return splitter.split_text(text)

# ===============================
# SESSION STATE
# ===============================
def init_session_state():
    defaults = {
        "setup_complete": False,
        "client": None,
        "embedding_model": None,
        "processor_agent": None,
        "selected_voice": "coral",
        "last_query_time": 0.0,
        "reranker": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

# ===============================
# SIDEBAR
# ===============================
def sidebar():
    with st.sidebar:
        st.header("⚙️ Setup")
        qdrant_url = st.text_input("Qdrant URL", type="password")
        qdrant_api_key = st.text_input("Qdrant API Key", type="password")
        firecrawl_api_key = st.text_input("Firecrawl API Key", type="password")
        openai_api_key = st.text_input("OpenAI API Key", type="password")
        doc_url = st.text_input("Documentation URL", placeholder="https://docs.streamlit.io/")
        
        st.markdown("### 🎤 Voice Settings")
        voices = ["alloy", "ash", "ballad", "coral", "echo", "fable", "onyx", "nova", "sage", "shimmer", "verse"]
        st.session_state.selected_voice = st.selectbox("Voice", voices, index=voices.index("coral"))

        col1, col2 = st.columns(2)
        with col1:
            if st.button("🚀 Initialize", use_container_width=True):
                if not all([qdrant_url, qdrant_api_key, firecrawl_api_key, openai_api_key, doc_url]):
                    st.error("Please fill all required fields")
                    return
                with st.status("Initializing system...", expanded=True) as status:
                    status.update(label="Setting up Qdrant...", state="running")
                    client = setup_qdrant(qdrant_url, qdrant_api_key)
                    
                    status.update(label="Loading embedding model...", state="running")
                    embedding_model = setup_embedding_model(openai_api_key)
                    
                    status.update(label="Crawling documentation...", state="running")
                    pages = crawl_docs(firecrawl_api_key, doc_url)
                    
                    status.update(label="Storing embeddings...", state="running")
                    store_embeddings(client, embedding_model, pages)
                    
                    status.update(label="Creating AI agent...", state="running")
                    processor_agent = setup_agent(openai_api_key)
                    reranker = CrossEncoder(RERANK_MODEL)
                    
                    st.session_state.client = client
                    st.session_state.embedding_model = embedding_model
                    st.session_state.processor_agent = processor_agent
                    st.session_state.reranker = reranker
                    st.session_state.setup_complete = True
                    
                    status.update(label="System ready!", state="complete")
                st.success("✅ System initialized successfully!")
                
        with col2:
            if st.button("🔄 Reset", use_container_width=True):
                st.session_state.clear()
                st.rerun()

# ===============================
# QDRANT SETUP
# ===============================
def setup_qdrant(url: str, api_key: str):
    client = QdrantClient(url=url, api_key=api_key)
    try:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )
    except Exception:
        # Collection likely already exists
        pass
    return client

# ===============================
# EMBEDDINGS (OpenAI)
# ===============================
def setup_embedding_model(openai_api_key: str):
    openai_client = OpenAI(api_key=openai_api_key)
    def embed(texts: List[str]) -> List[List[float]]:
        response = openai_client.embeddings.create(input=texts, model=EMBEDDING_MODEL)
        return [item.embedding for item in response.data]
    return embed

# ===============================
# FIRECRAWL CRAWLING
# ===============================
def crawl_docs(api_key: str, url: str) -> List[Dict]:
    firecrawl = FirecrawlApp(api_key=api_key)
    pages = []
    job = firecrawl.crawl(
        url=url,
        limit=CRAWL_LIMIT,
        scrape_options={
            "formats": ["markdown"],
            "exclude_paths": ["/blog/*", "/forum/*", "/changelog/*", "/community/*"],
            "only_main_content": True,
        },
    )
    for page in job.data:
        if not page.markdown or len(page.markdown) < 200:
            continue
        pages.append({
            "content": page.markdown,
            "url": getattr(page.metadata, "sourceURL", url),
            "metadata": {
                "title": getattr(page.metadata, "title", ""),
                "crawl_date": datetime.now().isoformat(),
            },
        })
    return pages

# ===============================
# STORE EMBEDDINGS
# ===============================
def store_embeddings(client, embedding_model, pages):
    progress = st.progress(0)
    total = len(pages)
    for i, page in enumerate(pages):
        chunks = chunk_text(page["content"])
        vectors = embedding_model(chunks)
        points = []
        for chunk, vector in zip(chunks, vectors):
            points.append(
                models.PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vector,
                    payload={
                        "content": chunk,
                        "url": page["url"],
                        **page["metadata"],
                    },
                )
            )
        client.upsert(collection_name=COLLECTION_NAME, points=points)
        progress.progress((i + 1) / total)
    st.success(f"Stored {sum(len(chunk_text(p['content'])) for p in pages)} chunks")

# ===============================
# AGENT SETUP (OpenAI Agents SDK)
# ===============================
def setup_agent(openai_api_key: str):
    os.environ["OPENAI_API_KEY"] = openai_api_key
    return Agent(
        name="Documentation Assistant",
        instructions=(
            "You are an expert documentation assistant. "
            "Answer questions using ONLY the provided context. "
            "Be concise, accurate, and cite sources with URLs when possible."
        ),
        model="gpt-4o-mini",
    )

# ===============================
# TRANSCRIPTION
# ===============================
async def transcribe_audio(audio_bytes):
    async_openai = AsyncOpenAI()
    transcript = await retry_openai(
        lambda: async_openai.audio.transcriptions.create(
            file=audio_bytes,
            model="gpt-4o-mini-transcribe"
        )
    )
    return transcript.text if transcript else None

# ===============================
# QUERY PROCESSING (Optimized RAG)
# ===============================
async def process_query(question: str):
    if time.time() - st.session_state.last_query_time < QUERY_COOLDOWN_SECONDS:
        return "⏳ Please wait a few seconds before asking again.", None, []
    
    st.session_state.last_query_time = time.time()
    client = st.session_state.client
    embed_fn = st.session_state.embedding_model
    agent = st.session_state.processor_agent
    reranker = st.session_state.reranker

    # 1. Query Rewriting
    with st.spinner("Optimizing query..."):
        rewritten = await retry_openai(
            lambda: Runner.run(agent, f"Rewrite this query for better semantic search: {question}")
        ) or question

    query_vector = embed_fn([rewritten])[0]

    # 2. Hybrid Retrieval
    keywords = [kw for kw in rewritten.lower().split() if len(kw) > 3]
    query_filter = None
    if keywords:
        query_filter = Filter(
            must=[FieldCondition(key="content", match=MatchValue(value=kw)) for kw in keywords]
        )

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        query_filter=query_filter,
        limit=MAX_RETRIEVE,
        with_payload=True,
    ).points

    if not results:
        return "No relevant documentation found.", None, []

    # 3. Reranking
    pairs = [(rewritten, r.payload.get("content", "")) for r in results]
    scores = reranker.predict(pairs)
    top_indices = scores.argsort(descending=True)[:3]
    top_results = [results[i] for i in top_indices]

    # 4. Build Context
    context_blocks, sources = [], set()
    total_len = 0
    for r in top_results:
        text = r.payload.get("content", "")
        url = r.payload.get("url", "Unknown")
        if total_len + len(text) > MAX_CONTEXT_CHARS:
            break
        context_blocks.append(text)
        sources.add(url)
        total_len += len(text)

    prompt = f"""
Answer using **ONLY** the provided documentation context.
Documentation:
{chr(10).join(context_blocks)}

Question: {question}

Provide a concise, accurate answer. Cite sources with URLs when relevant.
"""

    # 5. Generate Answer
    with st.spinner("Generating answer..."):
        result = await retry_openai(
            lambda: Runner.run(agent, prompt)
        )
        if result is None:
            return "⚠️ AI is busy. Please try again later.", None, list(sources)
        answer = result.final_output

    # 6. Generate Audio
    audio_path = None
    with st.spinner("Generating voice response..."):
        async_openai = AsyncOpenAI()
        audio_response = await retry_openai(
            lambda: async_openai.audio.speech.create(
                model="gpt-4o-mini-tts",
                voice=st.session_state.selected_voice,
                input=answer,
                response_format="mp3",
            )
        )
        if audio_response:
            audio_path = os.path.join(tempfile.gettempdir(), f"response_{uuid.uuid4()}.mp3")
            with open(audio_path, "wb") as f:
                f.write(audio_response.content)

    return answer, audio_path, list(sources)

# ===============================
# MAIN APP
# ===============================
def run_app():
    st.set_page_config(
        page_title="AI Voice Documentation Agent",
        page_icon="🎙️",
        layout="centered",
        initial_sidebar_state="expanded"
    )

    init_session_state()
    sidebar()

    st.title("🎙️ AI Documentation Voice Agent")
    st.caption("Ask questions about any documentation via text or voice")

    if not st.session_state.setup_complete:
        st.info("👈 Open the sidebar to initialize the system with your API keys and documentation URL")
        return

    # Voice Input
    audio_input = st.audio_input("🎤 Ask by voice (tap to record)")
    voice_question = None
    if audio_input:
        with st.spinner("Transcribing your voice..."):
            voice_question = asyncio.run(transcribe_audio(audio_input))
        if voice_question:
            st.success(f"**You said:** {voice_question}")

    # Text Input
    question = st.text_input(
        "💬 Or type your question",
        placeholder="How do I deploy a Streamlit app?",
        key="text_query"
    )

    final_question = voice_question or question

    if final_question:
        answer, audio_path, sources = asyncio.run(process_query(final_question))

        st.subheader("Answer")
        st.markdown(answer)

        if audio_path:
            st.audio(audio_path, format="audio/mp3")

        if sources:
            st.subheader("Sources")
            for src in sources:
                st.markdown(f"- {src}")

if __name__ == "__main__":
    run_app()
