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
from agents import Agent, Runner
from openai import OpenAI, AsyncOpenAI, RateLimitError

# ===============================
# CONFIG
# ===============================
COLLECTION_NAME = "docs_embeddings"
CHUNK_SIZE = 512  # Updated to token-based sizing
CHUNK_OVERLAP = 100
MAX_CONTEXT_CHARS = 5000
QUERY_COOLDOWN_SECONDS = 15
EMBEDDING_MODEL = "text-embedding-3-small"  # OpenAI embedding model
EMBEDDING_DIM = 1536  # Dimension for text-embedding-3-small
RERANK_MODEL = 'cross-encoder/ms-marco-MiniLM-L-6-v2'  # Reranker model
CRAWL_LIMIT = 30  # Increased for better coverage

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

def chunk_text(text: str, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ".", " "]  # Semantic boundaries
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
# SIDEBAR (MOBILE SAFE)
# ===============================
def sidebar():
    with st.sidebar:
        st.header("⚙️ Setup")
        qdrant_url = st.text_input("Qdrant URL", type="password")
        qdrant_api_key = st.text_input("Qdrant API Key", type="password")
        firecrawl_api_key = st.text_input("Firecrawl API Key", type="password")
        openai_api_key = st.text_input("OpenAI API Key", type="password")
        doc_url = st.text_input("Documentation URL")
        st.markdown("### 🎤 Voice")
        voices = [
            "alloy", "ash", "ballad", "coral", "echo",
            "fable", "onyx", "nova", "sage", "shimmer", "verse"
        ]
        st.session_state.selected_voice = st.selectbox("Voice", voices, index=voices.index("coral"))
        if st.button("🚀 Initialize System", use_container_width=True):
            if not all([qdrant_url, qdrant_api_key, firecrawl_api_key, openai_api_key, doc_url]):
                st.error("Fill all fields")
                return
            with st.status("Initializing...", expanded=True):
                client = setup_qdrant(qdrant_url, qdrant_api_key)
                embedding_model = setup_embedding_model(openai_api_key)
                pages = crawl_docs(firecrawl_api_key, doc_url)
                store_embeddings(client, embedding_model, pages)
                processor_agent = setup_agent(openai_api_key)
                reranker = CrossEncoder(RERANK_MODEL)
                st.session_state.client = client
                st.session_state.embedding_model = embedding_model
                st.session_state.processor_agent = processor_agent
                st.session_state.reranker = reranker
                st.session_state.setup_complete = True
            st.success("System ready")
        if st.button("🔄 Reset App", use_container_width=True):
            st.session_state.clear()
            st.rerun()

# ===============================
# QDRANT
# ===============================
def setup_qdrant(url: str, api_key: str):
    client = QdrantClient(url=url, api_key=api_key)
    try:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )
    except Exception:
        pass
    return client

# ===============================
# EMBEDDINGS
# ===============================
def setup_embedding_model(openai_api_key: str):
    os.environ["OPENAI_API_KEY"] = openai_api_key
    openai_client = OpenAI()
    def embed_texts(texts: List[str]) -> List[List[float]]:
        response = openai_client.embeddings.create(input=texts, model=EMBEDDING_MODEL)
        return [item.embedding for item in response.data]
    return embed_texts

# ===============================
# FIRECRAWL
# ===============================
def crawl_docs(api_key: str, url: str) -> List[Dict]:
    firecrawl = FirecrawlApp(api_key=api_key)
    pages = []
    job = firecrawl.crawl(
        url=url,
        limit=CRAWL_LIMIT,
        scrape_options={"formats": ["markdown"], "exclude_paths": ["/blog/*", "/forum/*"]},
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
    for page in pages:
        chunks = chunk_text(page["content"])
        vectors = embedding_model(chunks)
        points = []
        for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
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
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=points,
        )

# ===============================
# AGENT
# ===============================
def setup_agent(openai_api_key: str):
    os.environ["OPENAI_API_KEY"] = openai_api_key
    return Agent(
        name="Documentation Assistant",
        instructions=(
            "Answer using ONLY the provided documentation context. "
            "Be concise and accurate. Cite sources."
        ),
        model="gpt-4o-mini",
    )

# ===============================
# VOICE → TEXT
# ===============================
async def transcribe_audio(audio_bytes):
    async_openai = AsyncOpenAI()
    transcript = await retry_openai(
        lambda: async_openai.audio.transcriptions.create(
            file=audio_bytes,
            model="gpt-4o-mini-transcribe"  # Updated to correct model
        )
    )
    return transcript.text if transcript else None

# ===============================
# QUERY PIPELINE
# ===============================
async def process_query(question: str):
    if time.time() - st.session_state.last_query_time < QUERY_COOLDOWN_SECONDS:
        return "⏳ Please wait a few seconds.", None, []
    st.session_state.last_query_time = time.time()
    client = st.session_state.client
    embedding_model = st.session_state.embedding_model
    processor_agent = st.session_state.processor_agent
    reranker = st.session_state.reranker

    # Query Rewriting for better retrieval
    rewritten_query = await retry_openai(
        lambda: Runner.run(processor_agent, f"Rewrite this query for better retrieval: {question}")
    )
    if rewritten_query is None:
        rewritten_query = question  # Fallback

    query_vector = embedding_model([rewritten_query])[0]

    # Hybrid Search: Vector + Keyword Filter
    keywords = [kw for kw in rewritten_query.split() if len(kw) > 3]
    query_filter = Filter(
        must=[FieldCondition(key="content", match=MatchValue(value=kw)) for kw in keywords]
    ) if keywords else None

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        query_filter=query_filter,
        limit=10,  # Retrieve more for reranking
        with_payload=True,
    ).points

    if not results:
        return "No relevant documentation found.", None, []

    # Reranking
    pairs = [(rewritten_query, r.payload.get("content", "")) for r in results]
    scores = reranker.predict(pairs)
    sorted_indices = scores.argsort(descending=True)[:3]  # Top-3 after rerank
    sorted_results = [results[i] for i in sorted_indices]

    context_blocks, sources = [], set()
    total_len = 0
    for r in sorted_results:
        text = r.payload.get("content", "")
        url = r.payload.get("url", "unknown")
        if total_len + len(text) > MAX_CONTEXT_CHARS:
            break
        context_blocks.append(text)
        sources.add(url)
        total_len += len(text)

    prompt = f"""
Answer using the documentation below.
Documentation:
{chr(10).join(context_blocks)}
Question:
{question}
"""
    result = await retry_openai(
        lambda: Runner.run(processor_agent, prompt)
    )
    if result is None:
        return "⚠️ AI busy. Try again later.", None, list(sources)
    answer = result.final_output
    async_openai = AsyncOpenAI()
    audio = await retry_openai(
        lambda: async_openai.audio.speech.create(
            model="gpt-4o-mini-tts",
            voice=st.session_state.selected_voice,
            input=answer,
            response_format="mp3",
        )
    )
    audio_path = None
    if audio:
        audio_path = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4()}.mp3")
        with open(audio_path, "wb") as f:
            f.write(audio.content)
    return answer, audio_path, list(sources)

# ===============================
# UI (MOBILE FIRST)
# ===============================
def run_app():
    st.set_page_config(
        page_title="AI Voice Support Agent",
        layout="centered"
    )
    init_session_state()
    sidebar()
    st.title("🎙️ AI Documentation Voice Agent")
    st.caption("Ask documentation questions by text or voice")
    if not st.session_state.setup_complete:
        st.info("👈 Open the sidebar and initialize the system")
        return
    # Voice input (mobile friendly)
    audio_input = st.audio_input("🎤 Ask by voice")
    voice_question = None
    if audio_input:
        with st.spinner("Transcribing voice..."):
            voice_question = asyncio.run(transcribe_audio(audio_input))
        if voice_question:
            st.success(f"You said: {voice_question}")
    question = st.text_input(
        "💬 Ask by text",
        placeholder="How do I deploy a Streamlit app?"
    )
    final_question = voice_question or question
    if final_question:
        with st.spinner("Thinking..."):
            answer, audio_path, sources = asyncio.run(
                process_query(final_question)
            )
        st.subheader("Answer")
        st.write(answer)
        if audio_path:
            st.audio(audio_path)
        if sources:
            st.subheader("Sources")
            for src in sources:
                st.write(src)

if __name__ == "__main__":
    run_app()
