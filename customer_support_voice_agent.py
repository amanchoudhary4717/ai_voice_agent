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
from qdrant_client.http.models import Distance, VectorParams
from fastembed import TextEmbedding
from agents import Agent, Runner
from openai import AsyncOpenAI, RateLimitError


# ===============================
# CONFIG
# ===============================
COLLECTION_NAME = "docs_embeddings"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
MAX_CONTEXT_CHARS = 6000
QUERY_COOLDOWN_SECONDS = 15


# ===============================
# HELPERS
# ===============================
def chunk_text(text: str, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks


async def retry_openai(call, retries=3, base_delay=2):
    for attempt in range(retries):
        try:
            return await call()
        except RateLimitError:
            if attempt == retries - 1:
                return None
            await asyncio.sleep(base_delay * (attempt + 1))



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
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ===============================
# SIDEBAR
# ===============================
def sidebar():
    with st.sidebar:
        st.title("🔑 Configuration")

        qdrant_url = st.text_input("Qdrant URL", type="password")
        qdrant_api_key = st.text_input("Qdrant API Key", type="password")
        firecrawl_api_key = st.text_input("Firecrawl API Key", type="password")
        openai_api_key = st.text_input("OpenAI API Key", type="password")
        doc_url = st.text_input("Documentation URL")

        st.markdown("### 🎤 Voice")
        voices = ["alloy", "ash", "ballad", "coral", "echo", "fable", "onyx", "nova", "sage", "shimmer", "verse"]
        st.session_state.selected_voice = st.selectbox("Voice", voices, index=voices.index("coral"))

        if st.button("Initialize System", type="primary"):
            if not all([qdrant_url, qdrant_api_key, firecrawl_api_key, openai_api_key, doc_url]):
                st.error("Please fill all fields.")
                return

            with st.status("Initializing system...", expanded=True):
                client, embedding_model = setup_qdrant(qdrant_url, qdrant_api_key)
                pages = crawl_docs(firecrawl_api_key, doc_url)
                store_embeddings(client, embedding_model, pages)
                processor_agent = setup_agent(openai_api_key)

                st.session_state.client = client
                st.session_state.embedding_model = embedding_model
                st.session_state.processor_agent = processor_agent
                st.session_state.setup_complete = True

            st.success("System initialized successfully.")


# ===============================
# QDRANT
# ===============================
def setup_qdrant(url: str, api_key: str):
    client = QdrantClient(url=url, api_key=api_key)
    embedding_model = TextEmbedding()
    dim = len(list(embedding_model.embed(["test"]))[0])

    try:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )
    except Exception:
        pass

    return client, embedding_model


# ===============================
# FIRECRAWL
# ===============================
def crawl_docs(api_key: str, url: str) -> List[Dict]:
    firecrawl = FirecrawlApp(api_key=api_key)
    pages = []

    job = firecrawl.crawl(
        url=url,
        limit=5,
        scrape_options={"formats": ["markdown"]},
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
# EMBEDDINGS (FIXED)
# ===============================
def store_embeddings(client, embedding_model, pages):
    for page in pages:
        for chunk in chunk_text(page["content"]):
            vector = list(embedding_model.embed([chunk]))[0]

            client.upsert(
                collection_name=COLLECTION_NAME,
                points=[
                    models.PointStruct(
                        id=str(uuid.uuid4()),
                        vector=vector.tolist(),
                        payload={
                            "content": chunk,
                            "url": page["url"],
                            **page["metadata"],
                        },
                    )
                ],
            )


# ===============================
# AGENT
# ===============================
def setup_agent(openai_api_key: str):
    os.environ["OPENAI_API_KEY"] = openai_api_key

    return Agent(
        name="Documentation Assistant",
        instructions=(
            "Answer the user's question using ONLY the provided documentation context. "
            "Be concise, clear, and accurate. Cite sources when relevant."
        ),
        model="gpt-4o-mini",
    )


# ===============================
# QUERY PIPELINE
# ===============================
async def process_query(question: str):
    # cooldown
    if time.time() - st.session_state.last_query_time < QUERY_COOLDOWN_SECONDS:
        st.warning("Please wait a few seconds before asking again.")
        return None, None

    st.session_state.last_query_time = time.time()

    client = st.session_state.client
    embedding_model = st.session_state.embedding_model

    query_vector = list(embedding_model.embed([question]))[0]

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector.tolist(),
        limit=3,
        with_payload=True,
    ).points

    context_blocks = []
    total_len = 0

    for r in results:
        text = r.payload.get("content", "")
        if total_len + len(text) > MAX_CONTEXT_CHARS:
            break
        context_blocks.append(f"{text}\n(Source: {r.payload.get('url')})")
        total_len += len(text)

    context = "\n\n".join(context_blocks)

    prompt = f"""
Use the documentation below to answer the question.

Documentation:
{context}

Question:
{question}
"""

processor_result = await retry_openai(
    lambda: Runner.run(st.session_state.processor_agent, prompt)
)

if processor_result is None:
    return (
        "⚠️ The AI is temporarily busy due to rate limits. "
        "Please wait a minute and try again.",
        None
    )
if not answer:
    return answer, None

answer = processor_result.final_output


    async_openai = AsyncOpenAI()
    audio = await retry_openai(
        lambda: async_openai.audio.speech.create(
            model="gpt-4o-mini-tts",
            voice=st.session_state.selected_voice,
            input=answer,
            response_format="mp3",
        )
    )

    audio_path = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4()}.mp3")
    with open(audio_path, "wb") as f:
        f.write(audio.content)

    return answer, audio_path


# ===============================
# UI
# ===============================
def run_app():
    st.set_page_config(page_title="AI Documentation Voice Agent", layout="wide")
    init_session_state()
    sidebar()

    st.title("🎙️ AI Documentation Voice Agent")

    if not st.session_state.setup_complete:
        st.info("Configure the system using the sidebar.")
        return

    question = st.text_input("Ask a question about the documentation")

    if question:
        with st.spinner("Thinking..."):
            answer, audio_path = asyncio.run(process_query(question))

        if answer:
            st.markdown("### Answer")
            st.write(answer)

            st.markdown("### 🔊 Audio")
            st.audio(audio_path)


if __name__ == "__main__":
    run_app()
