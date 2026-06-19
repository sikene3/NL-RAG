"""
app_rag.py - Phase 5: Multi-Mode RAG + Data Analysis Chat

A ChatGPT-like web UI supporting two modes:
  - Document Q&A (RAG): PDF ingestion → Chroma vector store → retrieval chain
  - Data Analysis (Agent): CSV/Excel → Pandas DataFrame → LLM agent for calculations
"""

import os
import tempfile
import pandas as pd
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.combine_documents.stuff import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
try:
    from langchain_experimental.agents import create_pandas_dataframe_agent
    HAS_PANDAS_AGENT = True
except ImportError:
    HAS_PANDAS_AGENT = False

CHROMA_PERSIST_DIR = "../data/chroma_db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
OLLAMA_MODEL = "llama3.3:latest"
RETRIEVAL_K = 3
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

SYSTEM_PROMPT = (
    "Answer the question based ONLY on the following context. "
    "If the context does not contain enough information to answer the question, "
    "say 'I don't know based on the provided document.' "
    "Do not make up or infer information beyond what is explicitly stated.\n\n"
    "Context:\n{context}"
)

# ── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(page_title="Local AI Document Chat", layout="wide")
st.title("Local AI Document Chat")
st.caption(
    "Upload PDFs for document Q&A, or CSV/Excel files for data analysis. "
    "All processing runs locally."
)


# ── Cached resources (loaded once) ───────────────────────────────────────────

@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


@st.cache_resource
def load_vectorstore():
    embeddings = load_embeddings()
    return Chroma(
        persist_directory=CHROMA_PERSIST_DIR,
        embedding_function=embeddings,
    )


@st.cache_resource
def load_llm():
    return Ollama(model=OLLAMA_MODEL, temperature=0)


def build_rag_chain():
    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": RETRIEVAL_K})
    llm = load_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{input}"),
    ])
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, combine_docs_chain)


# ── Session state ────────────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []

if "ingested_files" not in st.session_state:
    st.session_state.ingested_files = set()

if "active_mode" not in st.session_state:
    st.session_state.active_mode = "Document Q&A (RAG)"

if "dataframe" not in st.session_state:
    st.session_state.dataframe = None

if "df_filename" not in st.session_state:
    st.session_state.df_filename = None


# ── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Mode")
    mode_options = ["Document Q&A (RAG)"]
    if HAS_PANDAS_AGENT:
        mode_options.append("Data Analysis (Agent)")
    else:
        st.caption("Install `langchain-experimental` to enable Data Analysis mode.")
    st.session_state.active_mode = st.radio(
        "Select mode:",
        mode_options,
        index=0 if st.session_state.active_mode.startswith("Document") else 1,
    )

    st.divider()
    st.header("Upload File")

    accepted_types = ["pdf", "csv", "xlsx"]
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=accepted_types,
        help="PDF → added to knowledge base. CSV/Excel → loaded for data analysis.",
    )

    if uploaded_file is not None:
        fname = uploaded_file.name
        ext = fname.rsplit(".", 1)[-1].lower()

        # ── PDF ingestion ────────────────────────────────────────────────────
        if ext == "pdf" and fname not in st.session_state.ingested_files:
            with st.spinner(f"Ingesting '{fname}' ..."):
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(uploaded_file.getvalue())
                        tmp_path = tmp.name

                    loader = PyPDFLoader(tmp_path)
                    documents = loader.load()
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=CHUNK_SIZE,
                        chunk_overlap=CHUNK_OVERLAP,
                    )
                    chunks = text_splitter.split_documents(documents)

                    # Filter out chunks with empty or whitespace-only text
                    valid_chunks = [
                        c for c in chunks if c.page_content.strip()
                    ]
                    skipped = len(chunks) - len(valid_chunks)

                    if not valid_chunks:
                        os.unlink(tmp_path)
                        st.warning(
                            f"**{fname}** — No extractable text found. "
                            "The PDF may be scanned/image-based."
                        )
                    else:
                        vectorstore = load_vectorstore()
                        vectorstore.add_documents(valid_chunks)

                        os.unlink(tmp_path)
                        st.session_state.ingested_files.add(fname)

                        msg = f"Ingested {len(valid_chunks)} chunks from '{fname}'!"
                        if skipped:
                            msg += f" ({skipped} empty chunks skipped)"
                        st.toast(msg, icon="✅")
                        st.success(
                            f"**{fname}** — {len(documents)} page(s) → "
                            f"{len(valid_chunks)} chunk(s) added to the knowledge base."
                            + (f" ({skipped} empty chunks skipped)" if skipped else "")
                        )
                        st.session_state.active_mode = "Document Q&A (RAG)"

                except Exception as e:
                    st.error(f"Failed to ingest PDF: {e}")

        # ── CSV / Excel loading ──────────────────────────────────────────────
        elif ext in ("csv", "xlsx"):
            with st.spinner(f"Loading '{fname}' ..."):
                try:
                    if ext == "csv":
                        df = pd.read_csv(uploaded_file)
                    else:
                        df = pd.read_excel(uploaded_file)

                    st.session_state.dataframe = df
                    st.session_state.df_filename = fname
                    st.session_state.active_mode = "Data Analysis (Agent)"

                    st.toast(f"Loaded '{fname}' for analysis!", icon="📊")
                    st.success(
                        f"**{fname}** — {df.shape[0]} rows × {df.shape[1]} columns ready for analysis."
                    )

                except Exception as e:
                    st.error(f"Failed to load file: {e}")

    # ── Status indicators ────────────────────────────────────────────────────
    st.divider()
    if st.session_state.ingested_files:
        st.caption("**Ingested PDFs:**")
        for fname in sorted(st.session_state.ingested_files):
            st.caption(f"- {fname}")

    if st.session_state.dataframe is not None:
        st.caption(f"**Active DataFrame:** {st.session_state.df_filename}")
        with st.expander("Preview (first 5 rows)"):
            st.dataframe(st.session_state.dataframe.head(5), use_container_width=True)


# ── Render chat history ─────────────────────────────────────────────────────

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("View source chunks"):
                for i, src in enumerate(msg["sources"], 1):
                    st.caption(f"**Chunk {i}**")
                    st.text(src)


# ── Chat input ───────────────────────────────────────────────────────────────

mode_label = st.session_state.active_mode
placeholder = (
    "Ask about your data (e.g. 'What is the average of column X?')..."
    if "Data" in mode_label
    else "Ask about your document..."
)

if question := st.chat_input(placeholder):
    # Display user message
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Generate answer
    with st.chat_message("assistant"):
        spinner_text = (
            "Analyzing data..."
            if "Data" in mode_label
            else "Searching documents & generating answer..."
        )

        with st.spinner(spinner_text):
            try:
                # ── Data Analysis mode ───────────────────────────────────────
                if "Data" in mode_label and st.session_state.dataframe is not None:
                    if not HAS_PANDAS_AGENT:
                        answer = (
                            "The `langchain-experimental` package is not installed. "
                            "Please install it with:\n\n"
                            "```bash\npip install langchain-experimental\n```"
                        )
                        st.warning(answer)
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": answer,
                        })
                    else:
                        llm = load_llm()
                        agent = create_pandas_dataframe_agent(
                            llm,
                            st.session_state.dataframe,
                            allow_dangerous_code=True,
                            verbose=False,
                            include_df_in_prompt=True,
                            number_of_head_rows=5,
                        )
                        result = agent.invoke({"input": question})
                        answer = result["output"]
                        st.markdown(answer)
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": answer,
                        })

                # ── Document Q&A mode ────────────────────────────────────────
                else:
                    rag_chain = build_rag_chain()
                    result = rag_chain.invoke({"input": question})

                    answer = result["answer"]
                    sources = [
                        doc.page_content[:300].replace("\n", " ")
                        for doc in result.get("context", [])
                    ]

                    st.markdown(answer)

                    if sources:
                        with st.expander("View source chunks"):
                            for i, src in enumerate(sources, 1):
                                st.caption(f"**Chunk {i}**")
                                st.text(src)

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                    })

            except Exception as e:
                error_msg = f"Error: {e}"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg,
                })
