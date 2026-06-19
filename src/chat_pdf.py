"""
chat_pdf.py - Phase 2: RAG Chat Interface

Loads the local Chroma vector store, retrieves relevant chunks for each
user query, and generates an answer via Ollama using only the retrieved context.
"""

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.combine_documents.stuff import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

CHROMA_PERSIST_DIR = "../data/chroma_db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
OLLAMA_MODEL = "llama3.3:latest"
RETRIEVAL_K = 3

SYSTEM_PROMPT = (
    "Answer the question based ONLY on the following context. "
    "If the context does not contain enough information to answer the question, "
    "say 'I don't know based on the provided document.' "
    "Do not make up or infer information beyond what is explicitly stated.\n\n"
    "Context:\n{context}"
)


def main():
    print(f"Loading vector store from '{CHROMA_PERSIST_DIR}' ...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = Chroma(
        persist_directory=CHROMA_PERSIST_DIR,
        embedding_function=embeddings,
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": RETRIEVAL_K})
    print(f"Retriever ready (top-{RETRIEVAL_K} chunks).")

    print(f"Initializing Ollama with model '{OLLAMA_MODEL}' ...")
    llm = Ollama(model=OLLAMA_MODEL, temperature=0)

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{input}"),
    ])

    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, combine_docs_chain)

    print("\nRAG Chat ready. Type 'exit' or 'quit' to stop.\n")

    while True:
        question = input("You: ").strip()
        if question.lower() in ("exit", "quit"):
            print("Goodbye!")
            break
        if not question:
            continue

        try:
            result = rag_chain.invoke({"input": question})

            print(f"\nAnswer: {result['answer']}\n")

            print("Sources:")
            for i, doc in enumerate(result.get("context", []), 1):
                snippet = doc.page_content[:200].replace("\n", " ")
                print(f"  [{i}] {snippet}...")
            print()

        except Exception as e:
            print(f"ERROR: {e}\n")


if __name__ == "__main__":
    main()
