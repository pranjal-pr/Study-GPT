import os
import time
from functools import lru_cache

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from observability import extract_usage_metrics

working_dir = os.path.dirname(os.path.abspath(__file__))
TOP_K = 3


@lru_cache(maxsize=1)
def get_embedding_model():
    # Lazy-load embeddings to avoid expensive startup/import side effects.
    return HuggingFaceEmbeddings()


def _get_relevant_docs(vector_db_path: str, query: str, k: int = TOP_K):
    """
    Retrieves top semantic matches from the user's vector DB.
    """
    vectordb = Chroma(persist_directory=vector_db_path, embedding_function=get_embedding_model())
    return vectordb.similarity_search(query, k=k)


def get_context_and_sources(vector_db_path: str, query: str, k: int = TOP_K):
    docs = _get_relevant_docs(vector_db_path, query, k=k)
    if not docs:
        return "", []

    context_parts = []
    sources = set()
    for doc in docs:
        context_parts.append(doc.page_content)
        sources.add(os.path.basename(doc.metadata.get("source", "Unknown")))

    return "\n\n---\n\n".join(context_parts), sorted(sources)


def process_documents_to_chroma_db(uploaded_files):
    timestamp = int(time.time() * 1000)
    new_db_folder = f"{working_dir}/vector_db_{timestamp}"

    all_documents = []

    for idx, uploaded_file in enumerate(uploaded_files):
        safe_name = os.path.basename(uploaded_file.name or "document.pdf")
        file_path = os.path.join(working_dir, f"tmp_{timestamp}_{idx}_{safe_name}")

        try:
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            loader = PyPDFLoader(file_path)
            documents = loader.load()

            for doc in documents:
                doc.metadata["source"] = safe_name

            all_documents.extend(documents)
        finally:
            if os.path.exists(file_path):
                os.remove(file_path)

    if not all_documents:
        raise ValueError("No valid PDF content found to index.")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    texts = text_splitter.split_documents(all_documents)

    # Create the vector DB in the NEW unique folder.
    Chroma.from_documents(
        documents=texts,
        embedding=get_embedding_model(),
        persist_directory=new_db_folder,
    )

    # Return the path so app.py knows where to look.
    return new_db_folder


def answer_question_with_agent(
    user_question: str,
    llm_instance,
    vector_db_path: str,
    chat_history_context: str = "",
):
    """
    Retrieves context and answers strictly from uploaded documents.
    """
    prompt, _sources = build_rag_prompt(
        user_question=user_question,
        vector_db_path=vector_db_path,
        chat_history_context=chat_history_context,
    )
    if prompt is None:
        return None

    response = llm_instance.invoke(prompt)
    return {
        "response": getattr(response, "content", str(response)),
        "usage": extract_usage_metrics(response),
    }


def build_rag_prompt(
    user_question: str,
    vector_db_path: str,
    chat_history_context: str = "",
):
    """
    Prepares the document-grounded prompt so non-streaming and streaming code paths stay aligned.
    """
    combined_context, sources = get_context_and_sources(vector_db_path, user_question)
    if not combined_context:
        return None, []

    source_list = ", ".join(sources)

    history_block = ""
    if chat_history_context:
        history_block = (
            "Conversation history (for continuity only; do not use as factual source):\n" f"{chat_history_context}\n\n"
        )

    prompt = (
        "You are a document QA assistant. Answer the user's question using only the CONTEXT below.\n"
        "Use history only to resolve references to prior turns.\n"
        "If the answer is not clearly present in the context, reply exactly with:\n"
        '"I couldn\'t find relevant information for that in your uploaded documents."\n\n'
        f"{history_block}"
        f"Question: {user_question}\n\n"
        f"CONTEXT:\n{combined_context}\n\n"
        f"Always end with: Sources: {source_list}"
    )
    return prompt, sources
