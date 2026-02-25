import os
import time
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

working_dir = os.path.dirname(os.path.abspath(__file__))
embedding = HuggingFaceEmbeddings()
TOP_K = 3


def _get_relevant_docs(vector_db_path: str, query: str, k: int = TOP_K):
    """
    Retrieves top semantic matches from the user's vector DB.
    """
    vectordb = Chroma(persist_directory=vector_db_path, embedding_function=embedding)
    return vectordb.similarity_search(query, k=k)

def process_documents_to_chroma_db(uploaded_files):
    timestamp = int(time.time())
    new_db_folder = f"{working_dir}/vector_db_{timestamp}"

    all_documents = []
    
    for uploaded_file in uploaded_files:
        file_path = os.path.join(working_dir, uploaded_file.name)
        
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
            
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        
        for doc in documents:
            doc.metadata['source'] = uploaded_file.name
            
        all_documents.extend(documents)
        
        if os.path.exists(file_path):
            os.remove(file_path)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    texts = text_splitter.split_documents(all_documents)
    
    # Create the vector DB in the NEW unique folder
    vectordb = Chroma.from_documents(
        documents=texts, 
        embedding=embedding, 
        persist_directory=new_db_folder
    )
    
    # Return the path so app.py knows where to look
    return new_db_folder

def answer_question_with_agent(user_question:str,llm_instance,vector_db_path:str):
    """
    Retrieves context and answers strictly from uploaded documents.
    """
    docs = _get_relevant_docs(vector_db_path, user_question)
    if not docs:
        return None

    context_parts = []
    sources = set()
    for doc in docs:
        context_parts.append(doc.page_content)
        sources.add(os.path.basename(doc.metadata.get("source", "Unknown")))

    combined_context = "\n\n---\n\n".join(context_parts)
    source_list = ", ".join(sorted(sources))

    prompt = (
        "You are a document QA assistant. Answer the user's question using only the CONTEXT below.\n"
        "If the answer is not clearly present in the context, reply exactly with:\n"
        "\"I couldn't find relevant information for that in your uploaded documents.\"\n\n"
        f"Question: {user_question}\n\n"
        f"CONTEXT:\n{combined_context}\n\n"
        f"Always end with: Sources: {source_list}"
    )

    response = llm_instance.invoke(prompt)
    return getattr(response, "content", str(response))
