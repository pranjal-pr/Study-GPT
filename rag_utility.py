import os
import time
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_classic.chains import RetrievalQA

working_dir = os.path.dirname(os.path.abspath(__file__))
embedding = HuggingFaceEmbeddings()

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

def answer_question_with_rag(user_question, llm_instance, vector_db_path):
    # Load the specific DB provided by the app
    vectordb = Chroma(persist_directory=vector_db_path, embedding_function=embedding)
    retriever = vectordb.as_retriever()

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm_instance,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    
    response = qa_chain.invoke({"query": user_question})
    
    answer = response["result"]
    source_documents = response["source_documents"]
    
    sources = set()
    for doc in source_documents:
        full_source = doc.metadata.get('source', 'Unknown')
        file_name = os.path.basename(full_source)
        sources.add(file_name)
    
    return answer, list(sources) 
