from fastapi import FastAPI,UploadFile,File,HTTPException
from pydantic import BaseModel
from typing import List,Optional 
import os 
import shutil 

#Import the functions from rag_utility
from rag_utility import process_documents_to_chroma_db,answer_question_with_agent

#LangChain LLM Wrappers
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

app=FastAPI(
    title="ShinzoGPT Agentic RAG API",
    description="Backend API for the Agentic Retrieval-Augmented Generation chatbot.",
    version="1.0"
)

#DATA MODELS 
#Pydantic models validate the incoming JSON payloads 
class ChatRequest(BaseModel):
    query:str
    provider:str
    model:str
    api_key:str
    vector_db_path:Optional[str]=None
    is_nvidia_key:bool=False


def should_use_rag(query: str) -> bool:
    """
    Route clearly general-knowledge prompts to normal LLM chat even when a vector DB exists.
    """
    q = query.strip().lower()
    if not q:
        return False

    doc_markers = [
        "document", "documents", "pdf", "file", "uploaded", "upload",
        "admit card", "resume", "invoice", "this card", "this file", "from the doc",
        "from this", "according to the document", "in the document"
    ]
    if any(marker in q for marker in doc_markers):
        return True

    general_prefixes = (
        "what is", "what are", "who is", "who are", "when is", "when was",
        "where is", "why is", "how does", "how do", "explain", "define",
        "tell me about", "give me an overview", "example of"
    )
    if q.startswith(general_prefixes):
        return False

    return True

#Utilities
def get_llm(provider:str,model:str,api_key:str,is_nvidia_key:bool):
    """Instantiates the correct LLM based on the user's request payload."""
    if provider=="Groq":
        return ChatGroq(model=model,api_key=api_key)
    elif provider=="Gemini":
        return ChatGoogleGenerativeAI(model=model,google_api_key=api_key)
    elif provider=="Moonshot Kimi":
        base="https://integrate.api.nvidia.com/v1" if is_nvidia_key else "https://api.moonshot.cn/v1"
        return ChatOpenAI(model=model,api_key=api_key,base_url=base)
    raise ValueError("Invalid LLM Provider")
    

#Endpoints
@app.post("/upload")
async def upload_documents(files: List[UploadFile]=File(...)):
    """Handles PDF uploads, saves them temporarily, and triggers the rag ingestion."""
    try:
        class StreamlitMockFile:
            """A small helper class to match the 'uploaded_file_name' structure from Streamlit."""
            def __init__(self,name,buffer):
                self.name=name
                self.buffer=buffer
            def getbuffer(self):
                return self.buffer
            
        processed_files=[]               
        for file in files:
            content=await file.read()
            processed_files.append(StreamlitMockFile(file.filename,content))

        db_path=process_documents_to_chroma_db(processed_files)

        return {
            "message": "Documents successfully ingested into the Vector database.",
            "vector_db_path":db_path
        }
    except Exception as e:
        raise HTTPException(status_code=500,detail=str(e))
    
@app.post("/chat")
async def chat(request : ChatRequest):
    """Takes user queries, initializes the LLM and passes them to the LangChain Agent."""
    try:
        #Boot up the LLM
        llm=get_llm(request.provider,request.model,request.api_key,request.is_nvidia_key)

        #Route the query
        if request.vector_db_path and should_use_rag(request.query):
            #RAG mode: Use the tool calling agent
            response = answer_question_with_agent(request.query, llm, request.vector_db_path)
            if response is None:
                response = "I couldn't find relevant information for that in your uploaded documents."
        else:
            #Standard Mode :Just chat with the LLM directly
            response=llm.invoke(request.query).content

        return {"response":response}
    
    except Exception as e:
        raise HTTPException(status_code=500,detail=str(e))
