import os
from dotenv import load_dotenv
import streamlit as st
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# Import functions
from rag_utility import process_documents_to_chroma_db, answer_question_with_rag

load_dotenv()

st.set_page_config(
    page_title="ShinzoGPT",
    page_icon="🤖",
    layout="centered"
)
st.title("💬 ShinzoGPT")

# --- CONFIGURATION ---
col1, col2 = st.columns(2)

with col1:
    provider = st.selectbox("Select Provider", ["Groq", "Gemini", "Moonshot Kimi"])

api_key = None
is_nvidia_key = False

if provider == "Groq":
    api_key = os.getenv("GROQ_API_KEY")
elif provider == "Gemini":
    api_key = os.getenv("GOOGLE_API_KEY")
elif provider == "Moonshot Kimi":
    api_key = os.getenv("MOONSHOT_API_KEY")
    if api_key and api_key.startswith("nvapi-"):
        is_nvidia_key = True

with col2:
    selected_model = None
    if provider == "Groq":
        selected_model = st.selectbox("Model:", ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"])
    elif provider == "Gemini":
        selected_model = st.selectbox("Model:", ["gemini-2.0-flash", "gemini-1.5-flash"])
    elif provider == "Moonshot Kimi":
        if is_nvidia_key:
            selected_model = st.selectbox("Model (NVIDIA):", ["moonshotai/kimi-k2.5", "moonshotai/kimi-k2-thinking"])
        else:
            selected_model = st.selectbox("Model:", ["kimi-k2.5", "moonshot-v1-8k", "moonshot-v1-32k"])

# --- LLM SETUP ---
def get_llm():
    if not api_key: return None
    if provider == "Groq": return ChatGroq(model=selected_model, api_key=api_key)
    elif provider == "Gemini": return ChatGoogleGenerativeAI(model=selected_model, google_api_key=api_key)
    elif provider == "Moonshot Kimi":
        base = "https://integrate.api.nvidia.com/v1" if is_nvidia_key else "https://api.moonshot.cn/v1"
        return ChatOpenAI(model=selected_model, api_key=api_key, base_url=base)
    return None

# --- STATE MANAGEMENT ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
# New: Store the path to the current vector DB
if "vector_db_path" not in st.session_state:
    st.session_state.vector_db_path = None

# --- UPLOAD SECTION ---
with st.popover("➕ Attach Knowledge", help="Upload PDF documents"):
    st.markdown("### 📂 Upload Files")
    uploaded_files = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True, label_visibility="collapsed")
    
    col_a, col_b = st.columns(2)
    with col_a:
        if uploaded_files:
            if st.button("Process Documents", type="primary"):
                with st.spinner("Analyzing documents..."):
                    try:
                        # Capture the NEW unique folder path
                        new_db_path = process_documents_to_chroma_db(uploaded_files)
                        # Save it to session state
                        st.session_state.vector_db_path = new_db_path
                        st.success("✅ RAG Active!")
                    except Exception as e:
                        st.error(f"Error: {e}")
    with col_b:
        if st.button("🗑️ Reset Chat"):
            st.session_state.chat_history = []
            st.session_state.vector_db_path = None # Clear the DB path reference
            st.rerun()

# --- DISPLAY CHAT ---
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- INPUT HANDLING ---
user_prompt = st.chat_input("Ask Chatbot...")

if user_prompt:
    st.chat_message("user").markdown(user_prompt)
    st.session_state.chat_history.append({"role": "user", "content": user_prompt})

    llm = get_llm()

    if not llm:
        st.error(f"⚠️ API Key for {provider} is missing.")
    else:
        with st.chat_message("assistant"):
            try:
                with st.spinner("Thinking..."):
                    
                    # LOGIC: Check if we have a valid DB path
                    if st.session_state.vector_db_path:
                        # RAG MODE: Pass the specific DB path
                        answer, sources = answer_question_with_rag(user_prompt, llm, st.session_state.vector_db_path)
                        
                        sources_text = f"\n\n**Sources:** *{', '.join(sources)}*" if sources else ""
                        full_response = f"**📄 RAG Answer:**\n\n{answer}{sources_text}"
                        
                        st.markdown(full_response)
                        st.session_state.chat_history.append({"role": "assistant", "content": full_response})
                        
                    else:
                        # STANDARD MODE
                        langchain_history = [SystemMessage(content="You are a helpful assistant")]
                        for msg in st.session_state.chat_history:
                            if msg["role"] == "user": langchain_history.append(HumanMessage(content=msg["content"]))
                            elif msg["role"] == "assistant": langchain_history.append(AIMessage(content=msg["content"]))
                        
                        response = llm.invoke(langchain_history)
                        st.markdown(response.content)
                        st.session_state.chat_history.append({"role": "assistant", "content": response.content})
                        
            except Exception as e:
                st.error(f"Error: {str(e)}")
