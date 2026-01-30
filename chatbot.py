import os
from dotenv import load_dotenv
import streamlit as st
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage,AIMessage,SystemMessage


#load the env variables
load_dotenv()

#streamlit page setup 
st.set_page_config(
    page_title="ShinzoGPT",
    page_icon="🤖",
    layout="centered"
)
st.title("💬ShinzoGPT")

col1,col2=st.columns(2)

with col1:
    provider=st.selectbox(
        "Select Provider",
        ["Groq", "Gemini","Moonshot Kimi"]
    )

api_key=None
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
    selected_model=None
    if provider=="Groq":
        selected_model=st.selectbox("Model:", ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"])

    elif provider=="Gemini":
        selected_model=st.selectbox("Model:", ["gemini-2.5-flash", "gemini-3-flash-preview"])

    elif provider=="Moonshot Kimi":
        if is_nvidia_key:
            # NVIDIA-hosted Kimi model names
            selected_model = st.selectbox("Model (NVIDIA):", 
                ["moonshotai/kimi-k2.5", "moonshotai/kimi-k2-thinking"]
            )
        else:
            # Official Moonshot model names
            selected_model = st.selectbox("Model:", 
                ["kimi-k2.5", "moonshot-v1-8k", "moonshot-v1-32k"]
            )

#llm initialization function
def get_llm():
    if not api_key:
        return None
    
    if provider=="Groq":
        return ChatGroq(model=selected_model,api_key=api_key)
    
    elif provider == "Gemini":
        return ChatGoogleGenerativeAI(model=selected_model, google_api_key=api_key)
    
    elif provider == "Moonshot Kimi":
        if is_nvidia_key:
            return ChatOpenAI(
                model=selected_model, 
                api_key=api_key, 
                base_url="https://integrate.api.nvidia.com/v1"
            )
        else:
            return ChatOpenAI(
                model=selected_model, 
                api_key=api_key, 
                base_url="https://api.moonshot.cn/v1"
            )
    return None

#chat logic
if "chat_history" not in st.session_state:
    st.session_state.chat_history=[]

#display chat history 
for message in st.session_state.chat_history:
    role=message["role"]
    content=message["content"]
    with st.chat_message(role):
        st.markdown(content)

#Input box
user_prompt=st.chat_input("Ask Chatbot...")

if user_prompt:
    st.chat_message("user").markdown(user_prompt)
    st.session_state.chat_history.append({"role":"user","content":user_prompt})

    llm=get_llm()

    if not llm:
        st.error(f"⚠️ API Key for {provider} is missing. Please check your .env file or enter it above.")
    else:
        langchain_history=[SystemMessage(content="You are a helpful assistant")]

        for msg in st.session_state.chat_history:
            if msg["role"]=="user":
                langchain_history.append(HumanMessage(content=msg["content"]))
            elif msg["role"]=="assistant":
                langchain_history.append(AIMessage(content=msg["content"]))
        
        with st.chat_message("assistant"):
            try:
                with st.spinner("Thinking..."):
                    response=llm.invoke(langchain_history)
                    assistant_response=response.content
                    st.markdown(assistant_response)
                    st.session_state.chat_history.append({"role":"assistant","content":assistant_response})
            except Exception as e:
                st.error(f"Error: {str(e)}")

