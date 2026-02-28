import streamlit as st
from google import genai
from dotenv import load_dotenv
import os

# Load API key from .env
load_dotenv()
api_key = os.getenv("GENERATIVEAI_API_KEY")
if not api_key:
    st.error("Missing API key in .env")
    st.stop()

# Initialize Gemini client
client = genai.Client(api_key=api_key)

# Page config
st.set_page_config(page_title="AI Ethics Chatbot", layout="wide")

# Sidebar for model selection
st.sidebar.title("Settings")
model = st.sidebar.selectbox("Choose model", ["gemini-3-flash-preview", "gemini-1.5-flash"])
st.sidebar.info("Ask questions about AI ethics and get thoughtful answers!")

# Initialize session state for messages
if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("💬 AI Ethics Chatbot")
st.markdown("Ask me anything about AI ethics. I’ll respond thoughtfully!")

# Display chat history
for msg in st.session_state.messages:
    color = "#F5F3FA" if msg["role"] == "user" else "#D6CFF1"
    st.markdown(
        f"<div style='text-align: left; background-color:{color}; padding:10px; border-radius:10px; margin:5px 0; max-width:70%;'>{msg['content']}</div>",
        unsafe_allow_html=True
    )

# Chat input
if prompt := st.chat_input("Ask about AI ethics:"):
    # Save and display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.markdown(
        f"<div style='text-align: left; background-color:#F5F3FA; padding:10px; border-radius:10px; margin:5px 0; max-width:70%;'>{prompt}</div>",
        unsafe_allow_html=True
    )

    # Call Gemini API
    try:
        response = client.models.generate_content(model=model, contents=prompt)
        assistant_reply = response.text
    except Exception as e:
        assistant_reply = f"❌ API request failed: {e}"

    # Save and display assistant reply
    st.session_state.messages.append({"role": "assistant", "content": assistant_reply})
    st.markdown(
        f"<div style='text-align: left; background-color:#D6CFF1; padding:10px; border-radius:10px; margin:5px 0; max-width:70%;'>{assistant_reply}</div>",
        unsafe_allow_html=True
    )