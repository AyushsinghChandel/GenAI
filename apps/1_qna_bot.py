from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
import streamlit as st

llm = ChatGoogleGenerativeAI(
    model="gemini-3.7-flash",
    temperature=1.0,
    max_retries=0,
)

st.title("AskBuddy - Your Personal Q&A Bot")
st.markdown("AskBuddy is a Q&A bot that can answer your questions. Just type your question in the input box below and hit enter.")

if "messages" not in st.session_state:
    st.session_state.messages = []


def extract_text(content):
    if isinstance(content, str):
        return content
    return "".join(
        block.get("text", "")
        for block in content
        if isinstance(block, dict) and block.get("type") == "text"
    )


for message in st.session_state.messages:
    role = message["role"]
    content = message["content"]
    st.chat_message(role).markdown(content)


query = st.chat_input("Ask me anything!")
if query:
    st.session_state.messages.append({"role": "user", "content": query})
    st.chat_message("user").markdown(query)

    res = llm.invoke(query)
    answer = extract_text(res.content)

    st.session_state.messages.append({"role": "ai", "content": answer})
    st.chat_message("ai").markdown(answer)