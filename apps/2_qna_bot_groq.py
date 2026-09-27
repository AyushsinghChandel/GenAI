from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver
from datetime import date
import streamlit as st

llm = ChatGroq(model="openai/gpt-oss-20b", streaming = True)
search = GoogleSerperAPIWrapper()
tools = [search.run]


if "memory" not in st.session_state:
    st.session_state.memory = MemorySaver()
    st.session_state.history = []


agent = create_agent(
    model=llm,
    tools=tools,
    checkpointer=st.session_state.memory,
    system_prompt=(
        "You are a helpful and a smart assiststant and can search on the google as well"
    ),
)


## BUILDING WEB INTERFACE
st.subheader("QuickAnswer - Answers at the speed of thought")

for message in st.session_state.history:
    role = message["role"]
    content = message["content"]
    st.chat_message(role).markdown(content)

query = st.chat_input("Ask your question here")
if query:
    st.chat_message("user").markdown(query)
    st.session_state.history.append({"role": "user", "content": query})
    response = agent.stream(
        {"messages": [{"role": "user", "content": query}]},
        {"configurable": {"thread_id": "1"}},
        stream_mode = "messages"
    )

    ai_container = st.chat_message("assistant")
    with ai_container:
        space = st.empty()
        msg = ""

        for chunk in response:
            msg = msg + chunk[0].content
            space.write(msg)


    st.session_state.history.append({"role": "assistant", "content": msg})
    
