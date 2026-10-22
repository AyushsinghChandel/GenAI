from dotenv import load_dotenv
load_dotenv()

from pydantic import BaseModel
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from typing import Annotated

class ChatState(BaseModel):
    messages:Annotated[list,add_messages]

llm = ChatOllama(model="gemma3:1b")
def chatBotNode(state: ChatState):
    res = llm.invoke(state.messages)      # no "messages="
    return {"messages": [res]}


memory = InMemorySaver()
graph = StateGraph(ChatState)
graph.add_node("chat_bot", chatBotNode)
graph.add_edge(START, "chat_bot")
graph.add_edge("chat_bot", END)

graph = graph.compile(checkpointer=memory)


config = {"configurable": {"thread_id": "1"}}

while True:
    user_input = input("You: ")
    if user_input.lower() == "quit":
        print("Exiting the chat. Goodbye!")
        break
    res = graph.invoke({"messages":[{"role":"user","content":user_input}]}, config)
    print(f"Bot: {res['messages'][-1].content}")

