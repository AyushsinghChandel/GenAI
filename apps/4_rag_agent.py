from dotenv import load_dotenv
load_dotenv()  

from langchain_community.document_loaders import PyPDFLoader, PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import InMemoryVectorStore
from langchain_ollama import ChatOllama
from langchain.tools import tool
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
import streamlit as st


### Data in st session state
if "document_uploaded" not in st.session_state:
    st.session_state.document_uploaded = False
if "agent" not in st.session_state:
    st.session_state.agent = None
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "messages" not in st.session_state:
    st.session_state.messages = []



def process_document(path):

    ### Loading Documents
    loader = PyPDFDirectoryLoader(path)
    docs = loader.load()

    ### Splitting Documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splitted_docs = text_splitter.split_documents(docs)

    ### Creating Embeddings and Vectorstore
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectorstore = InMemoryVectorStore.from_documents(splitted_docs, embeddings)

    ### Create a Agent - tool, llm, system_prompt
    llm = ChatOllama(model="qwen2.5:3b", temperature=0)

    @tool
    def retriever_tool(query:str):
        """Retrieves relevant information from the vectorstore based on the query."""
        print(f"Query received: {query}")
        same_record = vectorstore.similarity_search(query, k=3)
        context = ""
        for record in same_record:
            context += record.page_content + "\n"
        return context

    system_prompt = """You are a helpful assistant. ALWAYS call retriever_tool first to search the documents before answering any question and if you want you can call retriever_tool multiple times before answering. Answer only from the retrieved context."""

    memory = InMemorySaver()

    agent = create_agent(
        model=llm,
        tools=[retriever_tool],
        system_prompt=system_prompt,
        checkpointer=memory
    )
    st.session_state.agent = agent
    st.session_state.document_uploaded = True

### upload Ui
if not st.session_state.document_uploaded:
    uploaded = st.file_uploader("Upload a PDF document", type=["pdf"], accept_multiple_files=True)
    if uploaded:
        with st.spinner("Processing the uploaded document..."):
            path = "./docs_files/"
            for file in uploaded:
                with open(path + file.name, "wb") as f:
                    f.write(file.getvalue())
            process_document(path)
            st.rerun()


### chat ui
if st.session_state.document_uploaded and st.session_state.agent:
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.chat_message("user").markdown(message["content"])
        else:
            st.chat_message("assistant").markdown(message["content"])
             
    query = st.chat_input("Ask a question about the document:")
    if query:
        st.session_state.messages.append({"role": "user", "content": query})
        st.chat_message("user").markdown(query)
        with st.spinner("Generating response..."):
            response = st.session_state.agent.invoke(
                {"messages": [{"role": "user", "content": query}]},
                {"configurable": {"thread_id": 1}}
            )
            answer = response["messages"][-1].content
            st.chat_message("assistant").markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})