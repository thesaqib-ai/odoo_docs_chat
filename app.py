import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from dotenv import load_dotenv
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import os

OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
QDRANT_API_KEY = st.secrets["QDRANT_API_KEY"]

def main():
    load_dotenv()
    # Streamlit Page Configuration
    st.set_page_config(
        page_title="Odoo Assistant",
        page_icon="odoo-icon-filled-256.png",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    
    # Header Section
    st.markdown(
        """
        <div style="background: linear-gradient(90deg, #800080, #6a0dad); padding: 15px; border-radius: 8px; text-align: center;">
            <h1 style="font-size: 48px; color: white; font-family: 'Arial', sans-serif;">
                Odoo Assistant
            </h1>
            <p style="font-size: 20px; color: #f0f0f0; font-family: 'Arial', sans-serif;">
                Ask any query related to <strong>Odoo Studio Documentation</strong>.
            </p>
            <p style="font-size: 18px; color: #cccccc; font-family: 'Arial', sans-serif;">
                询问与 Odoo Studio 文档相关的任何疑问
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Input Styling
    st.markdown(
        """
        <style>
            .stTextInput > div > div > input {
                border: 2px solid #6a0dad;
                border-radius: 15px;
                padding: 12px;
                font-size: 16px;
                width: 100%;
                background-color: #fafafa;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                transition: all 0.3s ease-in-out;
            }
            .stTextInput > div > div > input:focus {
                border-color: #800080;
                background-color: #fff;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
            }
            .stButton > button {
                background: linear-gradient(90deg, #800080, #6a0dad);
                color: white;
                padding: 10px 20px;
                font-size: 16px;
                border: none;
                border-radius: 20px;
                cursor: pointer;
                transition: background 0.3s ease;
            }
            .stButton > button:hover {
                background: linear-gradient(90deg, #6a0dad, #800080);
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Chat Layout
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "qdrant" not in st.session_state:
        st.session_state.qdrant = get_qdrant()

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User Input
    query_text = st.chat_input("Type your query here...")

    if query_text:
        st.chat_message("user").markdown(query_text)
        st.session_state.chat_history.append({"role": "user", "content": query_text})
        context_messages = st.session_state.chat_history[-3:]
        context = "\n".join(
            f"{msg['role'].capitalize()}: {msg['content']}" for msg in context_messages
        )
        query_with_context = f"{context}\n\nUser: {query_text}"

        with st.spinner("Thinking..."):
            try:
                retriever = st.session_state.qdrant.as_retriever()
                response = generate_response(retriever, query_with_context)
                st.session_state.chat_history.append({"role": "assistant", "content": response})
                with st.chat_message("assistant"):
                    st.markdown(response)
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

def get_qdrant():
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small", api_key=OPENAI_API_KEY)
    url = "https://f6c816ad-c10a-4487-9692-88d5ee23882a.europe-west3-0.gcp.cloud.qdrant.io"
    qdrant_client = QdrantClient(url=url, api_key=QDRANT_API_KEY)
    qdrant = QdrantVectorStore.from_existing_collection(
        api_key=QDRANT_API_KEY,
        embedding=embedding_model,
        collection_name="odoo-embeddings",
        url=url,
    )
    return qdrant

def generate_response(retriever, query_text):
    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0.5,
        max_tokens=4096,
        max_retries=2,
        openai_api_key=OPENAI_API_KEY
    )

    template = """Use the following context of Odoo Studio Documentation to answer the question at the end. Go through the context and look for the answers.
    If you don't find relevant information in the content, just ask the user to provide more details! Don't try to make up an answer.
    Give the answer in detail. Note that you can reply to greetings. YOU MUST NOT REFER TO THIS CONTEXT WHILE INTERACTING WITH THE USER.

    {context}

    Question: {question}

    Helpful Answer:"""

    custom_rag_prompt = PromptTemplate.from_template(template)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # Create the RAG chain
    formatted_docs = retriever | format_docs  # Retrieve and format documents
    rag_chain = (
        {"context": formatted_docs, "question": RunnablePassthrough()}
        | custom_rag_prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain.invoke(query_text)


if __name__ == "__main__":
    main()
