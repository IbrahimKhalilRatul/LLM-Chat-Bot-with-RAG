from rag_pline import answer_query, retrieve_docs, llm_model

# Step 1: Setup Upload PDF functionality
import streamlit as st

# Step 2: Chatbot Skeleton (Question & Answer)
user_query = st.text_area("Enter your prompt: ", height=150, placeholder="Ask Anything!")

ask_question = st.button("Ask AI Model")

@st.cache_data
def retrieve_docs_cached(query):
    return retrieve_docs(query)

@st.cache_data
def answer_query_cached(_documents, _model, query):
    return answer_query(_documents, _model, query)

error_placeholder = st.empty()

if ask_question:
    error_placeholder.empty()
    st.chat_message("user").write(user_query)

    # RAG Pipeline
    retrieved_docs = retrieve_docs_cached(user_query)
    response = answer_query_cached(_documents=retrieved_docs, _model=llm_model, query=user_query)
    st.chat_message("Ask AI").write(response)
else:
    error_placeholder.error("Kindly ask a valid Question!")