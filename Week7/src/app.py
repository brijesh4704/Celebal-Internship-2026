import streamlit as st
from vectorstore import VectorStore
from chatbot import Chatbot

st.set_page_config(
    page_title="Document Question Answering System (RAG)",
    page_icon="📄",
    layout="wide"
)


def main():

    st.title("📄 Document Question Answering System (RAG)")
    st.write("Upload a PDF document and ask questions based on its content.")

    # ---------------- Session State ----------------

    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None

    if "chatbot" not in st.session_state:
        st.session_state.chatbot = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # ---------------- Sidebar ----------------

    with st.sidebar:

        st.header("🔑 Cohere API")

        cohere_api_key = st.text_input(
            "Cohere API Key",
            type="password"
        )

        st.markdown("---")

        uploaded_file = st.file_uploader(
            "Upload PDF",
            type=["pdf"]
        )

        if uploaded_file:
            st.success(f"Uploaded: {uploaded_file.name}")

        if st.button("📄 Process Document"):

            if uploaded_file is None:
                st.error("Please upload a PDF.")
                return

            if not cohere_api_key:
                st.error("Please enter your Cohere API Key.")
                return

            with open("uploaded_document.pdf", "wb") as f:
                f.write(uploaded_file.getbuffer())

            with st.spinner("Processing document..."):

                st.session_state.vectorstore = VectorStore(
                    "uploaded_document.pdf"
                )

                st.session_state.chatbot = Chatbot(
                    st.session_state.vectorstore,
                    cohere_api_key
                )

            st.success("✅ Document processed successfully!")

        if st.button("🗑 Clear Chat"):
            st.session_state.chat_history = []
            st.success("Chat history cleared.")

    # ---------------- Ask Question ----------------

    if st.session_state.chatbot:

        st.markdown("---")

        question = st.text_input(
            "Ask a question about your document"
        )

        if st.button("🚀 Get Answer"):

            if not question.strip():
                st.warning("Please enter a question.")
                return

            with st.spinner("Generating answer..."):

                answer, retrieved_docs = (
                    st.session_state.chatbot.respond(question)
                )

                st.session_state.chat_history.append(
                    (question, answer)
                )

            st.subheader("📌 Answer")
            st.success(answer)

            if retrieved_docs:

                st.subheader("📖 Retrieved Context")

                for i, doc in enumerate(retrieved_docs, start=1):

                    with st.expander(f"Chunk {i}"):

                        st.write(doc["text"])

    # ---------------- Chat History ----------------

    if st.session_state.chat_history:

        st.markdown("---")
        st.header("💬 Conversation History")

        for question, answer in reversed(st.session_state.chat_history):

            with st.expander(question):

                st.markdown("**Answer**")
                st.write(answer)


if __name__ == "__main__":
    main()