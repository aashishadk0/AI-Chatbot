import streamlit as st
from database import init_db
from auth import register_user, login_user, logout_user, require_login
from rag_utils import (
    save_uploaded_pdfs,
    parse_pdf_with_llama,
    get_parsed_pdfs,
    read_parsed_pdf,
    chunk_text
)
from rag_utils import (
    save_uploaded_pdfs,
    parse_pdf_with_llama,
    get_parsed_pdfs,
    read_parsed_pdf,
    chunk_text,
    create_vector_store,
    retrieve_chunks,
    generate_answer
)

st.set_page_config(
    page_title="PDF RAG Chatbot",
    page_icon="🤖",
    layout="wide"
)

init_db()

st.markdown("""
<style>
    .stApp {
        background-color: #0f172a;
        color: #e5e7eb;
    }

    h1, h2, h3, p, label {
        color: #e5e7eb !important;
    }

    .main-title {
        font-size: 38px;
        font-weight: 700;
        color: #22c55e;
        text-align: center;
        margin-bottom: 10px;
    }

    .subtitle {
        text-align: center;
        color: #94a3b8;
        margin-bottom: 30px;
    }

    .auth-box {
        max-width: 430px;
        margin: auto;
        padding: 30px;
        border-radius: 18px;
        background: #111827;
        border: 1px solid #1f2937;
    }

    .stButton > button {
        width: 100%;
        background-color: #22c55e;
        color: #020617;
        border: none;
        border-radius: 10px;
        padding: 10px;
        font-weight: 700;
    }

    .stButton > button:hover {
        background-color: #16a34a;
        color: white;
    }

    .chat-container {
        background: #020617;
        border-radius: 18px;
        padding: 25px;
        border: 1px solid #1e293b;
        min-height: 70vh;
    }

    .user-msg {
        background: #22c55e;
        color: #020617;
        padding: 12px 16px;
        border-radius: 16px;
        margin: 10px 0 10px auto;
        max-width: 70%;
        font-weight: 500;
    }

    .bot-msg {
        background: #1e293b;
        color: #e5e7eb;
        padding: 12px 16px;
        border-radius: 16px;
        margin: 10px auto 10px 0;
        max-width: 70%;
    }
</style>
""", unsafe_allow_html=True)


def auth_page():
    st.markdown("<div class='main-title'>PDF RAG Chatbot</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Login or register to start chatting with your PDFs.</div>", unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["Login", "Register"])

    with tab1:
        email = st.text_input("Email", key="login_email")
        password = st.text_input("Password", type="password", key="login_password")

        if st.button("Login"):
            if login_user(email, password):
                st.success("Login successful.")
                st.rerun()
            else:
                st.error("Invalid email or password.")

    with tab2:
        username = st.text_input("Username", key="register_username")
        email = st.text_input("Email", key="register_email")
        password = st.text_input("Password", type="password", key="register_password")
        confirm_password = st.text_input("Confirm Password", type="password")

        if st.button("Create Account"):
            if not username or not email or not password:
                st.error("All fields are required.")
            elif password != confirm_password:
                st.error("Passwords do not match.")
            else:
                success, message = register_user(username, email, password)
                if success:
                    st.success(message)
                else:
                    st.error(message)


def chatbot_page():
    with st.sidebar:
        st.title("🤖 PDF Chatbot")
        st.write(f"Welcome, **{st.session_state['username']}**")

        if st.button("New Chat"):
            st.session_state["messages"] = []

        st.divider()

        uploaded_files = st.file_uploader(
            "Upload PDFs",
            type=["pdf"],
            accept_multiple_files=True
        )

        if uploaded_files:
            if st.button("Parse PDFs"):
                saved_files = save_uploaded_pdfs(uploaded_files)

                with st.spinner("Parsing PDFs using LlamaParse..."):
                    for file_path in saved_files:
                        parse_pdf_with_llama(file_path)

                st.success("PDFs parsed successfully.")
                st.rerun()

        parsed_pdfs = get_parsed_pdfs()

        if parsed_pdfs:
            selected_pdf = st.selectbox(
                "Select PDF to chat with",
                parsed_pdfs
            )
        else:
            selected_pdf = None
            st.info("Upload and parse PDFs first.")

        st.divider()

        if st.button("Logout"):
            logout_user()
            st.rerun()

    st.title("Chat with your PDF")

    if selected_pdf:
        parsed_text = read_parsed_pdf(selected_pdf)
        chunks = chunk_text(parsed_text)

        tab1, tab2, tab3, tab4 = st.tabs([
    "Chat",
    "Parsed PDF",
    "Chunks",
    "Retrieved Chunks"
])

        with tab1:
            if "messages" not in st.session_state:
                st.session_state["messages"] = []

            st.markdown("<div class='chat-container'>", unsafe_allow_html=True)

            if not st.session_state["messages"]:
                st.markdown("""
                <div class='bot-msg'>
                    PDF selected. Now ask questions from this document.
                </div>
                """, unsafe_allow_html=True)

            for msg in st.session_state["messages"]:
                if msg["role"] == "user":
                    st.markdown(f"<div class='user-msg'>{msg['content']}</div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div class='bot-msg'>{msg['content']}</div>", unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True)

            question = st.chat_input("Ask a question from the selected PDF...")

            if question:
                with st.spinner("Retrieving relevant chunks and generating answer..."):
                    create_vector_store(selected_pdf, chunks)

                    retrieved_chunks = retrieve_chunks(
                        selected_pdf,
                        question,
                        top_k=3
                    )

                    st.session_state["retrieved_chunks"] = retrieved_chunks

                    answer = generate_answer(question, retrieved_chunks)

                st.session_state["messages"].append({
                    "role": "assistant",
                    "content": answer
                })
                st.rerun()

        with tab2:
            st.subheader("Parsed PDF Content")
            st.text_area(
                "Parsed Markdown/Text",
                parsed_text,
                height=500
            )

        with tab3:
            st.subheader("Text Chunks")

            for i, chunk in enumerate(chunks, start=1):
                with st.expander(f"Chunk {i}"):
                    st.write(chunk)

        with tab4:
            st.subheader("Retrieved Chunks Used for Answer")

            retrieved_chunks = st.session_state.get("retrieved_chunks", [])

            if retrieved_chunks:
                for i, chunk in enumerate(retrieved_chunks, start=1):
                    with st.expander(f"Retrieved Chunk {i}"):
                        st.write(chunk)
            else:
                st.info("Ask a question first to see retrieved chunks.")

    else:
        st.info("Please upload, parse, and select a PDF first.")

if require_login():
    chatbot_page()
else:
    auth_page()