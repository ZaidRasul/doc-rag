import os
import streamlit as st
from dotenv import load_dotenv
from rag_engine import RAGEngine
from document_loader import load_files

load_dotenv()

st.set_page_config(
    page_title="Document RAG",
    page_icon="📄",
    layout="wide",
    #write="upload rag docs"
)
st.write("Upload your documents and ask questions about them.")
st.write("Currently supported file types: PDF, TXT, MD, Docx")


def initialize_session_state():
    if "rag_engine" not in st.session_state:
        groq_api_key = os.getenv("GROQ_API_KEY")

        if not groq_api_key:
            st.error(
                "GROQ_API_KEY is missing. Add it to the project-root .env file."
            )
            st.stop()

        with st.spinner("Loading embedding model..."):
            st.session_state.rag_engine = RAGEngine(
                groq_api_key=groq_api_key
            )

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "documents_ready" not in st.session_state:
        st.session_state.documents_ready = False

    if "processed_file_signature" not in st.session_state:
        st.session_state.processed_file_signature = None


def make_file_signature(uploaded_files) -> tuple:
    """
    Identify the selected files so Streamlit does not reprocess them
    after every chat message.
    """
    return tuple(
        sorted(
            (
                uploaded_file.name,
                uploaded_file.size,
            )
            for uploaded_file in uploaded_files
        )
    )


initialize_session_state()

st.title("📚 Chat with your documents")
st.caption(
    "Upload PDF, TXT, or Markdown files, process them, and ask questions."
)

with st.sidebar:
    st.header("Documents")

    uploaded_files = st.file_uploader(
        "Upload your files",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
    )

    top_k = st.slider(
        "Number of chunks to retrieve",
        min_value=1,
        max_value=10,
        value=6,
    )

    process_button = st.button(
        "Process documents",
        type="primary",
        use_container_width=True,
        disabled=not uploaded_files,
    )

    clear_button = st.button(
        "Clear documents and chat",
        use_container_width=True,
    )

    if clear_button:
        st.session_state.rag_engine.clear_documents()
        st.session_state.messages = []
        st.session_state.documents_ready = False
        st.session_state.processed_file_signature = None
        st.rerun()

    if process_button:
        current_signature = make_file_signature(uploaded_files)

        try:
            all_documents = []

            with st.spinner("Reading uploaded files..."):
                for uploaded_file in uploaded_files:
                    file_documents = load_files(
                        filename=uploaded_file.name,
                        file_bytes=uploaded_file.getvalue(),
                    )
                    all_documents.extend(file_documents)

            # Replace the previous uploaded-document collection.
            st.session_state.rag_engine.clear_documents()

            with st.spinner("Chunking and embedding documents..."):
                chunk_count = (
                    st.session_state.rag_engine.add_documents(
                        all_documents
                    )
                )

            st.session_state.documents_ready = True
            st.session_state.processed_file_signature = current_signature
            st.session_state.messages = []

            st.success(
                f"Processed {len(uploaded_files)} file(s) "
                f"into {chunk_count} chunks."
            )

        except Exception as error:
            st.session_state.documents_ready = False
            st.error(f"Could not process the files: {error}")

    if st.session_state.documents_ready:
        st.success("Documents are ready.")

        st.write("Loaded files:")

        for filename in st.session_state.rag_engine.document_names:
            st.write(f"- {filename}")
    else:
        st.info("Upload files and click Process documents.")

# Show the existing conversation.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message.get("sources"):
            with st.expander("Sources"):
                for source in message["sources"]:
                    label = source["source"]

                    if source["page"] is not None:
                        label += f" — page {source['page']}"

                    st.write(
                        f"- {label} "
                        f"(similarity: {source['similarity_score']:.2f})"
                    )

question = st.chat_input(
    "Ask a question about your documents",
    disabled=not st.session_state.documents_ready,
)

if question:
    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching the documents..."):
            try:
                result = (
                    st.session_state.rag_engine.answer_question(
                        question=question,
                        chat_history=st.session_state.messages[:-1],
                        top_k=top_k,
                    )
                )

                answer = result["answer"]
                sources = result["sources"]

                st.markdown(answer)

                if sources:
                    with st.expander("Sources"):
                        for source in sources:
                            label = source["source"]

                            if source["page"] is not None:
                                label += f" — page {source['page']}"

                            st.write(
                                f"- {label} "
                                f"(similarity: "
                                f"{source['similarity_score']:.2f})"
                            )

            except Exception as error:
                answer = f"Unable to generate an answer: {error}"
                sources = []
                st.error(answer)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": sources,
        }
    )
