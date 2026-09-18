# doc-rag

Implemention of RAG pipeline
First data is loaded from pdf and rag prep(chunking, embeddings) are performed and
stored in vector store(vec store used: chromadb).

Retreiver functions takes a user query, performs the same rag prep on query and does
similarity search and retrieves the similar data from store.

This query and data is fed into llm as context and llm gives a response.

Flow:

```text

                 Streamlit
                    │
          User uploads 3 files
                    │
                    ▼
              load_files()
                    │
       ┌────────────┼────────────┐
       ▼            ▼            ▼
   python.pdf     ml.pdf      notes.txt
       │            │            │
       ▼            ▼            ▼
   load_pdf()   load_pdf()   load_txt()
       │            │            │
       └────────────┼────────────┘
                    ▼
              all_documents
                    │
                    ▼
               Text Splitter
                    │
                    ▼
                Embeddings
                    │
                    ▼
                 Chroma

```
