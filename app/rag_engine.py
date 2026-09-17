import uuid
from typing import Any, Dict, List

import chromadb
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer


class RAGEngine:
    def __init__(self, groq_api_key: str, model_name: str = "openai/gpt-oss-20b", embedding_model_name: str = "all-MiniLM-L6-v2"):
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY is not set. Please set the environment variable.")
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.llm = ChatGroq(groq_api_key=groq_api_key, model_name=model_name, temperature=0.1, max_tokens=1024)
        self.chroma_client = chromadb.Client()
        self.collection = self.chroma_client.create_collection(name="documents", metadata={"hnsw:space": "cosine"})
        self.document_names: List[str] = []

    def query(self, query: str, top_k: int = 3) -> str:
        pass

    def doc_splitter(self, documents: List[Document], chunk_size: int = 800, chunk_overlap: int = 200) -> List[Document]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
                                                  )
        split_docs = splitter.split_documents(documents)
        return split_docs

    def embedder(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.embedding_model.encode(texts, show_progress_bar=False, normalize_embeddings=True)

        return embeddings

    def add_documents(self, documents: List[Document]) -> int:
        chunks = self.doc_splitter(documents)
        texts = [doc.page_content for doc in chunks]
        embeddings = self.embedder(texts)
        
        ids = []
        metadatas = []
        embeddings_list = []
        
        for i, (doc, embedding) in enumerate(zip(chunks, embeddings)):
            ids.append(f"chunk_{uuid.uuid4().hex}")
            metadata = dict(doc.metadata)
            metadata["chunk_index"] = i
            metadata["content_length"] = len(doc.page_content)
            metadatas.append(metadata)

            embeddings_list.append(embedding.tolist())

        # add  everything to chroma
        self.collection.add(
            ids = ids,
            documents = texts,
            embeddings = embeddings_list,
            metadatas = metadatas
            )
        return len(chunks)
        
    
    def retrieve(self, query: str,  top_k: int = 6) -> List[Dict[str, Any]]:
        if self.collection.count() == 0:
            return []
        query_embedding = self.embedder([query])[0]
        number_of_results = min(top_k, self.collection.count())
        
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=number_of_results
            )

        retrieved_docs = []
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]
        ids = results["ids"][0]
        for doc, metadata, distance, id in zip(documents, metadatas, distances, ids):
            retrieved_docs.append({
                "id": id,
                "content": doc,
                "metadata": metadata,
                "distance": distance,
                "similarity_score": 1 - distance
                })

        return retrieved_docs
    def answer(self, query: str, chat_history: List[Dict[str, str]] | None = None, top_k: int = 3) -> Dict[str, Any]:
        retrieved_docs = self.retrieve(query, top_k=top_k)
        context_parts = []
        for result in retrieved_docs:
            metadata = result["metadata"]
            source = metadata.get("source", "Unknown")
            page = metadata.get("page")

            source_label = source
            if page is not None:
                source_label += f", (Page {int(page)+1})"
            context_parts.append(
                f"Source: {source_label}\n" 
                f"Content: \n{result['content']}"
                )

        context = "\n\n".join(context_parts)
        history_text = self._format_chat_history(chat_history or [])
        prompt = f"""
You are a document-question-answering assistant.

Answer the user's question using only the supplied document context.

Rules:
1. Do not invent facts that are not present in the context.
2. If the context is insufficient, clearly say that the uploaded documents
   do not contain enough information.
3. Give a clear and concise answer.
4. When useful, mention the source filename or page.
5. Previous conversation is provided only to understand follow-up questions.
   Document context remains the factual source of truth.

Previous conversation:
{history_text}

Document context:
{context}

User question:
{query}

Answer:
""".strip()

        response = self.llm.invoke([HumanMessage(content=prompt)])
        sources = self._build_source_list(retrieved_docs)
    
        return {
            "answer": response.content,
            "sources": sources,
        }

    @staticmethod
    def _format_chat_history(
        chat_history: List[Dict[str, str]],
        maximum_messages: int = 6,
    ) -> str:
        """
        Include a small amount of recent history for follow-up questions.
        """
        recent_messages = chat_history[-maximum_messages:]

        if not recent_messages:
            return "No previous conversation."

        return "\n".join(
            f"{message['role'].title()}: {message['content']}"
            for message in recent_messages
        )

    @staticmethod
    def _build_source_list(
        retrieved_documents: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        sources = []
        seen = set()

        for result in retrieved_documents:
            metadata = result["metadata"]
            source = metadata.get("source", "Unknown file")
            page = metadata.get("page")

            source_key = (source, page)

            if source_key in seen:
                continue

            seen.add(source_key)

            sources.append(
                {
                    "source": source,
                    "page": int(page) + 1 if page is not None else None,
                    "similarity_score": result["similarity_score"],
                }
            )

        return sources