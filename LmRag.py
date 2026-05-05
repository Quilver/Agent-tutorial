from langchain_text_splitters  import RecursiveCharacterTextSplitter
from typing import List, Dict, Any, Optional
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from pathlib import Path

class RAG:
    def __init__(self, embedding_model, vector_store_path: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> None:
        self.embeddings = embedding_model
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.vectorstore: Optional[FAISS] = self.__get_local_vectorstore(vector_store_path)
        self.rag_chain: Optional[Any] = None
    def __get_local_vectorstore(self, vector_store_path: str) -> Optional[FAISS]:
        # If vector db exists, load it. Otherwise, return None.
        if Path(vector_store_path).exists():
            print(f"Loading existing vector store from {vector_store_path}")
            return FAISS.load_local(vector_store_path, self.embeddings, allow_dangerous_deserialization=True)
        else:
            return None
    def ingest_document(self, document: Document, vector_store_path: str) -> None:
        """Add a Document to the vector store, splitting it into chunks."""
        split_docs: List[Document] = self.splitter.split_documents([document])
        
        if self.vectorstore is None:
            self.vectorstore = FAISS.from_documents(split_docs, self.embeddings)
        else:
            self.vectorstore.add_documents(split_docs)
        
        self.vectorstore.save_local(vector_store_path)
    def ingest_text(self, text: str, vector_store_path: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Add raw text to the vector store as a Document."""
        doc = Document(page_content=text, metadata=metadata or {})
        self.ingest_document(doc, vector_store_path)
    def retrieve(self, query: str, top_k: int = 2) -> List[Document]:
        """Retrieve documents similar to the query."""
        if self.vectorstore is None:
            raise RuntimeError("Vector store not initialized. Ingest documents first.")
        
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": top_k})
        return retriever.invoke(query)
