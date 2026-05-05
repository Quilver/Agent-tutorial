from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from typing import List, Dict, Any, Optional
from pathlib import Path

class RAG:
    def __init__(self, 
                 embedding_model: str, 
                 llm_model: str,
                 vector_store_path: str, 
                 chunk_size: int = 1000, 
                 chunk_overlap: int = 200) -> None:
        # Initialize Ollama embeddings
        self.embeddings = OllamaEmbeddings(model=embedding_model)
        self.llm_model = llm_model
        self.vector_store_path = vector_store_path
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        
        # Load or initialize vector store
        self.vectorstore: Optional[FAISS] = self.__get_local_vectorstore(vector_store_path)
        self.rag_chain: Optional[Any] = None

    def __get_local_vectorstore(self, vector_store_path: str) -> Optional[FAISS]:
        if Path(vector_store_path).exists():
            print(f"Loading existing vector store from {vector_store_path}")
            return FAISS.load_local(vector_store_path, self.embeddings, allow_dangerous_deserialization=True)
        else:
            return None

    def ingest_document(self, document: Document, vector_store_path: str="not-important") -> None:
        """Add a Document to the vector store, splitting it into chunks."""
        split_docs = self.splitter.split_documents([document])
        
        if self.vectorstore is None:
            self.vectorstore = FAISS.from_documents(split_docs, self.embeddings)
        else:
            self.vectorstore.add_documents(split_docs)
        
        self.vectorstore.save_local(self.vector_store_path)

    def ingest_text(self, text: str, metadata: Optional[Dict[str, Any]] = None, vector_store_path: str="not-important") -> None:
        """Add raw text to the vector store as a Document."""
        doc = Document(page_content=text, metadata=metadata or {})
        self.ingest_document(doc)

    def retrieve(self, query: str, top_k: int = 2) -> List[Document]:
        """Retrieve documents similar to the query."""
        if self.vectorstore is None:
            raise RuntimeError("Vector store not initialized. Ingest documents first.")
        
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": top_k})
        return retriever.invoke(query)

    def create_rag_chain(self) -> None:
        """Initialize the LLM chain using Ollama."""
        llm = ChatOllama(model=self.llm_model, temperature=0.0)
        prompt = ChatPromptTemplate.from_template("""
            You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question.
            If you don't know the answer, just say that you don't know. Keep the answer concise.
            
            Context:
            {context}
            
            Question:
            {question}
        """)
        self.rag_chain = prompt | llm | StrOutputParser()

    def query(self, question: str) -> str:
        """Execute a RAG query using Ollama LLM and the vector store."""
        if self.rag_chain is None:
            raise RuntimeError("RAG chain not initialized. Call create_rag_chain() first.")
            
        context_docs = self.retrieve(question)
        context = "\n\n".join([doc.page_content for doc in context_docs])
        return self.rag_chain.invoke({"context": context, "question": question})
