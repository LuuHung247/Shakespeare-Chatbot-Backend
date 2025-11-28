from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma, FAISS
from langchain.schema import Document
from app.config import settings
from typing import List, Optional
import os

class VectorStoreManager:
    """Quản lý Vector Database cho RAG"""
    
    def __init__(self):
        self.embeddings = None
        self.vector_store = None
        self.db_type = settings.VECTOR_DB_TYPE
        
    def initialize_embeddings(self):
        """Khởi tạo embedding model"""
        self.embeddings = HuggingFaceEmbeddings(
            model_name=settings.EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'},  # hoặc 'cuda' nếu có GPU
            encode_kwargs={'normalize_embeddings': True}
        )
        return self.embeddings
    
    def create_vector_store(self, documents: List[Document]):
        """Tạo vector store từ documents"""
        if not self.embeddings:
            self.initialize_embeddings()
        
        if self.db_type == "chroma":
            self.vector_store = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory=settings.VECTOR_DB_PATH
            )
            self.vector_store.persist()
        
        elif self.db_type == "faiss":
            self.vector_store = FAISS.from_documents(
                documents=documents,
                embedding=self.embeddings
            )
            # Save FAISS index
            os.makedirs(settings.VECTOR_DB_PATH, exist_ok=True)
            self.vector_store.save_local(settings.VECTOR_DB_PATH)
        
        return self.vector_store
    
    def load_vector_store(self):
        """Load vector store đã tạo"""
        if not self.embeddings:
            self.initialize_embeddings()
        
        try:
            if self.db_type == "chroma":
                self.vector_store = Chroma(
                    persist_directory=settings.VECTOR_DB_PATH,
                    embedding_function=self.embeddings
                )
            
            elif self.db_type == "faiss":
                self.vector_store = FAISS.load_local(
                    settings.VECTOR_DB_PATH,
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
            
            return self.vector_store
        
        except Exception as e:
            raise Exception(f"Error loading vector store: {str(e)}")
    
    def similarity_search(self, query: str, k: int = None) -> List[Document]:
        """Tìm kiếm documents tương tự"""
        if not self.vector_store:
            self.load_vector_store()
        
        k = k or settings.TOP_K_RETRIEVAL
        
        try:
            docs = self.vector_store.similarity_search(query, k=k)
            return docs
        except Exception as e:
            raise Exception(f"Error in similarity search: {str(e)}")
    
    def add_documents(self, documents: List[Document]):
        """Thêm documents vào vector store"""
        if not self.vector_store:
            self.load_vector_store()
        
        self.vector_store.add_documents(documents)
        
        if self.db_type == "chroma":
            self.vector_store.persist()
        elif self.db_type == "faiss":
            self.vector_store.save_local(settings.VECTOR_DB_PATH)

# Singleton instance
vector_store_manager = VectorStoreManager()