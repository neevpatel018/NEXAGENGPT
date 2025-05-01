import os
from typing import List, Dict
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

class RAGEngine:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        self.vector_store = None

    def load_documents(self, file_path: str) -> List[Dict]:
        """
        Load documents from a file and split them into chunks.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            List[Dict]: List of document chunks
        """
        if file_path.endswith('.pdf'):
            loader = PyPDFLoader(file_path)
        else:
            loader = TextLoader(file_path)
            
        documents = loader.load()
        return self.text_splitter.split_documents(documents)

    def create_vector_store(self, documents: List[Dict]):
        """
        Create a vector store from document chunks.
        
        Args:
            documents: List of document chunks
        """
        self.vector_store = FAISS.from_documents(
            documents=documents,
            embedding=self.embeddings
        )
        # Save the vector store
        os.makedirs("vector_store", exist_ok=True)
        self.vector_store.save_local("vector_store/faiss_index")

    def load_vector_store(self):
        """
        Load an existing vector store from disk.
        """
        if os.path.exists("vector_store/faiss_index"):
            self.vector_store = FAISS.load_local(
                "vector_store/faiss_index",
                self.embeddings
            )

    def query(self, query: str, k: int = 4) -> List[Dict]:
        """
        Query the vector store for similar documents.
        
        Args:
            query: Query string
            k: Number of results to return
            
        Returns:
            List[Dict]: List of similar documents
        """
        if not self.vector_store:
            raise ValueError("Vector store not initialized. Load or create one first.")
            
        return self.vector_store.similarity_search(query, k=k) 