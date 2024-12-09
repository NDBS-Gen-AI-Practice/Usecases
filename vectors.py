import os
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import Qdrant
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http import models
from unstructured.partition.pptx import partition_pptx
from unstructured.partition.docx import partition_docx
import faiss
from langchain.vectorstores import FAISS

# Load environment variables from .env
load_dotenv('.env')

class EmbeddingsManager:
    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en",
        device: str = "cpu",
        encode_kwargs: dict = {"normalize_embeddings": True},
        faiss_index_path: str = "faiss_index",
    ):
        self.model_name = model_name
        self.device = device
        self.encode_kwargs = encode_kwargs
        self.faiss_index_path = faiss_index_path
        self.embeddings = HuggingFaceBgeEmbeddings(
            model_name=self.model_name,
            model_kwargs={"device": self.device},
            encode_kwargs=self.encode_kwargs,
        )
        self.index = None

    def _load_document(self, file_path: str):
        extension = os.path.splitext(file_path)[-1].lower()
        if extension == ".pdf":
            loader = UnstructuredPDFLoader(file_path)
            return loader.load()
        elif extension == ".txt":
            with open(file_path, "r", encoding="utf-8") as file:
                text = file.read()
            return [{"text": text}]
        else:
            raise ValueError(f"Unsupported file type: {extension}")

    def create_embeddings(self, file_path: str):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file {file_path} does not exist.")

        docs = self._load_document(file_path)
        if not docs:
            raise ValueError("No documents were loaded from the file.")

        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ". ", " ", ""],
            chunk_size=1000,
            chunk_overlap=250
        )
        splits = text_splitter.create_documents(docs)
        if not splits:
            raise ValueError("No text chunks were created from the documents.")

        # Create embeddings and store in FAISS
        try:
            vector_store = FAISS.from_documents(splits, self.embeddings)
            # Save the FAISS index to a file
            faiss.write_index(vector_store.index, self.faiss_index_path)
        except Exception as e:
            raise RuntimeError(f"Failed to create FAISS index: {e}")

        return f"FAISS index successfully created and stored at {self.faiss_index_path}!"