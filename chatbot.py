# chatbot.py

import os
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_groq import ChatGroq
from qdrant_client import QdrantClient
from langchain_qdrant import Qdrant
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain.vectorstores import FAISS
import faiss





class ChatbotManager:
    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en",
        device: str = "cpu",
        encode_kwargs: dict = {"normalize_embeddings": True},
        llm_model: str = "llama3-70b-8192",
        llm_temperature: float = 0.7,
        faiss_index_path: str = "faiss_index",
    ):
        """
        Initializes the ChatbotManager with embedding models, LLM, and vector store.
        """
        self.model_name = model_name
        self.device = device
        self.encode_kwargs = encode_kwargs
        self.llm_model = llm_model
        self.llm_temperature = llm_temperature
        self.faiss_index_path = faiss_index_path

        # Initialize embeddings
        self.embeddings = HuggingFaceBgeEmbeddings(
            model_name=self.model_name,
            model_kwargs={"device": self.device},
            encode_kwargs=self.encode_kwargs,
        )
        if os.path.exists(self.faiss_index_path):
            index = faiss.read_index(self.faiss_index_path)
            self.db = FAISS(embedding_function=self.embeddings.embed_query, index=index)
        else:
            raise FileNotFoundError("FAISS index file not found. Please create the embeddings first.")

        # Initialize the Groq LLM model
        self.llm = ChatGroq(
            model_name=self.llm_model,
            temperature=self.llm_temperature,
        )

        # Define the prompt template
        self.prompt_template = """Use the following pieces of information to answer the user's question.
            If you don't know the answer, just say that you don't know, don't try to make up an answer.

            Context: {context}
            Question: {question}

            Only return the helpful answer. Answer must be detailed and well explained.
            Helpful answer:
            """
        
        # Initialize Qdrant client
        #self.qdrant_client = QdrantClient(url=self.qdrant_url, api_key=self.api_key, prefer_grpc=False)

        # Initialize the Qdrant vector store
        # self.db = Qdrant(
        #     client=self.qdrant_client,
        #     embeddings=self.embeddings,
        #     collection_name=self.collection_name
        # )

        # Initialize the prompt
        self.prompt = PromptTemplate(
            template=self.prompt_template,
            input_variables=['context', 'question']
        )

        # Initialize the retriever
        self.retriever = self.db.as_retriever(search_kwargs={"k": 1})

        # Define chain type kwargs
        self.chain_type_kwargs = {"prompt": self.prompt}

        # Initialize the RetrievalQA chain with the LLM and retriever
        self.qa = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            return_source_documents=False,
            chain_type_kwargs=self.chain_type_kwargs,
            verbose=False
        )

    def get_response(self, query: str) -> str:
        """
        Processes the user's query and returns the chatbot's response.
 
        Args:
            query (str): The user's input question.
 
        Returns:
            str: The chatbot's response.
        """
        try:
            docs = self.db.similarity_search(query, k=1)
            response = self.qa.run(query)
            return response
        except Exception as e:
            return "⚠️ Sorry, I couldn't process your request at the moment."

# Example usage of the chatbot
# if __name__ == "__main__":
#     chatbot = ChatbotManager()

#     # Example user query
#     query = "Recommendations for enterprises to accelerate digital adoption in the AI and skills-first era?"

#     # Get and print the chatbot's response
#     response = chatbot.get_response(query)
#     print(response)