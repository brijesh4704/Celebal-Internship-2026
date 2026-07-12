import fitz
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter


class VectorStore:

    def __init__(self, pdf_path):

        self.pdf_path = pdf_path

        # Embedding model
        self.embedding_model = SentenceTransformer(
            "all-MiniLM-L6-v2"
        )

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100
        )

        self.chunks = []
        self.index = None

        self.load_pdf()
        self.split_text()
        self.create_vector_store()

    # -----------------------------------------
    # Read PDF
    # -----------------------------------------

    def load_pdf(self):

        text = ""

        pdf = fitz.open(self.pdf_path)

        for page in pdf:
            text += page.get_text()

        pdf.close()

        self.text = text

    # -----------------------------------------
    # Split into chunks
    # -----------------------------------------

    def split_text(self):

        self.chunks = self.text_splitter.split_text(
            self.text
        )

    # -----------------------------------------
    # Create FAISS index
    # -----------------------------------------

    def create_vector_store(self):

        embeddings = self.embedding_model.encode(
            self.chunks,
            convert_to_numpy=True
        )

        embeddings = embeddings.astype("float32")

        dimension = embeddings.shape[1]

        self.index = faiss.IndexFlatL2(dimension)

        self.index.add(embeddings)

        self.embeddings = embeddings

    # -----------------------------------------
    # Retrieve similar chunks
    # -----------------------------------------

    def retrieve(
        self,
        query,
        top_k=3
    ):

        query_embedding = self.embedding_model.encode(
            [query],
            convert_to_numpy=True
        )

        query_embedding = query_embedding.astype(
            "float32"
        )

        distances, indices = self.index.search(
            query_embedding,
            top_k
        )

        retrieved_docs = []

        for idx in indices[0]:

            retrieved_docs.append(
                {
                    "text": self.chunks[idx]
                }
            )

        return retrieved_docs