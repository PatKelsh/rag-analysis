from typing import List
import os
from pathlib import Path

from transformers import AutoTokenizer
from langchain.docstore.document import Document as LangchainDocument
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy

class DataIngest():

    MARKDOWN_SEPARATORS = [
        "\n#{1,6} ",
        "```\n",
        "\n\\*\\*\\*+\n",
        "\n---+\n",
        "\n___+\n",
        "\n\n",
        "\n",
        " ",
        "",
    ]

    def __init__(self, model: str, chunk_size: int = 512):
        self.model = model
        self.chunk_size = chunk_size

    def _split_documents(self, knowledge_base: List[LangchainDocument]):
        """
        Split documents into chunks of maximum size `chunk_size` tokens and return a list of documents.
        """
        print("splitting documents")
        text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
            AutoTokenizer.from_pretrained(self.model),
            chunk_size=self.chunk_size,
            chunk_overlap=int(self.chunk_size / 10),
            add_start_index=True,
            strip_whitespace=True,
            separators=self.MARKDOWN_SEPARATORS,
        )

        docs_processed = []
        for doc in knowledge_base:
            docs_processed += text_splitter.split_documents([doc])

        # Remove duplicates
        unique_texts = {}
        docs_processed_unique = []
        for doc in docs_processed:
            if doc.page_content not in unique_texts:
                unique_texts[doc.page_content] = True
                docs_processed_unique.append(doc)

        self.docs_processed = docs_processed_unique
        return docs_processed_unique

    def ingest_markdown_directory(self, rootdir: str) -> List[LangchainDocument]:
        print("ingesting documents")
        files_dict = []
        for subdir, dirs, files in os.walk(rootdir):
            for file in files:
                filepath = subdir + os.sep + file

                if filepath.endswith(".md" or "mdx"):
                    files_dict.append({
                        "text": Path(filepath).read_text(),
                        "source" : filepath
                    })
        knowledge_base = [
            LangchainDocument(page_content=doc["text"], metadata={"source": doc["source"]}) for doc in files_dict
        ]

        return self._split_documents(knowledge_base)
    
    def create_vector_store(self) -> FAISS:
        print("embedding documents in vector store")
        embedding_model = HuggingFaceEmbeddings(
            model_name=self.model,
            multi_process=True,
            model_kwargs={"device": "cuda"},
            encode_kwargs={"normalize_embeddings": True},  # set True for cosine similarity
        )

        self.vector_store = FAISS.from_documents(
            self.docs_processed, embedding_model, distance_strategy=DistanceStrategy.COSINE
        )