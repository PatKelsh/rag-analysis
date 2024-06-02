from data_ingest import DataIngest
from inference import LLMInference
import gradio as gr
from langchain_community.vectorstores import FAISS
from ragatouille import RAGPretrainedModel

from typing import Optional, Tuple, List

if __name__ == '__main__':
    DATASET = "../data/cal-itp/docs"
    EMBEDDING_MODEL_NAME = "thenlper/gte-small"
    READER_MODEL_NAME = "HuggingFaceH4/zephyr-7b-beta"
    RERANKER_MODEL = "colbert-ir/colbertv2.0"
    CHUNK_SIZE = 512

    ingest = DataIngest(EMBEDDING_MODEL_NAME)
    ingest.ingest_markdown_directory(DATASET)

    ingest.create_vector_store()
    llm = LLMInference(READER_MODEL_NAME)
    llm.set_rag_prompt()

    RERANKER = RAGPretrainedModel.from_pretrained(RERANKER_MODEL)

    def answer_with_rag(
        question: str,
        use_rerank: Optional[bool] = False,
        num_retrieved_docs: int = 30,
        num_docs_final: int = 5,
    ) -> Tuple[str, List[str]]:
        # Gather documents with retriever
        print("=> Retrieving documents...")
        relevant_docs = ingest.vector_store.similarity_search(query=question, k=num_retrieved_docs)
        doc_sources = [doc.metadata for doc in relevant_docs]
        relevant_docs_content = [doc.page_content for doc in relevant_docs]  # keep only the text

        # Optionally rerank results
        if use_rerank:
            print("=> Reranking documents...")
            ranked_docs = RERANKER.rerank(question, relevant_docs_content, k=num_docs_final)
            # doc_sources = [doc.metadata for doc in relevant_docs]
            ranked_sources = []
            for doc in ranked_docs:
                index = relevant_docs_content.index(doc["content"])
                ranked_sources.append(doc_sources[index])
            doc_sources = ranked_sources
            relevant_docs_content = [doc["content"] for doc in ranked_docs]


        relevant_docs_content = relevant_docs_content[:num_docs_final]

        # Build the final prompt
        context = "\nExtracted documents:\n"
        context += "".join([f"Document {str(i)}:::\n" + doc for i, doc in enumerate(relevant_docs_content)])

        final_prompt = llm.rag_prompt_template.format(question=question, context=context)

        # Generate an answer
        print("=> Generating answer...")
        answer = llm.llm_pipeline(final_prompt)[0]["generated_text"]

        sources = [doc_source['source'] for doc_source in doc_sources[:num_docs_final]]

        return answer, sources

    def sourced_response(message, history) -> str:
        answer, sources = answer_with_rag(message, use_rerank=True)
        response = f'{answer}\n'
        for source in sources:
            response = response + f'- {source}/n'
        return response

    demo = gr.ChatInterface(fn=sourced_response, examples=["hello", "how are you", "how do i handle geospatial data"], title="ModelMatch AI Demo")
    demo.launch()
