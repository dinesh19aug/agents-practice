from datasets import load_dataset
from pathlib import Path
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.ollama import OllamaEmbedding
import nest_asyncio
from llama_index.llms.ollama import Ollama
from llama_index.core.query_engine import BaseQueryEngine
from llama_index.core.ingestion import IngestionPipeline
import asyncio
from llama_index.core.evaluation import FaithfulnessEvaluator

#### GETTING/SOURCING DATA ####
"""This method loads the 'finepersonas-v0.1-tiny' dataset and saves 
each persona to a separate text file in the 'data' directory."""
def get_fine_personas():
    """
    Load the 'finepersonas-v0.1-tiny' dataset and save each persona to a text file.
    """
    # Load the dataset
    dataset = load_dataset(path="dvilasuero/finepersonas-v0.1-tiny", split="train")

    Path("data").mkdir(parents=True, exist_ok=True)
    for i, persona in enumerate(dataset):
        with open(Path("data") / f"persona_{i}.txt", "w") as f:
            f.write(persona["persona"])

### LOADING DATA ###


def load_data():
    """
    Load data from the 'data' directory.
    """
    documents = SimpleDirectoryReader("data").load_data()
    print(f"Loaded {len(documents)} documents")
    return documents
    
async def create_vector_store(list_of_documents):
    """
    Create the documents into small node/chunks and Create a vector store using ChromaDB.
    """
    # Initialize ChromaDB client
    chroma_db = chromadb.PersistentClient(path="./chroma_db")

    # Create a collection
    collection = chroma_db.get_or_create_collection(name="fine_personas")

    # Load data and split into sentences
    
    splitter = SentenceSplitter(chunk_size=100, chunk_overlap=10)
    # Create a vector store
    vector_store = ChromaVectorStore(
        chroma_collection=collection
    )
    pipleine = IngestionPipeline( transformations= [splitter, 
                                                    OllamaEmbedding(model_name="qwen3")
                                                    ],
                                 vector_store=vector_store)

    print("\n=== Documents selected for indexing ===")
    for i, doc in enumerate(list_of_documents[:10]):
      print(f"[Document {i}]\n{doc.text[:300]}...\n")

    nodes = await pipleine.arun(documents=list_of_documents[:10])
    
    print(f"Vector store created successfully. Nodes Created: {len(nodes)}")
    return vector_store

def create_indexing(vectorStore: ChromaVectorStore):
    """
    Create an index using the vector store.
    """
    # Create an index from the vector store
    index = VectorStoreIndex.from_vector_store(embed_model=OllamaEmbedding(model_name="qwen3"), vector_store=vectorStore)

    print("Index created successfully")
    return index

def create_query_engine(index: VectorStoreIndex) ->BaseQueryEngine:
    """
    Create a RAG index using the vector store.
    """
    # Run the query engine
    nest_asyncio.apply()

    llm=Ollama(
    model="qwen3",
    temperature=0.7,
    max_tokens=100,
    request_timeout=120,
    provider="auto")

    query_engine = index.as_query_engine(llm=llm,response_mode="tree_summarize",
                                         similarity_top_k=3,
                                         similarity_top_k_filtering=True,
                                         verbose=True)
    return query_engine

def evaluate_query_engine(response, query:str):
    llm=Ollama(
    model="qwen3",
    temperature=0.7,
    max_tokens=100,
    request_timeout=120,
    provider="auto")
    
    evaluator = FaithfulnessEvaluator(llm=llm)
    eval_response = evaluator.evaluate(
        query=query,
        response=response.response,  # this is the LLM answer text
        contexts=[node.node.text for node in response.source_nodes]
    )

    print(f"Faithfulness Pass: {eval_response.passing}")
    print(f"Query: {eval_response.query}")
    print(f"Response: {eval_response.response}")
    print(f"score: {eval_response.score}")


async def main():
    list_of_documents = load_data()
    vector_store = await create_vector_store(list_of_documents)
    index = create_indexing(vector_store)
    query_engine = create_query_engine(index)
    query = "Respond using a persona of a a teacher?"
    response = query_engine.query(query)
    # Inspect the source nodes retrieved
    for i, source in enumerate(response.source_nodes):
      print(f"\n[Source {i}]")
      print(source.node.text)

    print(f"*" * 50)  
    print(response)
    print(f"*" * 50)  
    evaluate_query_engine(response, query)

if __name__ == "__main__":
    asyncio.run(main())