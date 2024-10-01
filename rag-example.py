import chromadb
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding

# Intialize Ollama for Mistral 7B
Settings.llm = Ollama(model="mistral", request_timeout=60)
ollama_embedding = OllamaEmbedding(model_name="mistral")
Settings.embed_model = ollama_embedding

# Initialize ChromaDB vector database and collection
db = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = db.get_or_create_collection("movies")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# Load data from file and create an index
documents = SimpleDirectoryReader("./data").load_data()
index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context
)

# Using query engine, generate a query for LLM about our data
query_engine = index.as_query_engine()
response = query_engine.query("What is nice movie in 2023?")
print(response)