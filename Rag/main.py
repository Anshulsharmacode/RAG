import sys
from pathlib import Path

from Rag_search import RAG

# ensure project root is on sys.path so sibling package `constant` can be imported
# sys.path.append(str(Path(__file__).resolve().parents[1]))

from langchain_community.document_loaders import TextLoader , DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from VectorDB import VectorDB
from Embedding import NomicEmbeddingModel
from tqdm.auto import tqdm


emb = NomicEmbeddingModel()
db = VectorDB()
# load = TextLoader('./data/sample.txt', encoding='utf-8')

loader = DirectoryLoader(   
    "./medical",
    glob='**/*.txt',
    loader_cls=TextLoader,
    loader_kwargs={'encoding': 'utf-8'},
    show_progress=True   # <-- turn on loader progress
)

# load documents from the directory
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,     # adjust size as needed
    chunk_overlap=50    # adjust overlap as needed
)
chunks = splitter.split_documents(documents)

text = [doc.page_content for doc in chunks]

# generate embeddings with progress
embed = emb.generate_embeddings(text, show_progress=True)

# pass show_progress flag to your DB insertion if supported
save = db.add_Document(document=chunks, embeddings=embed)


rag = RAG(db=db, emb=emb)

rag.retrieve("What is diabetes?" , top_k=3 , score_threshold=0.1)

print("Done.")