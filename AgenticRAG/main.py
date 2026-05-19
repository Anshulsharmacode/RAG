from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

from funtion import build_agentic_rag_graph
from llm import get_chat_model, get_embedding_model


def build_retriever():
    text_path = "text.txt"
    loader = TextLoader(str(text_path), encoding="utf-8")
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    embeddings = get_embedding_model(model="nomic-embed-text:latest")
    vectordb = Chroma.from_documents(chunks, embedding=embeddings)
    return vectordb.as_retriever(search_kwargs={"k": 4})


def main():
    retriever = build_retriever()
    llm = get_chat_model(model="llama3.2:1b", temperature=0)

    app = build_agentic_rag_graph(retriever=retriever, llm=llm)

    question = "Explain what this text is mainly about."
    result = app.invoke({"question": question})

    print("Question:", question)
    print("Needs retrieval:", result.get("needs_retrieval"))
    print("Answer:\n", result.get("answer", ""))


if __name__ == "__main__":
    main()
