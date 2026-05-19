from __future__ import annotations

from typing import Any, Callable, List, TypedDict

from langchain_core.documents import Document
from langgraph.graph import END, StateGraph


class AgentState(TypedDict, total=False):
    question: str
    documents: List[Document]
    answer: str
    needs_retrieval: bool


def _to_text(result: Any) -> str:
    """Normalize LLM outputs (AIMessage/string/dict) to plain text."""
    if result is None:
        return ""

    if hasattr(result, "content"):
        content = result.content
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = [item.get("text", "") for item in content if isinstance(item, dict)]
            return " ".join(part for part in parts if part).strip()

    if isinstance(result, dict) and "answer" in result:
        return str(result["answer"])

    return str(result)


def _format_documents(documents: List[Document]) -> str:
    if not documents:
        return ""

    formatted = []
    for i, doc in enumerate(documents, start=1):
        source = (doc.metadata or {}).get("source", "unknown")
        formatted.append(f"[{i}] ({source}) {doc.page_content}")
    return "\n\n".join(formatted)


def decide_retrieval(state: AgentState) -> AgentState:
    """
    Decide whether retrieval is needed for the current question.
    """
    question = state.get("question", "").strip().lower()

    if not question:
        return {**state, "needs_retrieval": False, "documents": []}

    smalltalk_keywords = {
        "hi",
        "hello",
        "hey",
        "thanks",
        "thank you",
        "good morning",
        "good evening",
    }
    retrieval_keywords = {
        "what",
        "how",
        "why",
        "when",
        "where",
        "explain",
        "describe",
        "tell me",
        "summarize",
        "compare",
    }

    is_smalltalk = question in smalltalk_keywords
    has_retrieval_intent = any(keyword in question for keyword in retrieval_keywords)

    # Slightly longer factual prompts usually benefit from context lookup.
    needs_retrieval = (not is_smalltalk) and (has_retrieval_intent or len(question.split()) > 6)

    return {**state, "needs_retrieval": needs_retrieval, "documents": state.get("documents", [])}


def make_retrieve_documents_node(retriever: Any) -> Callable[[AgentState], AgentState]:
    """
    Build a retrieval node with an injected retriever.

    The retriever must expose .invoke(question) -> List[Document] (or compatible output).
    """

    def retrieve_documents(state: AgentState) -> AgentState:
        question = state["question"]
        raw_docs = retriever.invoke(question)

        if raw_docs is None:
            documents: List[Document] = []
        elif isinstance(raw_docs, list):
            documents = raw_docs
        else:
            documents = [raw_docs]

        return {**state, "documents": documents}

    return retrieve_documents


def make_generate_answer_node(llm: Any) -> Callable[[AgentState], AgentState]:
    """
    Build an answer-generation node with an injected LLM.

    The LLM should support .invoke(messages_or_prompt).
    """

    def generate_answer(state: AgentState) -> AgentState:
        question = state["question"]
        documents = state.get("documents", [])

        if documents:
            context = _format_documents(documents)
            prompt = (
                "You are a helpful assistant answering from retrieved context. "
                "If the answer is not in the context, say so clearly.\n\n"
                f"Question:\n{question}\n\n"
                f"Context:\n{context}\n\n"
                "Answer:"
            )
        else:
            prompt = (
                "You are a helpful assistant. Answer the user's question directly.\n\n"
                f"Question:\n{question}\n\n"
                "Answer:"
            )

        result = llm.invoke(prompt)
        answer = _to_text(result).strip()

        return {**state, "answer": answer}

    return generate_answer


def _route_retrieval(state: AgentState) -> str:
    return "retrieve_documents" if state.get("needs_retrieval", False) else "generate_answer"


def build_agentic_rag_graph(retriever: Any, llm: Any):
    """
    Create and compile an agentic RAG graph.

    Flow:
    decide_retrieval -> (retrieve_documents | generate_answer) -> END
    """
    graph = StateGraph(AgentState)

    graph.add_node("decide_retrieval", decide_retrieval)
    graph.add_node("retrieve_documents", make_retrieve_documents_node(retriever))
    graph.add_node("generate_answer", make_generate_answer_node(llm))

    graph.set_entry_point("decide_retrieval")

    graph.add_conditional_edges(
        "decide_retrieval",
        _route_retrieval,
        {
            "retrieve_documents": "retrieve_documents",
            "generate_answer": "generate_answer",
        },
    )
    graph.add_edge("retrieve_documents", "generate_answer")
    graph.add_edge("generate_answer", END)

    return graph.compile()
