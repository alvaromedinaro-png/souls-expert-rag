"""Agente RAG en LangGraph con memoria de conversación.

Grafo:  START -> rewrite_query -> retrieve -> generate -> END

- rewrite_query: convierte la última pregunta (con el historial) en una
  consulta autocontenida en inglés y detecta el juego para filtrar.
- retrieve: busca en ChromaDB en CADA turno.
- generate: Gemini responde con el system prompt, el contexto y el historial.

La memoria la aporta el checkpointer de LangGraph: cada conversación es un
thread_id y su estado (mensajes incluidos) se guarda entre llamadas.
"""

import uuid
from typing import Literal, Optional

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage, trim_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from pydantic import BaseModel, Field

from src.config import REWRITE_MODEL, get_llm, get_vectorstore
from src.prompts import CONTEXT_TEMPLATE, REWRITE_PROMPT, SYSTEM_PROMPT

TOP_K = 6
MAX_HISTORY_MESSAGES = 12  # últimos mensajes que se envían al LLM


class SearchQuery(BaseModel):
    """Consulta de búsqueda generada a partir de la pregunta del usuario."""
    query: str = Field(description="Consulta autocontenida en inglés")
    game: Optional[Literal["dark_souls", "bloodborne", "elden_ring"]] = Field(
        default=None, description="Juego al que se refiere, o null")


class AgentState(MessagesState):
    search_query: str
    game: Optional[str]
    context: list[Document]


def recent_history(messages):
    """Recorta el historial para no exceder el contexto en charlas largas."""
    return trim_messages(messages, max_tokens=MAX_HISTORY_MESSAGES,
                         token_counter=len, start_on="human")


def format_context(docs: list[Document]) -> str:
    return "\n\n".join(
        f"[{i}] ({d.metadata['title']} | {d.metadata['game']})\n{d.page_content}"
        for i, d in enumerate(docs, 1)
    )


def build_agent(checkpointer=None):
    """Construye y compila el grafo. Devuelve el agente listo para invoke()."""
    llm = get_llm()
    rewriter = get_llm(temperature=None, model=REWRITE_MODEL).with_structured_output(SearchQuery)
    vectorstore = get_vectorstore()

    def rewrite_query(state: AgentState) -> dict:
        result = rewriter.invoke(
            [SystemMessage(REWRITE_PROMPT), *recent_history(state["messages"])]
        )
        return {"search_query": result.query, "game": result.game}

    def retrieve(state: AgentState) -> dict:
        game = state.get("game")
        # Los artículos "general" (FromSoftware, Demon's Souls...) se incluyen siempre.
        where = {"game": {"$in": [game, "general"]}} if game else None
        docs = vectorstore.similarity_search(state["search_query"], k=TOP_K, filter=where)
        return {"context": docs}

    def generate(state: AgentState) -> dict:
        context = CONTEXT_TEMPLATE.format(context=format_context(state["context"]))
        response = llm.invoke([
            SystemMessage(SYSTEM_PROMPT),
            SystemMessage(context),
            *recent_history(state["messages"]),
        ])
        return {"messages": [response]}

    graph = StateGraph(AgentState)
    graph.add_node("rewrite_query", rewrite_query)
    graph.add_node("retrieve", retrieve)
    graph.add_node("generate", generate)
    graph.add_edge(START, "rewrite_query")
    graph.add_edge("rewrite_query", "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)
    return graph.compile(checkpointer=checkpointer or MemorySaver())


def new_thread_id() -> str:
    return uuid.uuid4().hex[:8]


def ask(agent, question: str, thread_id: str) -> dict:
    """Envía una pregunta en un hilo y devuelve respuesta, consulta y fuentes."""
    state = agent.invoke(
        {"messages": [HumanMessage(question)]},
        config={"configurable": {"thread_id": thread_id}},
    )
    sources = list(dict.fromkeys(
        (d.metadata["title"], d.metadata["url"]) for d in state["context"]
    ))
    return {
        "answer": state["messages"][-1].text,
        "search_query": state["search_query"],
        "game": state["game"],
        "sources": sources,
    }


def chat(agent, thread_id: Optional[str] = None) -> None:
    """Bucle de chat por texto para el notebook (escribe 'salir' para terminar)."""
    thread_id = thread_id or new_thread_id()
    print(f"Conversación {thread_id}. Escribe 'salir' para terminar.\n")
    while True:
        question = input("Tú: ").strip()
        if question.lower() in {"salir", "exit", "quit", ""}:
            print("La llama se apaga... ¡hasta la próxima, Ashen One!")
            break
        result = ask(agent, question, thread_id)
        print(f"\nGuardiana: {result['answer']}\n")
        print("Fuentes: " + ", ".join(t for t, _ in result["sources"]) + "\n")
