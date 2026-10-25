"""Interfaz web en Streamlit para el experto Souls.

Ejecutar en local:  streamlit run app.py
"""

import streamlit as st

from src.agent import ask, build_agent, new_thread_id
from src.config import get_vectorstore

st.set_page_config(page_title="La Guardiana del Fuego", page_icon="🔥")


@st.cache_resource(show_spinner="Encendiendo la hoguera...")
def load_agent():
    # Si la base vectorial no existe (p. ej. primer arranque), se indexa.
    if get_vectorstore()._collection.count() == 0:
        from src.ingest import ingest
        ingest()
    return build_agent()


def reset_conversation():
    st.session_state.thread_id = new_thread_id()
    st.session_state.history = []


if "thread_id" not in st.session_state:
    reset_conversation()

with st.sidebar:
    st.header("🔥 La Guardiana del Fuego")
    st.write(
        "Experta en **Dark Souls I-III**, **Bloodborne** y **Elden Ring**. "
        "Responde usando una base de conocimiento de wikis de Fandom y "
        "Wikipedia (RAG con Gemini + ChromaDB + LangGraph)."
    )
    st.button("Nueva conversación", on_click=reset_conversation, use_container_width=True)
    show_debug = st.toggle("Mostrar consulta de búsqueda", value=False)
    st.caption(f"Conversación: `{st.session_state.thread_id}`")
    st.caption("Contenido de las wikis bajo licencia CC BY-SA.")

st.title("Pregunta a la Guardiana del Fuego")

agent = load_agent()


def render_extras(turn: dict):
    if show_debug:
        st.caption(f"🔎 Búsqueda: *{turn['search_query']}* · juego: `{turn['game'] or 'todos'}`")
    with st.expander("Fuentes"):
        for title, url in turn["sources"]:
            st.markdown(f"- [{title}]({url})")


for turn in st.session_state.history:
    with st.chat_message("user"):
        st.markdown(turn["question"])
    with st.chat_message("assistant", avatar="🔥"):
        st.markdown(turn["answer"])
        render_extras(turn)

if question := st.chat_input("¿Qué quieres saber, Ashen One?"):
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant", avatar="🔥"):
        with st.spinner("Consultando los archivos..."):
            try:
                turn = {"question": question, **ask(agent, question, st.session_state.thread_id)}
            except Exception as exc:
                st.error(f"Error al consultar a Gemini: {exc}")
                st.stop()
        st.markdown(turn["answer"])
        render_extras(turn)
    st.session_state.history.append(turn)
