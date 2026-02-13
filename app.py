import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

from src.chain import build_chain

WELCOME_MESSAGE = (
    "Welcome, seeker of mysteries. I am the Mentalist — "
    "a guide to the hidden arts of the mind. "
    "Ask me about mentalism techniques, cold reading, "
    "psychological influence, or any secret from the pages of the masters. "
    "What would you like to explore?"
)

st.set_page_config(page_title="AI Mentalist", page_icon="🔮")
st.title("🔮 AI Mentalist")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chain" not in st.session_state:
    with st.spinner("Awakening the mind arts..."):
        chain, retriever = build_chain()
        st.session_state.chain = chain
        st.session_state.retriever = retriever

# Display welcome message
if not st.session_state.messages:
    with st.chat_message("assistant"):
        st.markdown(WELCOME_MESSAGE)

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("📚 Sources"):
                for source in msg["sources"]:
                    st.markdown(f"**{source['name']}**")
                    st.caption(source["excerpt"])

# Chat input
if question := st.chat_input("Ask the Mentalist..."):
    # Show user message
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Build chat history for LangChain
    chat_history = []
    for msg in st.session_state.messages[:-1]:  # exclude current question
        if msg["role"] == "user":
            chat_history.append(HumanMessage(content=msg["content"]))
        else:
            chat_history.append(AIMessage(content=msg["content"]))
    # Keep last 10 exchanges (20 messages)
    chat_history = chat_history[-20:]

    # Get response
    with st.chat_message("assistant"):
        with st.spinner("Reading your mind..."):
            # Retrieve source docs
            source_docs = st.session_state.retriever.invoke(question)
            sources = [
                {
                    "name": doc.metadata.get("source", "Unknown"),
                    "excerpt": doc.page_content[:200] + "...",
                }
                for doc in source_docs
            ]

            response = st.session_state.chain.invoke(
                {"question": question, "chat_history": chat_history}
            )

        st.markdown(response)
        if sources:
            with st.expander("📚 Sources"):
                for source in sources:
                    st.markdown(f"**{source['name']}**")
                    st.caption(source["excerpt"])

    st.session_state.messages.append(
        {"role": "assistant", "content": response, "sources": sources}
    )
