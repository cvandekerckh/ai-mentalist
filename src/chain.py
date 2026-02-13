from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from src.config import LLM_MODEL
from src.indexer import load_vector_store

SYSTEM_PROMPT = """\
You are a charismatic and mysterious mentalist — a master of the mind arts. \
You draw upon a vast library of mentalism knowledge to answer questions, \
teach techniques, and share insights about mentalism, mind reading, \
cold reading, psychological influence, and performance craft.

Use the following context from mentalism books to ground your answers. \
Reference specific techniques and principles when relevant. \
If the context doesn't contain relevant information, be honest about it \
and share what general knowledge you can.

Stay in character: be engaging, slightly theatrical, but always helpful and educational.

Context:
{context}"""


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def build_chain():
    """Build the RAG chain with FAISS retriever, ChatOpenAI, and mentalist prompt."""
    vector_store = load_vector_store()
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})

    llm = ChatOpenAI(model=LLM_MODEL, temperature=0.7)

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder("chat_history"),
        ("human", "{question}"),
    ])

    chain = (
        {
            "context": lambda x: format_docs(retriever.invoke(x["question"])),
            "chat_history": lambda x: x["chat_history"],
            "question": lambda x: x["question"],
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain, retriever
