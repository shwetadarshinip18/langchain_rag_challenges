"""
=============================================================
  PYTHON CODING CHALLENGE
  Topic   : LangChain v1 · RAG Agents · pgvector ·
            Embeddings · LangSmith
  Level   : Intermediate
  Tasks   : 20  (project-style, grouped by topic)
=============================================================


SETUP — install dependencies before you begin
----------------------------------------------
  pip install langchain langchain-openai langchain-community
              langchain-core langsmith psycopg2-binary numpy
              python-dotenv


ENVIRONMENT VARIABLES — create a .env file or export these:
  OPENAI_API_KEY       = "sk-..."
  LANGCHAIN_API_KEY    = "ls__..."        # LangSmith
  LANGCHAIN_TRACING_V2 = "true"
  LANGCHAIN_PROJECT    = "rag-challenge"
  PG_CONNECTION_STRING = "postgresql+psycopg2://user:pass@localhost:5432/vectordb"


TOPIC SECTIONS
--------------
  Section A — LangChain Core         (Tasks  1 – 4)
  Section B — Embeddings             (Tasks  5 – 8)
  Section C — pgvector               (Tasks  9 – 13)
  Section D — RAG Agents             (Tasks 14 – 17)
  Section E — LangSmith              (Tasks 18 – 20)


RULES
-----
  - Implement every function stub below.
  - Do NOT add extra libraries beyond those listed in Setup.
  - Keep function signatures exactly as given.
  - For tasks that call an LLM, handle API errors gracefully
    with try/except.
=============================================================
"""


import os
import numpy as np
from dotenv import load_dotenv


load_dotenv()


# ─────────────────────────────────────────────────────────────
# TASK 6 — Cosine Similarity (from scratch, then with numpy)
# ─────────────────────────────────────────────────────────────
"""
TASK 6: Cosine Similarity
---------------------------
Part A: Implement cosine_similarity_manual(v1, v2) WITHOUT
        using numpy.  Use only Python loops / math.
Part B: Implement cosine_similarity_numpy(v1, v2) using numpy.


Both should return a float between -1 and 1.


Then embed these two pairs and print which pair is more similar:
  Pair 1: "dog" vs "puppy"
  Pair 2: "dog" vs "automobile"


Formula:
  cosine_similarity = (v1 · v2) / (||v1|| × ||v2||)


HINT:
  dot product: sum(a*b for a, b in zip(v1, v2))
  magnitude  : sum(x**2 for x in v) ** 0.5
  numpy equiv: np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
"""


import math




def cosine_similarity_manual(v1: list, v2: list) -> float:
    """Computes cosine similarity using pure Python."""
    dot_product = sum(a * b for a, b in zip(v1, v2))
    magnitude_v1 = math.sqrt(sum(x * x for x in v1))
    magnitude_v2 = math.sqrt(sum(x * x for x in v2))




    if magnitude_v1 == 0 or magnitude_v2 == 0:
        return 0.0




    return dot_product / (magnitude_v1 * magnitude_v2)




import numpy as np




def cosine_similarity_numpy(v1: list, v2: list) -> float:
    """Computes cosine similarity using numpy."""
    v1 = np.array(v1)
    v2 = np.array(v2)




    return float(
        np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    )






from langchain_openai import OpenAIEmbeddings




def compare_word_pairs() -> dict:
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")




    dog = embeddings.embed_query("dog")
    puppy = embeddings.embed_query("puppy")
    automobile = embeddings.embed_query("automobile")




    sim_dog_puppy = cosine_similarity_numpy(dog, puppy)
    sim_dog_auto = cosine_similarity_numpy(dog, automobile)




    return {
        "dog_vs_puppy": sim_dog_puppy,
        "dog_vs_automobile": sim_dog_auto,
        "more_similar_pair": (
            "dog vs puppy"
            if sim_dog_puppy > sim_dog_auto
            else "dog vs automobile"
        ),
    }


# ─────────────────────────────────────────────────────────────
# TASK 7 — Batch Embedding with Chunking
# ─────────────────────────────────────────────────────────────
"""
TASK 7: Batch Embedding with Chunking
----------------------------------------
Given a long text document, split it into overlapping chunks
using RecursiveCharacterTextSplitter, then embed all chunks
in a single batch call.  Return:
  {
    "num_chunks"   : int,
    "chunk_size"   : int,   # configured chunk size
    "overlap"      : int,   # configured overlap
    "embedding_dim": int,
    "chunks"       : list[str]
  }


Use chunk_size=200, chunk_overlap=40.


HINT:
  from langchain.text_splitter import RecursiveCharacterTextSplitter
  splitter = RecursiveCharacterTextSplitter(
      chunk_size=200, chunk_overlap=40
  )
  chunks = splitter.split_text(long_text)
  vectors = embeddings.embed_documents(chunks)
"""


SAMPLE_DOCUMENT = """
LangChain is a framework for developing applications powered by language models.
It provides tools for prompt management, chains, agents, and memory.
LangChain integrates with many LLM providers including OpenAI, Anthropic, and Cohere.
The framework also supports vector stores, document loaders, and output parsers.
RAG (Retrieval-Augmented Generation) is a technique that enhances LLM responses
by fetching relevant documents from a knowledge base at query time.
pgvector is a PostgreSQL extension that enables efficient storage and similarity
search of high-dimensional vector embeddings directly inside a relational database.
LangSmith is an observability platform for LangChain applications that provides
tracing, evaluation, and debugging of LLM pipelines.
"""






from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings




def batch_embed_with_chunks(
    text: str,
    chunk_size: int,
    overlap: int
) -> dict:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap
    )




    chunks = splitter.split_text(text)




    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectors = embeddings.embed_documents(chunks)




    return {
        "num_chunks": len(chunks),
        "chunk_size": chunk_size,
        "overlap": overlap,
        "embedding_dim": len(vectors[0]),
        "chunks": chunks
    }






# ─────────────────────────────────────────────────────────────
# TASK 8 — Compare Two Embedding Models
# ─────────────────────────────────────────────────────────────
"""
TASK 8: Compare Two Embedding Models
--------------------------------------
Embed the same sentence using two different OpenAI models:
  Model A: text-embedding-3-small   (1536 dims)
  Model B: text-embedding-3-large   (3072 dims)


For the sentence:  "Vector databases power semantic search."


Return a dict:
  {
    "sentence"   : str,
    "model_a"    : {"model": str, "dims": int, "first_3": list[float]},
    "model_b"    : {"model": str, "dims": int, "first_3": list[float]},
    "dim_ratio"  : float   # model_b_dims / model_a_dims
  }


HINT:
  OpenAIEmbeddings(model="text-embedding-3-small")
  OpenAIEmbeddings(model="text-embedding-3-large")
  embeddings.embed_query(sentence) → single vector (list of floats)
"""


from langchain_openai import OpenAIEmbeddings




def compare_embedding_models(sentence: str) -> dict:
    """Embeds a sentence with two models and compares their dimensions."""
    model_a = OpenAIEmbeddings(model="text-embedding-3-small")
    model_b = OpenAIEmbeddings(model="text-embedding-3-large")




    vec_a = model_a.embed_query(sentence)
    vec_b = model_b.embed_query(sentence)




    return {
        "sentence": sentence,
        "model_a": {
            "model": "text-embedding-3-small",
            "dims": len(vec_a),
            "first_3": vec_a[:3],
        },
        "model_b": {
            "model": "text-embedding-3-large",
            "dims": len(vec_b),
            "first_3": vec_b[:3],
        },
        "dim_ratio": len(vec_b) / len(vec_a),
    }


"""
TASK 14: Basic RAG Pipeline
------------------------------
Build an end-to-end RAG chain that:
  1. Loads documents from a list of strings.
  2. Stores them in a PGVector vectorstore.
  3. Creates a retriever (top-3 results).
  4. Passes retrieved context + question to ChatOpenAI.
  5. Returns the final answer string.


Use the LCEL pattern:
  chain = (
      {"context": retriever | format_docs, "question": RunnablePassthrough()}
      | prompt
      | llm
      | StrOutputParser()
  )


HINT:
  def format_docs(docs):
      return "\n\n".join(doc.page_content for doc in docs)


  prompt = ChatPromptTemplate.from_template(
      "Answer using only this context:\n{context}\n\nQuestion: {question}"
  )
"""


RAG_DOCUMENTS = [
    "LangChain v0.2 introduced LangChain Expression Language (LCEL) for composing chains.",
    "pgvector is a PostgreSQL extension supporting L2, inner product, and cosine distance.",
    "LangSmith provides tracing for every LLM call including token counts and latency.",
    "RAG stands for Retrieval-Augmented Generation and improves factual accuracy of LLMs.",
    "OpenAI's text-embedding-3-small produces 1536-dimensional embedding vectors.",
    "LangChain agents use a ReAct loop: Thought → Action → Observation → Answer.",
]


from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import PGVector
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
def basic_rag_pipeline(documents: list, question: str) -> str:
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")




    docs = [Document(page_content=d) for d in documents]




    store = PGVector.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name="rag_basic",
        connection_string=os.environ["PG_CONNECTION_STRING_RAW"],
        pre_delete_collection=True
    )




    retriever = store.as_retriever(search_kwargs={"k": 3})




    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)




    prompt = ChatPromptTemplate.from_template(
        "Answer using only this context:\n{context}\n\nQuestion: {question}"
    )




    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)




    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )




    return chain.invoke(question)
   
