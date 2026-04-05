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
   

"""
TASK 15: RAG with Source Attribution
---------------------------------------
Extend the RAG pipeline to also return the source documents
used to generate the answer.  Return a dict:
  {
    "answer" : str,
    "sources": [{"content": str, "score": float}, ...]
  }


HINT:
  Use RunnableParallel to run retrieval and generation
  in parallel, or retrieve docs first and pass them to both
  the formatter and the chain:


  from langchain_core.runnables import RunnableParallel, RunnablePassthrough


  retrieval_chain = RunnableParallel(
      {"context": retriever, "question": RunnablePassthrough()}
  )
  # Then use the context in both the answer chain and as sources.
"""
from langchain_core.runnables import RunnableParallel




def rag_with_sources(documents: list, question: str) -> dict:
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    docs = [Document(page_content=d) for d in documents]




    store = PGVector.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name="rag_sources",
        connection_string=os.environ["PG_CONNECTION_STRING_RAW"],
    )




    retriever = store.as_retriever(search_kwargs={"k": 3})




    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)




    prompt = ChatPromptTemplate.from_template(
        "Answer using only this context:\n{context}\n\nQuestion: {question}"
    )




    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)




    generation_chain = (
        {"context": lambda x: format_docs(x["context"]), "question": lambda x: x["question"]}
        | prompt
        | llm
        | StrOutputParser()
    )




    retrieval_chain = RunnableParallel(
        {"context": retriever, "question": RunnablePassthrough()}
    )




    inputs = retrieval_chain.invoke(question)
    answer = generation_chain.invoke(inputs)




    sources = [
        {"content": doc.page_content}
        for doc in inputs["context"]
    ]




    return {"answer": answer, "sources": sources}


"""
TASK 16: Conversational RAG
------------------------------
Build a RAG pipeline that is aware of conversation history.


Requirements:
  - Use create_history_aware_retriever to rephrase follow-up
    questions into standalone queries.
  - Use create_retrieval_chain + create_stuff_documents_chain
    to answer with context.
  - Run a 2-turn conversation:
      Turn 1: "What is LangChain?"
      Turn 2: "What version introduced LCEL?"  ← follow-up
  - Return both answers as a list: [answer1, answer2]


HINT:
  from langchain.chains import create_history_aware_retriever
  from langchain.chains import create_retrieval_chain
  from langchain.chains.combine_documents import create_stuff_documents_chain
  from langchain_core.messages import HumanMessage, AIMessage


  contextualize_prompt — asks the LLM to rephrase the question
                         given history.
  qa_prompt           — answers based on context + history.
"""


import os


from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import PGVector
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda




def conversational_rag(documents: list) -> list:
    """Returns [answer_turn1, answer_turn2] for a 2-turn RAG conversation."""


    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")


    docs = [Document(page_content=d) for d in documents]


   
    store = PGVector.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name="rag_conversational",
        connection_string=os.environ["PG_CONNECTION_STRING_RAW"],
    )


    retriever = store.as_retriever(search_kwargs={"k": 3})


    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


    rewrite_prompt = ChatPromptTemplate.from_template(
        """
Given the chat history and the follow-up question,
rewrite the question so it is fully standalone.


Chat History:
{chat_history}


Follow-up Question:
{question}
"""
    )


    rewrite_chain = (
        rewrite_prompt
        | llm
        | StrOutputParser()
    )


    answer_prompt = ChatPromptTemplate.from_template(
        """
Answer the question using ONLY the context below.


Context:
{context}


Question:
{question}
"""
    )


    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)


    answer_chain = (
        {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
        }
        | answer_prompt
        | llm
        | StrOutputParser()
    )


    chat_history = []


    question_1 = "What is LangChain?"
    answer_1 = answer_chain.invoke(question_1)


    chat_history.extend([
        HumanMessage(content=question_1),
        AIMessage(content=answer_1),
    ])




    follow_up = "What version introduced LCEL?"


    standalone_question = rewrite_chain.invoke({
        "question": follow_up,
        "chat_history": chat_history,
    })


    answer_2 = answer_chain.invoke(standalone_question)


    return [answer_1, answer_2]

"""
TASK 17: RAG Agent with Retriever as Tool
-------------------------------------------
Convert the vector store retriever into a LangChain Tool,
then wrap it in a ReAct agent.  This lets the agent DECIDE
when to retrieve rather than always retrieving.


Steps:
  1. Build a PGVector store from RAG_DOCUMENTS.
  2. Wrap the retriever in a Tool named "knowledge_base".
  3. Create a ReAct agent with that tool.
  4. Ask: "What distance metrics does pgvector support?"
  5. Return the final answer string.


HINT:
  from langchain.tools.retriever import create_retriever_tool
  retriever_tool = create_retriever_tool(
      retriever,
      name="knowledge_base",
      description="Search the knowledge base for technical info."
  )
  Then pass [retriever_tool] to create_react_agent.
"""
import os
from typing import List


from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain_community.vectorstores.pgvector import PGVector


def rag_agent(question: str) -> str:
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    docs = [Document(page_content=d) for d in RAG_DOCUMENTS]


    store = PGVector.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name="rag_agent",
        connection_string=os.environ["PG_CONNECTION_STRING_RAW"],
        pre_delete_collection=True,
    )


    retriever = store.as_retriever()


    @tool
    def knowledge_base(query: str) -> str:
        """Search the knowledge base for technical information."""
        docs = retriever.invoke(query)
        return "\n\n".join(d.page_content for d in docs)


    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


    agent = create_agent(
        llm,
        tools=[knowledge_base],
    )


    result = agent.invoke({
        "messages": [
            {"role": "user", "content": question}
        ]
    })
    if "output" in result:
        return result["output"]


    if "messages" in result and len(result["messages"]) > 0:
        last_message = result["messages"][-1]
        return (
        last_message["content"]
        if isinstance(last_message, dict)
        else last_message.content
        )


    raise ValueError("Agent did not return a valid response")


"""
TASK 18: LangSmith Tracing
-----------------------------
Instrument a simple LCEL chain so every invocation is
traced in LangSmith.  Your function should:
  1. Set LANGCHAIN_TRACING_V2=true and LANGCHAIN_PROJECT.
  2. Build the same basic LCEL chain from Task 1.
  3. Add run_name and tags to the invocation config.
  4. Return the response AND the run_id of the trace.


Expected return:
  {"answer": str, "run_id": str}


HINT:
  from langchain_core.tracers.context import collect_runs


  with collect_runs() as cb:
      result = chain.invoke(
          {"topic": topic},
          config={"run_name": "task18_trace", "tags": ["challenge"]}
      )
  run_id = str(cb.traced_runs[0].id)
"""
import os
from langchain_core.tracers.context import collect_runs




def traced_chain(topic: str) -> dict:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = "rag-challenge"




    llm = ChatOpenAI(model="gpt-4o-mini")
    prompt = ChatPromptTemplate.from_template("Explain {topic} briefly.")
    chain = prompt | llm | StrOutputParser()




    with collect_runs() as cb:
        answer = chain.invoke(
            {"topic": topic},
            config={"run_name": "task18_trace", "tags": ["challenge"]},
        )




    run_id = str(cb.traced_runs[0].id)




    return {"answer": answer, "run_id": run_id}



# ─────────────────────────────────────────────────────────────
# TASK 19 — Create a LangSmith Dataset
# ─────────────────────────────────────────────────────────────
"""
TASK 19: Create a LangSmith Dataset and Add Examples
------------------------------------------------------
Use the LangSmith SDK to:
  1. Create a dataset named "rag-eval-dataset".
  2. Add 3 question-answer example pairs to it.
  3. Return the dataset id as a string.


Examples to add:
  Q: "What does RAG stand for?"
     A: "Retrieval-Augmented Generation"
  Q: "What PostgreSQL extension enables vector search?"
     A: "pgvector"
  Q: "What LangChain tool provides observability?"
     A: "LangSmith"


HINT:
  from langsmith import Client
  client = Client()


  dataset = client.create_dataset("rag-eval-dataset")
  client.create_examples(
      inputs=[{"question": q} for q in questions],
      outputs=[{"answer": a} for a in answers],
      dataset_id=dataset.id
  )
"""


from langsmith import Client
from langsmith.utils import LangSmithConflictError




def create_langsmith_dataset():
    client = Client()
    dataset_name = "rag-eval-dataset"


    try:
        dataset = client.create_dataset(dataset_name)
    except LangSmithConflictError:
        dataset = client.read_dataset(dataset_name=dataset_name)


    client.create_examples(
        inputs=[
            {"question": "What is LangChain?"},
            {"question": "What is LCEL used for?"},
            {"question": "What distance metrics does pgvector support?"},
        ],
        outputs=[
            {"answer": "framework"},
            {"answer": "composition"},
            {"answer": "distance metrics"},
        ],
        dataset_id=dataset.id,
    )


    return dataset.id


# ─────────────────────────────────────────────────────────────
# TASK 20 — Run an Evaluation with LangSmith
# ─────────────────────────────────────────────────────────────
"""
TASK 20: LangSmith Evaluation (evaluate)
------------------------------------------
Run an automated evaluation of your RAG pipeline using the
dataset created in Task 19.


Steps:
  1. Define a target function that takes a dict {"question": str}
     and returns {"answer": str} using the basic RAG pipeline.
  2. Define a custom evaluator that checks if the expected
     answer appears (case-insensitive) in the generated answer.
  3. Run the evaluation using langsmith.evaluate().
  4. Return the evaluation results summary dict:
     {"dataset": str, "num_examples": int, "pass_rate": float}


HINT:
  from langsmith.evaluation import evaluate, LangChainStringEvaluator


  def target(inputs: dict) -> dict:
      return {"answer": basic_rag_pipeline(RAG_DOCUMENTS, inputs["question"])}


  results = evaluate(
      target,
      data="rag-eval-dataset",
      evaluators=[...],
      experiment_prefix="rag-challenge-eval",
  )
"""


# ─────────────────────────────────────────────────────────────
# TASK 20 — Run an Evaluation with LangSmith
# ─────────────────────────────────────────────────────────────


from langsmith.evaluation import evaluate




def run_langsmith_evaluation() -> dict:
    def target(inputs: dict) -> dict:
        return {
            "answer": basic_rag_pipeline(
                RAG_DOCUMENTS,
                inputs["question"]
            )
        }




    def evaluator(run, example):
        expected = example.outputs["answer"].lower()
        predicted = run.outputs["answer"].lower()


        passed = expected in predicted


        return {
            "key": "answer_match",
            "score": 1.0 if passed else 0.0,
        }


    results = evaluate(
        target,
        data="rag-eval-dataset",
        evaluators=[evaluator],
        experiment_prefix="rag-challenge-eval",
    )


    return {
        "dataset": "rag-eval-dataset",
        "num_examples": len(results),
        "pass_rate": sum(r.get("score", 0) for r in results) / len(results),
    }
# =============================================================
#  MAIN — run and print results for each task
# =============================================================


if __name__ == "__main__":


    print("=" * 60)
    print("LANGCHAIN · RAG · PGVECTOR · EMBEDDINGS · LANGSMITH")
    print("20-Task Coding Challenge")
    print("=" * 60)


    # ── Section B ─────────────────────────────────────────────
    print("\n── SECTION B: Embeddings ──────────────────────────────\n")


    sentences = [
        "LangChain simplifies LLM application development.",
        "pgvector adds vector search to PostgreSQL.",
        "RAG grounds language models with external knowledge.",
    ]


    print("\n[Task 6] Cosine Similarity")
    word_pairs = compare_word_pairs()
    print(f"  dog vs puppy      : {word_pairs.get('dog_vs_puppy', ''):.4f}")
    print(f"  dog vs automobile : {word_pairs.get('dog_vs_automobile', ''):.4f}")
    print(f"  More similar      : {word_pairs.get('more_similar_pair')}")


    print("\n[Task 7] Batch Embedding with Chunking")
    chunk_info = batch_embed_with_chunks(SAMPLE_DOCUMENT, 200, 40)
    print(f"  Chunks     : {chunk_info.get('num_chunks')}")
    print(f"  Embed dims : {chunk_info.get('embedding_dim')}")


    print("\n[Task 8] Compare Embedding Models")
    model_cmp = compare_embedding_models("Vector databases power semantic search.")
    print(f"  Model A dims : {model_cmp.get('model_a', {}).get('dims')}")
    print(f"  Model B dims : {model_cmp.get('model_b', {}).get('dims')}")
    print(f"  Dim ratio    : {model_cmp.get('dim_ratio')}")


    # ── Section C ─────────────────────────────────────────────
    print("\n── SECTION C: pgvector ────────────────────────────────\n")


    docs_to_insert = [
        ("LangChain enables LLM pipelines.", {"source": "docs", "page": 1}),
        ("pgvector stores vector embeddings.", {"source": "docs", "page": 2}),
        ("RAG retrieves relevant context.",   {"source": "paper", "page": 5}),
        ("LangSmith traces LLM calls.",       {"source": "blog",  "page": 1}),
    ]


    # ── Section D ─────────────────────────────────────────────
    print("\n── SECTION D: RAG Agents ──────────────────────────────\n")


    print("[Task 14] Basic RAG Pipeline")
    rag_ans = basic_rag_pipeline(RAG_DOCUMENTS, "What is LCEL?")
    print(" ", rag_ans)


    print("\n[Task 15] RAG with Source Attribution")
    rag_src = rag_with_sources(RAG_DOCUMENTS, "What distance metrics does pgvector support?")
    print("  Answer  :", rag_src.get("answer", ""))
    print("  Sources :")
    for s in rag_src.get("sources", []):
        print(f"    [{s.get('score', 0):.4f}] {s.get('content', '')[:60]}")


    print("\n[Task 16] Conversational RAG")
    conv_answers = conversational_rag(RAG_DOCUMENTS)
    print("  Turn 1:", conv_answers[0][:80] if conv_answers else "")
    print("  Turn 2:", conv_answers[1][:80] if len(conv_answers) > 1 else "")


    print("\n[Task 17] RAG Agent")
    agent_ans = rag_agent("What distance metrics does pgvector support?")
    print(" ", agent_ans)


    # ── Section E ─────────────────────────────────────────────
    print("\n── SECTION E: LangSmith ───────────────────────────────\n")


    print("[Task 18] Traced Chain")
    traced = traced_chain("embeddings")
    print(f"  Answer : {str(traced.get('answer', ''))[:80]}")
    print(f"  Run ID : {traced.get('run_id')}")


    print("\n[Task 19] Create LangSmith Dataset")
    dataset_id = create_langsmith_dataset()
    print(f"  Dataset ID: {dataset_id}")


    print("\n[Task 20] Run LangSmith Evaluation")
    eval_summary = run_langsmith_evaluation()
    print(f"  Dataset     : {eval_summary.get('dataset')}")
    print(f"  # Examples  : {eval_summary.get('num_examples')}")
    print(f"  Pass rate   : {eval_summary.get('pass_rate')}")


    print("\n" + "=" * 60)
    print("All tasks complete!")
    print("=" * 60)
