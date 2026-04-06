
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
