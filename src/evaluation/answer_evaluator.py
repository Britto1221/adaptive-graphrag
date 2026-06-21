from pydantic import BaseModel , Field
import os
from typing import Literal
from dotenv import load_dotenv
load_dotenv()
from langchain_openai import ChatOpenAI
from src.pipelines.graph_rag import graph_pipeline
from src.pipelines.vector_rag import vector_pipeline
from src.pipelines.hybrid_rag import hybrid_pipeline

llm = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0,
)

class evaluater(BaseModel):
    correctness_score:int=Field(le=5,ge=1,description="1 to 5. 5 means the answer is factually correct compared to the expected answer.",)
    faithfulness_score:int=Field(le=5,ge=1,description="1 to 5. 5 means the answer is fully supported by the evidence and contains no unsupported by the evidence.",)
    evidence_score:int=Field(le=5,ge=1,description="1 to 5. 5 means the answer clearly uses strong evidence.",)
    refusal_score:int=Field(le=5,ge=1,description=(
        "Score from 1 to 5 for answerability handling. "
        "If the question is answerable and the model answers correctly, give 5. "
        "If the question is answerable but the model refuses, give 1. "
        "If the question is unanswerable and the model refuses correctly, give 5. "
        "If the question is unanswerable but the model invents an answer, give 1."
    ),)
    overall_score:int=Field(le=5,ge=1,description="1 to 5. Overall answer quality score from 1 to 5.",)
    evidence_used: Literal["vector", "graph", "both", "none"] = Field(description="Which evidence type the answer appears to use.",)
    reason: str = Field(description="Short explanation for the scores.",)

structured_llm = llm.with_structured_output(evaluater)

EVALUATION_PROMPT = """
You are a strict evaluator for a RAG benchmarking project.

You must evaluate the pipeline answer using:

1. The expected answer for benchmark correctness.
2. The actual pipeline evidence for faithfulness and evidence support.
3. The is_answerable label for refusal/answerability behavior.

Question:
{question}

Is answerable from the benchmark dataset:
{is_answerable}

Expected answer:
{expected_answer}

Pipeline answer:
{answer}

Actual pipeline evidence:
{evidence}

Scoring fields:

1. correctness_score:
   Does the pipeline answer match the expected answer and expected behavior?

* If is_answerable is True, the answer should correctly answer the question.
* If is_answerable is True and the answer matches the expected answer, give high correctness.
* If is_answerable is True and the model refuses even though the expected answer exists, give low correctness.
* If is_answerable is False, the correct behavior is to refuse or say the retrieved evidence/dataset does not contain enough information.
* If is_answerable is False and the model refuses correctly without inventing facts, give correctness_score = 5.
* If is_answerable is False but the model invents an unsupported answer, give low correctness.

2. faithfulness_score:
   Are all claims in the pipeline answer supported by the actual pipeline evidence?

* Use only the actual pipeline evidence, not outside knowledge.
* Do not trust the model's own evidence summary unless it matches the actual pipeline evidence.
* If the answer refuses because evidence is insufficient and does not invent facts, give high faithfulness.
* If the answer makes claims not supported by the actual evidence, reduce faithfulness.

3. evidence_score:
   Does the answer use the actual evidence correctly?

* If is_answerable is True, the answer should use relevant retrieved evidence to support the answer.
* If is_answerable is False, a correct refusal does not need to provide the missing fact. It should be rewarded if it correctly recognizes that the evidence does not contain the requested information.
* If the answer claims evidence that is not present, reduce evidence_score.

4. refusal_score:
   Evaluate answerability handling.

* If is_answerable is True and the model gives a correct answer, give refusal_score = 5.
* If is_answerable is True but the model refuses or says there is not enough information, give refusal_score = 1.
* If is_answerable is False and the model refuses correctly, give refusal_score = 5.
* If is_answerable is False but the model invents an answer, give refusal_score = 1.

5. overall_score:
   Overall benchmark quality from 1 to 5.

* If is_answerable is False and the answer correctly refuses without hallucination, overall_score should be 5.
* If is_answerable is True and the answer is correct, faithful, and evidence-supported, overall_score should be 5.
* If the answer is unsupported, wrong, hallucinatory, or incorrectly refuses, reduce overall_score.

Evidence used:
Set evidence_used to one of:

* "vector" if only vector evidence is present and used
* "graph" if only graph evidence is present and used
* "both" if both graph and vector evidence are present and used
* "none" if no actual evidence is present or the answer does not use evidence

Important rules:

* Do not reward outside knowledge.
* Actual pipeline evidence is more important than the model's own evidence summary.
* A correct refusal for an unanswerable question is a successful answer, not a failure.
* Do not punish an unanswerable answer for not giving information that is intentionally missing from the dataset.
* Return only valid JSON.

Return JSON with exactly these keys:
{{
  "correctness_score": int,
  "faithfulness_score": int,
  "evidence_score": int,
  "refusal_score": int,
  "overall_score": int,
  "evidence_used": "vector" | "graph" | "both" | "none",
  "reason": str
}}
"""


def evaluate_answer(question: str,expected_answer: str,answer: str,is_answerable:bool,evidence: str="")-> dict:
    prompt = EVALUATION_PROMPT.format(
        question=question,
        expected_answer=expected_answer,
        answer=answer,
        evidence=evidence,
        is_answerable=is_answerable,
    )
    try:
        result = structured_llm.invoke(prompt)
        return result.model_dump()
    except Exception as e:
        print(f"Evaluation failed: {e}")
        return None

if __name__ == "__main__":
    answer = evaluate_answer(
        "How is Elon Musk connected to Jeff Bezos, and what evidence explains their business competition?",
        "Elon Musk is connected to Jeff Bezos through a COMPETES_WITH relationship between their companies, SpaceX and Blue Origin. The evidence explains that SpaceX and Blue Origin compete in commercial launch services, reusable rockets, space infrastructure, government contracts, and long-term human-spaceflight ambitions.",
        "Elon Musk and Jeff Bezos are connected through a COMPETES_WITH relationship, as both lead companies that are in direct competition in the space industry. Specifically, Musk's SpaceX and Bezos's Blue Origin compete in areas such as commercial launch services, reusable rockets, space infrastructure, government contracts, and long-term human-spaceflight ambitions.",
        "Evidence used: vector Evidence summary: The vector context describes the competitive relationship between SpaceX and Blue Origin, highlighting their business competition in the space industry. VECTOR EVIDENCE: Document: data\raw\top_30_richest_people_paragraph_corpus.txt All chunks below were retrieved from this document. [Chunk 1] chunk_id: chunk_1 text: SpaceX, Blue Origin, Elon Musk, Jeff Bezos are connected through a COMPETES_WITH relationship. SpaceX and Blue Origin compete in commercial launch services, reusable rockets, space infrastructure, government contracts, and long-term human-spaceflight ambitions. The relationship applies to Ongoing and is identified in the corpus as REL-018. The source label for this relationship is Company business descriptions. [Chunk 2] chunk_id: chunk_2 text: Walmart, Amazon, Walton family, Jeff Bezos are connected through a COMPETES_WITH relationship. Walmart and Amazon compete in physical retail, e-commerce, advertising, logistics, online marketplaces, subscription programs, and delivery services. The relationship applies to Ongoing and is identified in the corpus as REL-017. The source label for this relationship is Company business descriptions. [Chunk 3] chunk_id: chunk_3 text: . His business network spans automobiles, energy, space transportation, satellite communications, social media, artificial intelligence, brain-computer interfaces, and infrastructure. The wider company network connected to Elon Musk includes several important relationships. Google and Fidelity invested about $1 billion in SpaceX in 2015, creating an investment relationship between Alphabet and SpaceX GRAPH EVIDENCE: none"
        )