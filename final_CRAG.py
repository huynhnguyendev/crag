# ============================================================
# FINAL CRAG - FULL VERSION
# ============================================================
# Architecture:
#
# Retrieve
#    ↓
# Evaluate each document
#    ↓
# ┌───────────────┬──────────────────────────────┐
# │ CORRECT       │ INCORRECT / AMBIGUOUS        │
# │      ↓        │             ↓                │
# │    Refine     │       Rewrite Query           │
# │      ↓        │             ↓                │
# │              │        Tavily Web Search       │
# │              │             ↓                │
# │              │           Refine              │
# └───────────────┴──────────────┬───────────────┘
#                                ↓
#                             Generate
#
# LLM:
# - Gemini: document evaluator + final answer
# - Groq: sentence filter + query rewriting
# - Tavily: web search
# - Gemini Embedding + FAISS: vector retrieval
# ============================================================


# ============================================================
# 1. IMPORTS + ENVIRONMENT
# ============================================================

import os
import re
import json
from pathlib import Path
from typing import List, TypedDict

from dotenv import load_dotenv, find_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.tools.tavily_search import TavilySearchResults

from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)

from langchain_groq import ChatGroq

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from langgraph.graph import StateGraph, START, END

from pydantic import BaseModel, Field


# ------------------------------------------------------------
# Load .env
# ------------------------------------------------------------
# find_dotenv() giúp tìm file .env từ thư mục hiện tại
# và tìm ngược lên các thư mục cha.
#
# override=True:
# Nếu biến môi trường cũ đã tồn tại thì lấy giá trị mới
# trong .env ghi đè lên.
# ------------------------------------------------------------

env_path = find_dotenv(usecwd=True)

if env_path:
    load_dotenv(env_path, override=True)
    print("Loaded .env from:", env_path)
else:
    # Fallback: thử .env ở thư mục hiện tại
    load_dotenv(".env", override=True)
    print("Warning: .env was not found by find_dotenv().")


# ------------------------------------------------------------
# Kiểm tra API keys
# ------------------------------------------------------------

groq_key = os.getenv("GROQ_API_KEY")
google_key = os.getenv("GOOGLE_API_KEY")
tavily_key = os.getenv("TAVILY_API_KEY")

print("\n=== API KEY CHECK ===")
print("GROQ_API_KEY exists :", groq_key is not None)
print("GOOGLE_API_KEY exists:", google_key is not None)
print("TAVILY_API_KEY exists:", tavily_key is not None)

if groq_key:
    print("GROQ prefix:", groq_key[:8])
else:
    print("GROQ prefix: None")


# ============================================================
# 2. LOAD PDF
# ============================================================

pdf_path = "./documents/CV_HuynhTrungNguyen_AI_Engineer.pdf"

docs = PyPDFLoader(pdf_path).load()

print("\nPages in document:", len(docs))


# ============================================================
# 3. CHUNK DOCUMENT
# ============================================================

chunks = RecursiveCharacterTextSplitter(
    chunk_size=900,
    chunk_overlap=150,
).split_documents(docs)

print("Chunks:", len(chunks))


# ============================================================
# 4. EMBEDDING + FAISS
# ============================================================

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
)

vector_store = FAISS.from_documents(
    chunks,
    embeddings,
)

retriever = vector_store.as_retriever(
    search_kwargs={"k": 3}
)


# ============================================================
# 5. THRESHOLDS
# ============================================================

UPPER_TH = 0.7
LOWER_TH = 0.3


# ============================================================
# 6. STATE
# ============================================================

class State(TypedDict):
    # -------------------------
    # Retrieval
    # -------------------------
    question: str
    docs: List[Document]

    # -------------------------
    # Document Evaluation
    # -------------------------
    good_docs: List[Document]
    verdict: str
    reason: str

    # -------------------------
    # Context Refinement
    # -------------------------
    strips: List[str]
    kept_strips: List[str]
    refined_context: str

    # -------------------------
    # Web Search
    # -------------------------
    web_docs: List[Document]
    web_query: str

    # -------------------------
    # Final Answer
    # -------------------------
    answer: str


# ============================================================
# 7. RETRIEVAL NODE
# ============================================================

def retrieve_node(state: State) -> State:

    # Lấy câu hỏi của user
    query = state["question"]

    # Search trong FAISS
    output = retriever.invoke(query)

    return {
        "docs": output
    }


# ============================================================
# 8. DOCUMENT EVALUATOR
# ============================================================

class DocEvalScore(BaseModel):

    # Score nằm trong khoảng 0 -> 1
    score: float = Field(
        ge=0.0,
        le=1.0,
    )

    # Lý do evaluator đưa ra score
    reason: str


doc_eval_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a strict retrieval evaluator for RAG.\n"
            "You will be given ONE retrieved chunk and a question.\n"
            "Return a relevance score in [0.0, 1.0].\n"
            "- 1.0: chunk alone is sufficient to answer fully/mostly\n"
            "- 0.0: chunk is irrelevant\n"
            "Be conservative with high scores.\n"
            "Also return a short reason.\n"
            "Output JSON only.",
        ),
        (
            "human",
            "Question: {question}\n\n"
            "Chunk:\n{chunk}",
        ),
    ]
)


# Gemini dùng cho document evaluator
llm_eval = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    temperature=0,
)


doc_eval_chain = (
    doc_eval_prompt
    | llm_eval.with_structured_output(DocEvalScore)
)


# ------------------------------------------------------------
# Document Evaluation Node
# ------------------------------------------------------------

def eval_each_doc_node(state: State) -> State:

    q = state["question"]

    scores: List[float] = []
    reasons: List[str] = []
    good: List[Document] = []

    # Đánh giá từng retrieved document
    for d in state["docs"]:

        out = doc_eval_chain.invoke(
            {
                "question": q,
                "chunk": d.page_content,
            }
        )

        scores.append(out.score)
        reasons.append(out.reason)

        # Document đủ liên quan
        if out.score > LOWER_TH:
            good.append(d)

    # --------------------------------------------------------
    # CORRECT
    # --------------------------------------------------------
    if any(s > UPPER_TH for s in scores):

        return {
            "good_docs": good,
            "verdict": "CORRECT",
            "reason": (
                f"At least one retrieved chunk "
                f"scored > {UPPER_TH}."
            ),
        }

    # --------------------------------------------------------
    # INCORRECT
    # --------------------------------------------------------
    if len(scores) > 0 and all(s < LOWER_TH for s in scores):

        return {
            "good_docs": [],
            "verdict": "INCORRECT",
            "reason": (
                f"All retrieved chunks scored < {LOWER_TH}. "
                "No chunk was sufficient."
            ),
        }

    # --------------------------------------------------------
    # AMBIGUOUS
    # --------------------------------------------------------

    return {
        "good_docs": good,
        "verdict": "AMBIGUOUS",
        "reason": (
            f"No chunk scored > {UPPER_TH}, "
            f"but not all were < {LOWER_TH}. "
            "Mixed relevance signals."
        ),
    }


# ============================================================
# 9. SENTENCE DECOMPOSER
# ============================================================

def decompose_to_sentences(text: str) -> List[str]:

    # Chuẩn hóa khoảng trắng
    text = re.sub(r"\s+", " ", text).strip()

    # Tách câu dựa trên ., !, ?
    sentences = re.split(
        r"(?<=[.!?])\s+",
        text,
    )

    result = []

    for sentence in sentences:

        sentence = sentence.strip()

        # Bỏ những sentence quá ngắn
        if len(sentence) > 20:
            result.append(sentence)

    return result


# ============================================================
# 10. SENTENCE FILTER
# ============================================================

class KeepOrDrop(BaseModel):

    keep: bool


filter_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a strict relevance filter.\n"
            "Return keep=true only if the sentence directly "
            "helps answer the question.\n"
            "Use ONLY the sentence. Output JSON only.",
        ),
        (
            "human",
            "Question: {question}\n\n"
            "Sentence:\n{sentence}",
        ),
    ]
)


# ------------------------------------------------------------
# QUAN TRỌNG:
# Explicitly truyền API key vào ChatGroq.
#
# File test riêng của bạn chạy được, nên key này phải hợp lệ.
# Việc truyền trực tiếp giúp loại bỏ khả năng ChatGroq trong
# notebook chính đang lấy một environment variable khác.
# ------------------------------------------------------------

if not groq_key:
    raise ValueError(
        "GROQ_API_KEY is missing. "
        "Please check your .env file."
    )


llm_filter = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
    api_key=groq_key,
)


filter_chain = (
    filter_prompt
    | llm_filter.with_structured_output(KeepOrDrop)
)


# ============================================================
# 11. REWRITE QUERY
# ============================================================

class WebQuery(BaseModel):
    query: str

rewrite_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        Bạn là query rewriting agent.

        Hãy viết lại câu hỏi thành một query phù hợp để tìm kiếm trên web.

        Chỉ trả về JSON theo format:

        {{
            "query": "your rewritten query"
        }}

        Không thêm markdown hay giải thích.
        """
    ),
    (
        "human",
        "Câu hỏi của người dùng: {question}"
    )
])


# ------------------------------------------------------------
# Tạo llm_rewrite bằng CHÍNH GROQ_API_KEY đã kiểm tra ở trên.
# ------------------------------------------------------------

llm_rewrite = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
    api_key=groq_key,
)


rewrite_chain = rewrite_prompt | llm_rewrite


# ============================================================
# 12. TEST REWRITE MODEL TRƯỚC KHI BUILD GRAPH
# ============================================================
#
# Đây là phần mình thêm để debug lỗi 401.
#
# Nếu test này chạy được thì llm_rewrite trong file chính
# chắc chắn có thể gọi Groq.
# ============================================================

print("\n=== TEST GROQ REWRITE ===")

test_rewrite = llm_rewrite.invoke(
    "What is Retrieval Augmented Generation?"
)

print("Groq test:", test_rewrite.content)

print("=== GROQ REWRITE TEST PASSED ===")


# ============================================================
# 13. REWRITE QUERY NODE
# ============================================================

def rewrite_query_node(state: State) -> State:

    response = rewrite_chain.invoke(
        {
            "question": state["question"]
        }
    )

    # Lấy text từ Groq
    content = response.content

    # Parse JSON
    data = json.loads(content)

    # Validate bằng Pydantic
    result = WebQuery.model_validate(data)

    return {
        "web_query": result.query
    }


# ============================================================
# 14. WEB SEARCH NODE
# ============================================================

tavily = TavilySearchResults(
    max_results=5
)


def web_search_node(state: State) -> State:

    # Ưu tiên query đã được rewrite
    # Nếu rewrite không có thì fallback về question gốc
    q = (
        state.get("web_query")
        or state["question"]
    )

    results = tavily.invoke(
        {
            "query": q
        }
    )

    web_docs = []

    for r in results or []:

        title = r.get(
            "title",
            "",
        )

        url = r.get(
            "url",
            "",
        )

        content = (
            r.get("content", "")
            or r.get("snippet", "")
        )

        text = (
            f"TITLE: {title}\n"
            f"URL: {url}\n"
            f"CONTENT:\n{content}"
        )

        web_docs.append(
            Document(
                page_content=text,
                metadata={
                    "url": url,
                    "title": title,
                },
            )
        )

    return {
        "web_docs": web_docs
    }


# ============================================================
# 15. REFINE NODE
# ============================================================

def refine(state: State) -> State:

    q = state["question"]

    # --------------------------------------------------------
    # Chọn nguồn context
    # --------------------------------------------------------

    if state.get("verdict") == "CORRECT":

        # Retrieval nội bộ đủ tốt
        docs_to_use = state["good_docs"]

    elif state.get("verdict") == "INCORRECT":

        # Retrieval nội bộ không tốt
        # => dùng web search
        docs_to_use = state["web_docs"]

    else:

        # AMBIGUOUS
        # => kết hợp nội bộ + web
        docs_to_use = (
            state["good_docs"]
            + state["web_docs"]
        )

    # --------------------------------------------------------
    # Ghép Document thành context
    # --------------------------------------------------------

    context = ""

    for doc in docs_to_use:

        context += doc.page_content
        context += "\n\n"

    context = context.strip()

    # --------------------------------------------------------
    # 1. DECOMPOSE
    # --------------------------------------------------------

    strips = decompose_to_sentences(
        context
    )

    # --------------------------------------------------------
    # 2. FILTER
    # --------------------------------------------------------

    kept = []

    for sentence in strips:

        result = filter_chain.invoke(
            {
                "question": q,
                "sentence": sentence,
            }
        )

        if result.keep:
            kept.append(sentence)

    # --------------------------------------------------------
    # 3. RECOMPOSE
    # --------------------------------------------------------

    refined_context = ""

    for sentence in kept:

        refined_context += sentence
        refined_context += "\n"

    refined_context = refined_context.strip()

    return {
        "strips": strips,
        "kept_strips": kept,
        "refined_context": refined_context,
    }


# ============================================================
# 16. FINAL ANSWER GENERATION
# ============================================================

answer_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful ML tutor.\n"
            "Answer ONLY using the provided refined context.\n"
            "If the context is empty or insufficient, "
            "say that the provided context is insufficient.",
        ),
        (
            "human",
            "Question: {question}\n\n"
            "Refined context:\n{refined_context}",
        ),
    ]
)


llm_answer = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    temperature=0,
)


def generate(state: State) -> State:

    # Lấy question
    question = state["question"]

    # Lấy refined context
    refined_context = state["refined_context"]

    # Input cho prompt
    prompt_input = {
        "question": question,
        "refined_context": refined_context,
    }

    # Prompt -> Gemini
    chain = answer_prompt | llm_answer

    response = chain.invoke(
        prompt_input
    )

    # Gemini có thể trả content dạng:
    #
    # [
    #     {
    #         "type": "text",
    #         "text": "..."
    #     }
    # ]
    #
    # hoặc string.
    answer = response.content

    if isinstance(answer, list):

        answer = "\n".join(
            item["text"]
            for item in answer
            if item.get("type") == "text"
        )

    return {
        "answer": answer
    }


# ============================================================
# 17. ROUTING
# ============================================================

def route_after_eval(state: State) -> str:

    if state["verdict"] == "CORRECT":

        return "refine"

    else:

        return "rewrite_query"


# ============================================================
# 18. BUILD LANGGRAPH
# ============================================================

g = StateGraph(State)


# ------------------------------------------------------------
# Add nodes
# ------------------------------------------------------------

g.add_node(
    "retrieve",
    retrieve_node,
)

g.add_node(
    "eval_each_doc",
    eval_each_doc_node,
)

g.add_node(
    "rewrite_query",
    rewrite_query_node,
)

g.add_node(
    "web_search",
    web_search_node,
)

g.add_node(
    "refine",
    refine,
)

g.add_node(
    "generate",
    generate,
)


# ------------------------------------------------------------
# Edges
# ------------------------------------------------------------

g.add_edge(
    START,
    "retrieve",
)

g.add_edge(
    "retrieve",
    "eval_each_doc",
)


# ------------------------------------------------------------
# Conditional routing
# ------------------------------------------------------------

g.add_conditional_edges(
    "eval_each_doc",
    route_after_eval,
    {
        "refine": "refine",
        "rewrite_query": "rewrite_query",
    },
)


# ------------------------------------------------------------
# Web path
# ------------------------------------------------------------

g.add_edge(
    "rewrite_query",
    "web_search",
)

g.add_edge(
    "web_search",
    "refine",
)


# ------------------------------------------------------------
# Final path
# ------------------------------------------------------------

g.add_edge(
    "refine",
    "generate",
)

g.add_edge(
    "generate",
    END,
)


# Compile
app = g.compile()


print("\n=== GRAPH COMPILED SUCCESSFULLY ===")


# ============================================================
# 19. TEST RUN
# ============================================================

question = (
    "Bạn hãy giải thích RAG là gì, "
    "retrieval hoạt động như thế nào, "
    "và RAG khác gì so với việc chỉ dùng LLM?"
)


res = app.invoke(
    {
        "question": question,

        "docs": [],

        "good_docs": [],

        "verdict": "",

        "reason": "",

        "strips": [],

        "kept_strips": [],

        "refined_context": "",

        "web_docs": [],

        "web_query": "",

        "answer": "",
    }
)


# ============================================================
# 20. PRINT RESULT
# ============================================================

print("\n==============================")
print("VERDICT:")
print(res["verdict"])

print("\nREASON:")
print(res["reason"])

print("\nWEB QUERY:")
print(res.get("web_query", ""))

print("\n==============================")
print("FINAL ANSWER:")
print(res["answer"])
print("==============================")
