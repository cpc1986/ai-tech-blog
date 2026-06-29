---
title: "Building a Production RAG Evaluation Pipeline with RAGAS and TruLens: A Hands-On Tutorial for 2026"
date: "2026-06-29"
excerpt: "A step-by-step tutorial for building a rigorous RAG evaluation pipeline using RAGAS and TruLens in 2026. Learn how to measure context relevance, faithfulness, answer relevancy, and hallucination rates with real code, synthetic test set generation, and CI/CD integration for continuous quality monitoring of your retrieval-augmented generation systems."
tags: ["RAG evaluation", "RAGAS", "TruLens", "retrieval-augmented generation", "LLM evaluation", "faithfulness", "hallucination detection", "RAG pipeline", "2026"]
category: "Tutorials"
---

You built a RAG system. It works — mostly. Users occasionally get brilliant answers, sometimes get plausible-sounding hallucinations, and too often get responses that cite irrelevant context. The problem isn't your LLM or your vector database; it's that you have no systematic way to measure whether your RAG pipeline is actually working.

**Evaluation is the missing 80% of most RAG projects.** Teams spend weeks choosing between Chroma and Pinecone, tuning chunk sizes, and debating embedding models — then ship with nothing more than "it looks good to me" as their quality assurance. This tutorial fixes that. We'll build a complete, automated RAG evaluation pipeline using **RAGAS** (Retrieval Augmented Generation Assessment) and **TruLens**, two of the most mature open-source evaluation frameworks in 2026, and integrate it into a CI/CD workflow that catches regressions before they reach users.

## Why RAG Evaluation Is Hard (And Why You Can't Skip It)

RAG systems have four distinct failure modes, each requiring different metrics to detect:

| Failure Mode | What Happens | Example | Metric That Catches It |
|-------------|-------------|---------|----------------------|
| Irrelevant retrieval | Top-k chunks don't contain the answer | User asks about Q3 revenue; retrieved docs are about Q2 marketing | Context Relevance / Context Precision |
| Hallucination | LLM generates claims not supported by context | Context says "revenue grew 12%"; answer says "revenue grew 25%" | Faithfulness |
| Incomplete answer | LLM ignores relevant context | Context contains 3 reasons; answer lists only 1 | Answer Completeness |
| Off-topic response | Answer doesn't address the question | User asks about pricing; answer discusses features | Answer Relevancy |

A single "accuracy" score cannot distinguish between these failures. You need a **multi-metric evaluation framework** — and that's exactly what RAGAS and TruLens provide.

## Tool Comparison: RAGAS vs. TruLens

Both frameworks measure RAG quality, but from different angles:

| Feature | RAGAS | TruLens |
|---------|-------|---------|
| Core focus | Reference-free metric computation | Instrumentation + monitoring |
| Evaluation method | LLM-as-judge + statistical metrics | LLM-as-judge with groundedness chains |
| Synthetic test generation | ✅ Built-in (`TestsetGenerator`) | ❌ Manual or external |
| Production monitoring | ❌ Batch evaluation only | ✅ Real-time dashboard (TruLens + Snowflake) |
|Streaming support | ❌ | ✅ |
| Dashboard | ❌ (use Weights & Biases) | ✅ TruLens Dashboard |
| Latency tracking | ❌ | ✅ Per-component |
| Cost tracking | ❌ | ✅ Per-query token and cost |
| Integration depth | Works on output CSVs | Wraps your app functions |
| Open source | ✅ Apache 2.0 | ✅ Apache 2.0 |
| Python package | `ragas` | `trulens` |

**The optimal strategy is to use both.** RAGAS excels at generating synthetic test sets and computing reference-free metrics in batch. TruLens excels at instrumenting your live application and tracking quality in production. This tutorial covers both.

## Prerequisites and Setup

You'll need Python 3.10+ and about 30 minutes. We'll use a real dataset: a collection of AI research papers.

```bash
# Create a virtual environment
python -m venv rag-eval-env
source rag-eval-env/bin/activate

# Install dependencies
pip install ragas==0.3.1 trulens==1.0.0 \
  langchain==0.3.* langchain-openai==0.3.* \
  langchain-chroma==0.2.* chromadb==1.0.* \
  openai==1.82.* datasets==3.6.* pandas==2.3.*
```

Set your API keys:

```bash
export OPENAI_API_KEY="sk-..."
export OPENAI_MODEL="gpt-4.1-mini"  # Cost-effective for evaluation
```

## Part 1: Building the RAG Application

Before we can evaluate, we need a RAG application. Here's a minimal but realistic one:

```python
# rag_app.py
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Initialize components
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)
vectorstore = Chroma(
    collection_name="ai_research",
    embedding_function=embeddings,
    persist_directory="./chroma_db"
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# RAG prompt
prompt = ChatPromptTemplate.from_template("""
You are an expert AI research assistant. Answer the question based only 
on the provided context. If the context doesn't contain enough information 
to answer the question, say "I don't have enough information to answer this."

Context:
{context}

Question: {question}

Answer:
""")

def format_docs(docs):
    return "\n\n---\n\n".join(doc.page_content for doc in docs)

# Build the chain
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Expose retrieval separately for evaluation
def retrieve(question: str):
    """Returns retrieved documents for evaluation."""
    return retriever.invoke(question)

def generate(question: str, context_docs: list):
    """Generates answer from context documents."""
    context = format_docs(context_docs)
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"context": context, "question": question})
```

## Part 2: Generating a Synthetic Test Set with RAGAS

The biggest challenge in RAG evaluation is getting test data. RAGAS solves this with its `TestsetGenerator`, which takes your document corpus and automatically generates (question, answer, contexts) triplets.

```python
# generate_testset.py
from ragas.testset import TestsetGenerator
from ragas.dataset_schema import Testset
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import DirectoryLoader, TextLoader

# Load your documents
loader = DirectoryLoader(
    "./data/ai_papers/",  # Your document directory
    glob="**/*.txt",
    loader_cls=TextLoader,
    loader_kwargs={"encoding": "utf-8"}
)
documents = loader.load()

print(f"Loaded {len(documents)} documents")

# Configure the generator
generator_llm = ChatOpenAI(model="gpt-4.1-mini")
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

generator = TestsetGenerator(
    llm=generator_llm,
    embedding_model=embeddings,
)

# Generate test set with distribution control
testset = generator.generate_with_langchain_docs(
    documents,
    testset_size=50,  # Start with 50, scale to 200+ for production
    distributions={
        "simple": 0.3,       # Direct fact lookup
        "reasoning": 0.4,    # Requires multi-step reasoning
        "multi_context": 0.3 # Requires synthesizing across documents
    }
)

# Save for reuse
testset_df = testset.to_pandas()
testset_df.to_csv("rag_test_set.csv", index=False)
print(f"Generated {len(testset_df)} test cases")
print(testset_df[["user_input", "reference"]].head())
```

The output looks like:

```
Loaded 47 documents
Generated 50 test cases
                                         user_input  \
0  What is the key innovation in the Mamba archite...   
1  How does mixture-of-experts routing differ from...   
2  What are the three main challenges in RLHF that...   

                                           reference  
0  The key innovation in Mamba is selective state ...  
1  MoE routing uses a learned gating function that...  
2  The three main challenges are reward hacking, s...  
```

## Part 3: Batch Evaluation with RAGAS

Now we evaluate our RAG pipeline against the test set using RAGAS's core metrics:

```python
# evaluate_ragas.py
import pandas as pd
from ragas import evaluate
from ragas.metrics import (
    context_precision,
    context_recall,
    faithfulness,
    answer_relevancy,
)
from datasets import Dataset

# Load test set
testset_df = pd.read_csv("rag_test_set.csv")

# Run RAG pipeline on each question to get answers and retrieved contexts
questions = testset_df["user_input"].tolist()
ground_truths = testset_df["reference"].tolist()

# Collect RAG outputs
answers = []
contexts = []

for q in questions:
    # Retrieve documents
    docs = retrieve(q)
    contexts.append([doc.page_content for doc in docs])
    # Generate answer
    answer = generate(q, docs)
    answers.append(answer)

# Build evaluation dataset
eval_data = {
    "question": questions,
    "answer": answers,
    "contexts": contexts,
    "ground_truth": ground_truths,
}
dataset = Dataset.from_dict(eval_data)

# Run evaluation with all four RAGAS metrics
metrics = [
    faithfulness,           # Are claims in the answer supported by context?
    answer_relevancy,       # Is the answer relevant to the question?
    context_precision,      # Are relevant chunks ranked higher?
    context_recall,         # Were all necessary chunks retrieved?
]

results = evaluate(
    dataset,
    metrics=metrics,
    llm=ChatOpenAI(model="gpt-4.1-mini"),
    embeddings=OpenAIEmbeddings(model="text-embedding-3-small"),
)

# Display results
print("\n=== RAGAS Evaluation Results ===")
print(results.to_pandas().describe())

# Save detailed results
results_df = results.to_pandas()
results_df.to_csv("ragas_evaluation_results.csv", index=False)
```

Typical output for a first evaluation:

```
=== RAGAS Evaluation Results ===
       faithfulness  answer_relevancy  context_precision  context_recall
count      50.00000         50.000000          50.000000       50.000000
mean        0.72400          0.836200           0.613000        0.572000
std         0.18352          0.124300           0.251300        0.213400
min         0.25000          0.420000           0.000000        0.100000
25%         0.62000          0.780000           0.500000        0.430000
50%         0.78000          0.870000           0.667000        0.600000
75%         0.88000          0.940000           0.800000        0.740000
max         1.00000          0.990000           1.000000        0.950000
```

**Interpreting the scores:** A context_recall of 0.57 means your retriever is missing ~43% of the relevant documents. That's the bottleneck. A faithfulness of 0.72 means ~28% of claims in answers aren't grounded in the retrieved context. Fix retrieval first (try increasing k, switching to MMR retrieval, or using a re-ranker), then address faithfulness.

## Part 4: Identifying and Fixing Weak Spots

RAGAS gives you per-question scores. Use them to categorize failures:

```python
# analyze_failures.py
import pandas as pd

results_df = pd.read_csv("ragas_evaluation_results.csv")

# Define failure thresholds
RETRIEVAL_FAILURE = results_df["context_recall"] < 0.4
FAITHFULNESS_FAILURE = results_df["faithfulness"] < 0.6
RELEVANCY_FAILURE = results_df["answer_relevancy"] < 0.7

print("=== Failure Analysis ===")
print(f"Retrieval failures (context_recall < 0.4): {RETRIEVAL_FAILURE.sum()}")
print(f"Faithfulness failures (faithfulness < 0.6): {FAITHFULNESS_FAILURE.sum()}")
print(f"Answer relevancy failures (< 0.7): {RELEVANCY_FAILURE.sum()}")

# Show worst examples per category
print("\n--- Worst Retrieval Failures ---")
worst_retrieval = results_df[RETRIEVAL_FAILURE].nsmallest(3, "context_recall")
for _, row in worst_retrieval.iterrows():
    print(f"Q: {row['question'][:80]}...")
    print(f"  context_recall: {row['context_recall']:.2f}")
    print()

print("\n--- Worst Faithfulness Failures ---")
worst_faith = results_df[FAITHFULNESS_FAILURE].nsmallest(3, "faithfulness")
for _, row in worst_faith.iterrows():
    print(f"Q: {row['question'][:80]}...")
    print(f"  faithfulness: {row['faithfulness']:.2f}")
    print(f"  Answer: {row['answer'][:120]}...")
    print()
```

### Common Fixes Ranked by Impact

| Fix | When to Apply | Expected Improvement |
|-----|--------------|---------------------|
| Increase k from 5 → 10 | context_recall < 0.5 | +15–25% context_recall |
| Add Cross-Encoder re-ranker | context_precision < 0.6 | +20–30% context_precision |
| Reduce chunk size (512 → 256) | faithfulness < 0.7 | +10–20% faithfulness |
| Add metadata filtering | domain-specific queries fail | +15–30% context_recall |
| Switch to MMR retrieval | redundant chunks in top-k | +10–15% context_recall |
| Improve prompt with citations | faithfulness < 0.8 | +5–15% faithfulness |
| Add HyDE (Hypothetical Document Embedding) | short factual queries fail | +10–20% context_recall |

## Part 5: Production Monitoring with TruLens

Batch evaluation catches regressions before deployment. But what about quality in production? TruLens instruments your RAG application to track every query in real time.

```python
# trulens_monitoring.py
from trulens.core import TruSession
from trulens.apps.basic import TruBasicApp
from trulens.core import Feedback
from trulens.providers.openai import OpenAI as TruOpenAI

session = TruSession()

# Set up feedback functions
provider = TruOpenAI(model_engine="gpt-4.1-mini")

# Groundedness: Is the answer supported by the retrieved context?
groundedness = Feedback(
    provider.groundedness_measure_with_cot_reasons,
    name="Groundedness"
).on_input_output()

# Answer relevance: Does the answer address the question?
answer_relevance = Feedback(
    provider.relevance,
    name="Answer Relevance"
).on_input_output()

# Context relevance: Is the retrieved context relevant to the question?
context_relevance = Feedback(
    provider.context_relevance_with_cot_reasons,
    name="Context Relevance"
).on_input().on_output()

# Wrap your RAG application
tru_rag = TruBasicApp(
    rag_chain.invoke,
    app_name="AI Research RAG",
    app_version="v1.2",
    feedbacks=[groundedness, answer_relevance, context_relevance]
)

# Use the instrumented app
with tru_rag as recording:
    result = tru_rag.main_call("What are the latest advances in mixture-of-experts architectures?")
    print(f"Answer: {result}")

# Launch the dashboard
from trulens.dashboard import run_dashboard
run_dashboard(session)
```

TruLens records every query with per-metric scores, latency, token counts, and cost. You can filter by score thresholds to find problematic queries:

```python
# Query low-quality interactions from the database
from trulens.core import TruSession

session = TruSession()

# Find all queries where groundedness < 0.5
records, feedbacks = session.get_records_and_feedback()
low_groundedness = feedbacks[feedbacks["Groundedness"] < 0.5]
print(f"Queries with groundedness < 0.5: {len(low_groundedness)}")
```

## Part 6: CI/CD Integration — Automated Regression Detection

The final piece is making evaluation automatic. Every time you change your RAG pipeline — new embedding model, different chunk size, updated prompt — the evaluation should run and block merges that degrade quality.

```yaml
# .github/workflows/rag-eval.yml
name: RAG Evaluation

on:
  pull_request:
    paths:
      - 'rag/**'
      - 'evaluation/**'

jobs:
  evaluate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
          
      - name: Install dependencies
        run: |
          pip install -r requirements-eval.txt
          
      - name: Run RAGAS evaluation
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
        run: |
          python evaluate_ragas.py
          
      - name: Check quality gates
        run: |
          python check_quality_gates.py
```

```python
# check_quality_gates.py
import pandas as pd
import sys

results_df = pd.read_csv("ragas_evaluation_results.csv")

# Define quality gates (adjust to your standards)
GATES = {
    "faithfulness":       {"threshold": 0.75, "action": "block"},
    "answer_relevancy":   {"threshold": 0.80, "action": "block"},
    "context_precision":  {"threshold": 0.60, "action": "warn"},
    "context_recall":     {"threshold": 0.55, "action": "warn"},
}

failed_gates = []
warning_gates = []

for metric, config in GATES.items():
    if metric not in results_df.columns:
        continue
    mean_score = results_df[metric].mean()
    status = "✅ PASS" if mean_score >= config["threshold"] else "❌ FAIL"
    print(f"{metric}: {mean_score:.3f} (threshold: {config['threshold']:.2f}) {status}")
    
    if mean_score < config["threshold"]:
        if config["action"] == "block":
            failed_gates.append(metric)
        else:
            warning_gates.append(metric)

if failed_gates:
    print(f"\n🚫 Blocked by quality gates: {', '.join(failed_gates)}")
    sys.exit(1)
elif warning_gates:
    print(f"\n⚠️  Warnings: {', '.join(warning_gates)}")
    sys.exit(0)
else:
    print("\n✅ All quality gates passed!")
    sys.exit(0)
```

## Part 7: Tracking Evaluation Over Time

Quality isn't a one-time measurement. You need to track trends. Here's a lightweight approach using a JSONL log:

```python
# track_metrics.py
import json
import pandas as pd
from datetime import datetime

def log_evaluation(pipeline_version: str, results_df: pd.DataFrame, testset_size: int):
    """Log evaluation results for trend tracking."""
    record = {
        "timestamp": datetime.now().isoformat(),
        "pipeline_version": pipeline_version,
        "testset_size": testset_size,
        "metrics": {
            "faithfulness_mean": float(results_df["faithfulness"].mean()),
            "faithfulness_std": float(results_df["faithfulness"].std()),
            "answer_relevancy_mean": float(results_df["answer_relevancy"].mean()),
            "answer_relevancy_std": float(results_df["answer_relevancy"].std()),
            "context_precision_mean": float(results_df["context_precision"].mean()),
            "context_precision_std": float(results_df["context_precision"].std()),
            "context_recall_mean": float(results_df["context_recall"].mean()),
            "context_recall_std": float(results_df["context_recall"].std()),
        }
    }
    
    with open("evaluation_history.jsonl", "a") as f:
        f.write(json.dumps(record) + "\n")
    
    return record

# Usage
log_evaluation("v1.3", results_df, len(results_df))

# Compare with previous runs
history_df = pd.read_json("evaluation_history.jsonl", lines=True)
metrics_cols = [c for c in history_df.columns if c.startswith("metrics.")]
print(history_df[["timestamp", "pipeline_version"] + metrics_cols].tail(5).to_string())
```

## Putting It All Together: The Evaluation Flywheel

```
┌─────────────────────┐
│  Generate Test Set  │ ← RAGAS TestsetGenerator (one-time + periodic refresh)
│  (50-200 questions) │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   Run Batch Eval    │ ← RAGAS: faithfulness, relevancy, context quality
│   (RAGAS metrics)   │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Analyze Failures   │ ← Categorize: retrieval vs faithfulness vs relevancy
│  & Apply Fixes      │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Check Quality Gates│ ← CI/CD: block if below threshold
│  (automated)        │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Deploy & Monitor   │ ← TruLens: real-time groundedness, cost, latency
│  (TruLens)          │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Track Trends Over  │ ← JSONL logs: detect drift before users notice
│  Time               │
└──────────┬──────────┘
           │
           └──────► Loop back to "Run Batch Eval" after each change
```

## Key Takeaways

1. **Don't skip evaluation.** A RAG system without metrics is just a demo. Production requires quantitative quality assurance.

2. **Start with RAGAS batch evaluation** to get baseline scores. Target faithfulness ≥ 0.75, answer relevancy ≥ 0.80, context recall ≥ 0.55 as minimum viable thresholds.

3. **Fix retrieval first.** In most RAG systems, retrieval failures (low context recall) cascade into hallucination and incomplete answers. Adding a re-ranker is usually the highest-impact single change.

4. **Use TruLens for production monitoring.** Batch evaluation catches pre-deployment regressions; TruLens catches post-deployment drift. You need both.

5. **Automate with CI/CD.** Manual evaluation is unsustainable. Wire quality gates into your pull request workflow so regressions are caught automatically.

6. **Track trends, not snapshots.** A single evaluation tells you where you are. Trend data tells you where you're going — and whether you're improving or slowly degrading.

7. **Regenerate test sets periodically.** As your document corpus changes, your test set becomes stale. Re-run `TestsetGenerator` monthly or after significant document updates.

## Resources

- [RAGAS Documentation](https://docs.ragas.io/) — Metric definitions and API reference
- [TruLens Documentation](https://www.trulens.org/) — Instrumentation and dashboard setup
- [RAGAS GitHub](https://github.com/explodinggradients/ragas) — Source code and examples
- [TruLens GitHub](https://github.com/truera/trulens) — Source code and integrations
- [LangChain Evaluation Guide](https://python.langchain.com/docs/guides/evaluation/) — Additional evaluation patterns

---

*Evaluating RAG systems isn't optional — it's the difference between a system that works sometimes and one that works reliably. Build the pipeline, set the gates, and let the metrics tell you when something breaks.*
