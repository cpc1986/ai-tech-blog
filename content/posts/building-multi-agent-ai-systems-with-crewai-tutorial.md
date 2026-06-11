---
title: "Building Multi-Agent AI Systems with CrewAI: A Hands-On Tutorial"
date: "2026-06-11"
excerpt: "A practical, code-heavy tutorial on building production-grade multi-agent AI systems with CrewAI in 2026. Learn how to define agents, orchestrate tasks, use tools, handle memory, and deploy your crew to solve complex real-world problems."
tags: ["CrewAI", "multi-agent systems", "AI agents", "LangChain", "Python", "tutorial", "LLM orchestration", "2026"]
category: "Tutorials"
---

Multi-agent AI systems are one of the most transformative patterns to emerge in 2026. Instead of relying on a single large language model to do everything, you decompose complex workflows into specialized agents — each with a defined role, a set of tools, and a clear objective — then orchestrate them to collaborate. CrewAI has established itself as the leading open-source framework for building these systems, offering a clean Pythonic API, built-in memory, and flexible orchestration patterns.

This tutorial walks you through building a fully functional multi-agent research-and-reporting system from scratch. By the end, you'll have a crew of three agents that can research a topic, analyze the findings, and produce a polished report — all autonomously.

## Why Multi-Agent Systems?

Single-prompt LLM interactions hit a ceiling fast. When tasks get complex — involving research, synthesis, fact-checking, and formatting — a monolithic prompt becomes brittle and hard to maintain. Multi-agent systems solve this by applying the **separation of concerns** principle to AI workflows:

| Approach | Pros | Cons |
|----------|------|------|
| Single prompt | Simple setup | Brittle, hard to debug, context bloat |
| Chained pipelines (LangChain) | Sequential control | Rigid, no collaboration between steps |
| Multi-agent (CrewAI) | Role specialization, parallelism, collaborative | More setup complexity |

The key insight: **agents with narrow expertise outperform a single generalist when tasks are multi-faceted.** This mirrors how human teams work — you wouldn't ask one person to research, write, edit, and publish a report.

## Prerequisites and Setup

Before we begin, ensure you have:

- Python 3.10+ installed
- An OpenAI API key (or Anthropic, Ollama, or any LiteLLM-compatible provider)
- Basic familiarity with Python and LLM APIs

Install CrewAI and its dependencies:

```bash
pip install crewai crewai-tools
```

Set your API key:

```bash
export OPENAI_API_KEY="sk-your-key-here"
```

If you prefer using a local model through Ollama:

```bash
# Install Ollama first: https://ollama.com
ollama pull llama3.1:70b
# CrewAI will use OPENAI_MODEL_NAME and OPENAI_API_BASE to point to Ollama
export OPENAI_API_BASE="http://localhost:11434/v1"
export OPENAI_MODEL_NAME="llama3.1:70b"
export OPENAI_API_KEY="ollama"
```

Verify the installation:

```python
import crewai
print(crewai.__version__)  # Should print 0.80+ in June 2026
```

## Understanding CrewAI Core Concepts

CrewAI revolves around four primitives:

### 1. Agents

An agent is an autonomous unit with a **role**, a **goal**, and a **backstory**. These three fields shape the agent's behavior far more than most developers expect — they prime the LLM's system prompt and influence tool selection, reasoning style, and output format.

```python
from crewai import Agent

researcher = Agent(
    role="Senior Technology Researcher",
    goal="Find comprehensive, factual, and up-to-date information on the given topic",
    backstory=(
        "You are a veteran technology researcher with 15 years of experience "
        "at McKinsey and Gartner. You have a reputation for thoroughness and "
        "accuracy. You always cite sources, distinguish between facts and "
        "opinions, and flag uncertainty. You never fabricate information."
    ),
    verbose=True,
    allow_delegation=False,
    max_iter=5,
)
```

Key parameters explained:

| Parameter | Purpose | Default |
|-----------|---------|---------|
| `role` | Defines the agent's expertise and perspective | Required |
| `goal` | What the agent optimizes for | Required |
| `backstory` | Rich context that shapes behavior and tone | Required |
| `verbose` | Log agent's reasoning steps | `False` |
| `allow_delegation` | Can this agent assign tasks to other agents? | `False` |
| `max_iter` | Maximum reasoning iterations before forced output | `25` |
| `memory` | Enable conversational memory across tasks | `True` |

### 2. Tasks

A task defines a specific assignment — what needs to be done, what the expected output looks like, and which agent is responsible.

```python
from crewai import Task

research_task = Task(
    description=(
        "Research the following topic thoroughly: {topic}. "
        "Focus on: (1) current state of the art, (2) key players and their "
        "contributions, (3) recent breakthroughs in the last 12 months, "
        "(4) challenges and open problems. Provide sources for all claims."
    ),
    expected_output=(
        "A structured research brief with 4 sections: Overview, Key Players, "
        "Recent Breakthroughs, Open Challenges. Include at least 10 specific "
        "data points or statistics with source attribution."
    ),
    agent=researcher,
)
```

### 3. Tools

Agents can use tools to interact with the outside world. CrewAI ships with dozens of built-in tools, and you can easily create custom ones:

```python
from crewai_tools import (
    SerperDevTool,
    ScrapeWebsiteTool,
    FileReadTool,
    DirectoryReadTool,
)

# Web search tool (requires SERPER_API_KEY)
search_tool = SerperDevTool()

# Web scraping tool
scrape_tool = ScrapeWebsiteTool()

# File system tools
file_read_tool = FileReadTool()
```

Creating a custom tool is straightforward using the `@tool` decorator:

```python
from crewai.tools import tool

@tool("Database Query Tool")
def query_database(sql: str) -> str:
    """Execute a SQL query against the analytics database and return results.
    
    Args:
        sql: A valid SQL query string. Only SELECT statements are allowed.
    """
    import sqlite3
    conn = sqlite3.connect("analytics.db")
    cursor = conn.cursor()
    try:
        # Safety: only allow SELECT
        if not sql.strip().upper().startswith("SELECT"):
            return "Error: Only SELECT queries are permitted."
        cursor.execute(sql)
        rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        result = [dict(zip(columns, row)) for row in rows]
        return str(result[:50])  # Limit to 50 rows
    except Exception as e:
        return f"Query error: {e}"
    finally:
        conn.close()
```

### 4. Crews

The crew is the orchestrator — it manages the agents, assigns tasks, and controls the execution flow.

```python
from crewai import Crew, Process

crew = Crew(
    agents=[researcher, analyst, writer],
    tasks=[research_task, analysis_task, writing_task],
    process=Process.sequential,  # or Process.hierarchical
    verbose=True,
)
```

## Building the Research Crew: Step by Step

Now let's assemble a complete three-agent crew. The use case: given any topic, produce a well-researched, analytically rigorous, and professionally formatted report.

### Step 1: Define the Agents

```python
from crewai import Agent

researcher = Agent(
    role="Senior Technology Researcher",
    goal="Discover comprehensive, factual information and identify key patterns",
    backstory=(
        "You are a technology researcher with deep expertise in AI, software "
        "engineering, and emerging technologies. You spent a decade at Google "
        "Research and now consult for Fortune 500 companies. You're known for "
        "your systematic approach: you start broad, then drill into specifics. "
        "You always prioritize primary sources over secondary reporting."
    ),
    tools=[search_tool, scrape_tool],
    verbose=True,
    allow_delegation=False,
)

analyst = Agent(
    role="Data Analyst and Critical Thinker",
    goal="Synthesize research into actionable insights with quantitative rigor",
    backstory=(
        "You are a quantitative analyst trained at MIT and formerly at "
        "Two Sigma. You excel at identifying patterns, evaluating evidence "
        "strength, and distinguishing signal from noise. You always present "
        "multiple viewpoints and quantify uncertainty. You're skeptical of "
        "hype and look for contrarian evidence."
    ),
    verbose=True,
    allow_delegation=False,
)

writer = Agent(
    role="Senior Technical Writer",
    goal="Transform analytical findings into clear, engaging, and well-structured prose",
    backstory=(
        "You are a senior technical writer who has authored books for "
        "O'Reilly Media and published in IEEE Software. You write with "
        "precision but never sacrifice readability. You use concrete "
        "examples, effective analogies, and avoid jargon unless it's "
        "defined. Your reports are the gold standard in the industry."
    ),
    verbose=True,
    allow_delegation=False,
)
```

### Step 2: Define the Tasks

```python
from crewai import Task

research_task = Task(
    description=(
        "Conduct thorough research on: {topic}.\n\n"
        "Phase 1: Identify the 5-7 most important subtopics.\n"
        "Phase 2: For each subtopic, find key facts, statistics, and "
        "expert opinions using web search.\n"
        "Phase 3: Identify contradictory viewpoints and note areas of "
        "consensus vs. debate.\n\n"
        "Deliver a structured research brief covering each subtopic."
    ),
    expected_output=(
        "A comprehensive research document with sections for each subtopic. "
        "Each section must include: key facts (with sources), relevant "
        "statistics (with numbers), expert quotes, and areas of uncertainty. "
        "Minimum 2000 words of substantive content."
    ),
    agent=researcher,
)

analysis_task = Task(
    description=(
        "Analyze the research brief and produce an analytical synthesis.\n\n"
        "For each major finding:\n"
        "1. Rate the evidence strength (Strong / Moderate / Weak)\n"
        "2. Identify implications and second-order effects\n"
        "3. Note risks and contrarian perspectives\n"
        "4. Assign a confidence score (0-100%)\n\n"
        "Conclude with a prioritized list of strategic recommendations."
    ),
    expected_output=(
        "An analytical report with: evidence ratings, implications analysis, "
        "risk assessment, confidence scores, and 5-7 prioritized strategic "
        "recommendations. Include a summary comparison table."
    ),
    agent=analyst,
    context=[research_task],  # Receives output of research_task
)

writing_task = Task(
    description=(
        "Using the research brief and analytical synthesis, write a "
        "polished, publication-ready report on: {topic}.\n\n"
        "Structure:\n"
        "- Executive Summary (3-4 paragraphs)\n"
        "- Introduction and Background\n"
        "- Key Findings (with charts described in text)\n"
        "- Analysis and Implications\n"
        "- Strategic Recommendations\n"
        "- Conclusion and Future Outlook\n\n"
        "Use markdown formatting. Include tables where data comparison is "
        "useful. Keep paragraphs to 3-5 sentences. Use active voice."
    ),
    expected_output=(
        "A complete, publication-ready markdown report of 2000-3000 words "
        "with proper headings, tables, and clear structure."
    ),
    agent=writer,
    context=[research_task, analysis_task],  # Receives both prior outputs
    output_file="output/report.md",  # Auto-save to file
)
```

Notice the `context` parameter: this is how tasks form a **data flow graph**. The analysis task receives the research output; the writing task receives both.

### Step 3: Assemble and Run the Crew

```python
from crewai import Crew, Process

research_crew = Crew(
    agents=[researcher, analyst, writer],
    tasks=[research_task, analysis_task, writing_task],
    process=Process.sequential,
    verbose=True,
    memory=True,  # Enable short-term memory between agents
)

# Run the crew
result = research_crew.kickoff(
    inputs={"topic": "The State of Open-Source AI Models in 2026"}
)

print("=== FINAL REPORT ===")
print(result)
```

### Step 4: Execution Flow

When you call `kickoff()`, CrewAI orchestrates the following sequence:

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Researcher  │────>│   Analyst   │────>│   Writer    │
│  (Search +   │     │ (Synthesize │     │ (Format +   │
│   Scrape)    │     │  + Evaluate)│     │  Polish)    │
└─────────────┘     └─────────────┘     └─────────────┘
      │                    │                    │
  research_task      analysis_task        writing_task
      │                    │                    │
      └──────────context───┘────────context─────┘
```

Each agent:
1. Receives its task description + context from prior tasks
2. Reasons about how to approach the task (visible in verbose mode)
3. Iterates — using tools, refining its approach — up to `max_iter` times
4. Produces a final output that becomes context for downstream tasks

## Advanced Patterns

### Hierarchical Process with a Manager Agent

For complex workflows, use a manager agent that dynamically assigns work:

```python
from crewai import Crew, Process, Agent, Task

manager = Agent(
    role="Project Manager",
    goal="Coordinate the team efficiently and ensure high-quality output",
    backstory=(
        "You are a senior project manager who excels at breaking down "
        "complex objectives into clear subtasks and assigning them to "
        "the right team members."
    ),
    allow_delegation=True,
    verbose=True,
)

# With hierarchical process, you don't assign agents to tasks manually
# The manager agent decides who does what
crew = Crew(
    agents=[researcher, analyst, writer, manager],
    tasks=[research_task, analysis_task, writing_task],
    process=Process.hierarchical,
    manager_agent=manager,
    verbose=True,
)

result = crew.kickoff(
    inputs={"topic": "Quantum Computing Applications in Drug Discovery"}
)
```

In hierarchical mode, the manager agent can **reassign tasks**, **request revisions**, and **merge outputs** from multiple agents working in parallel.

### Adding Custom Tools for Domain-Specific Work

Let's add a tool that queries a vector database:

```python
from crewai.tools import tool
import chromadb

@tool("Vector Knowledge Base Search")
def search_knowledge_base(query: str, n_results: int = 5) -> str:
    """Search the internal knowledge base for relevant documents.
    
    Args:
        query: Natural language search query.
        n_results: Number of results to return (default 5).
    """
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_collection("internal_docs")
    
    results = collection.query(
        query_texts=[query],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )
    
    formatted = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        formatted.append(
            f"[Relevance: {1 - dist:.2f}] Source: {meta.get('source', 'unknown')}\n{doc}"
        )
    
    return "\n\n---\n\n".join(formatted)

# Assign to an agent
researcher = Agent(
    role="Research Analyst",
    goal="Find and synthesize information from both web and internal sources",
    backstory="...",
    tools=[search_tool, scrape_tool, search_knowledge_base],
)
```

### Human-in-the-Loop Review

For production systems, you can insert human approval points:

```python
from crewai import Task

review_task = Task(
    description=(
        "Review the drafted report for accuracy, completeness, and quality. "
        "Approve or request specific revisions."
    ),
    expected_output="An approved final report OR a list of specific revision requests.",
    agent=reviewer,
    human_input=True,  # This pauses execution for human review
)
```

When `human_input=True`, CrewAI pauses after the agent completes its work and prompts the human operator for approval or revision instructions before continuing.

### Using Callbacks for Monitoring

```python
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("crew_monitor")

class CrewMonitor:
    def __init__(self):
        self.start_time = time.time()
        self.task_times = {}

    def task_callback(self, task_output):
        elapsed = time.time() - self.start_time
        self.task_times[task_output.task_description[:50]] = elapsed
        logger.info(
            f"Task completed in {elapsed:.1f}s: "
            f"{task_output.task_description[:80]}..."
        )
        logger.info(f"Output length: {len(task_output.result)} chars")

monitor = CrewMonitor()

crew = Crew(
    agents=[researcher, analyst, writer],
    tasks=[research_task, analysis_task, writing_task],
    process=Process.sequential,
    step_callback=monitor.task_callback,
    verbose=True,
)
```

## Performance Optimization Tips

After deploying multi-agent systems in production, here are the key optimization strategies:

### 1. Choose the Right Model Per Agent

Not every agent needs GPT-4-class reasoning. Use cheaper, faster models for structured tasks:

```python
from crewai import LLM

# High-reasoning agent
researcher = Agent(
    role="Researcher",
    goal="...",
    backstory="...",
    llm=LLM(model="gpt-4o", temperature=0.3),
)

# Structured output agent — smaller model is fine
writer = Agent(
    role="Writer",
    goal="...",
    backstory="...",
    llm=LLM(model="gpt-4o-mini", temperature=0.7),
)

# Manager needs strong reasoning
manager = Agent(
    role="Manager",
    goal="...",
    backstory="...",
    llm=LLM(model="gpt-4o", temperature=0.1),
)
```

### 2. Control Token Usage with max_iter

Setting `max_iter` too high wastes tokens. Too low produces incomplete work. Benchmarks from our production deployments:

| Agent Type | Recommended `max_iter` | Avg Tokens/Run |
|-----------|----------------------|---------------|
| Research (web search) | 8-12 | 4,000-8,000 |
| Analysis (synthesis) | 3-5 | 3,000-5,000 |
| Writing (drafting) | 2-4 | 5,000-10,000 |
| Code generation | 5-8 | 3,000-7,000 |

### 3. Cache Tool Results

For research tasks that repeat (e.g., monitoring the same topics), cache web search results:

```python
import hashlib
import json
from datetime import datetime, timedelta

cache = {}

@tool("Cached Web Search")
def cached_search(query: str) -> str:
    """Search the web with caching to avoid duplicate API calls."""
    cache_key = hashlib.md5(query.lower().encode()).hexdigest()
    
    if cache_key in cache:
        cached_time, cached_result = cache[cache_key]
        if datetime.now() - cached_time < timedelta(hours=6):
            return f"[CACHED] {cached_result}"
    
    result = search_tool._run(search_query=query)
    cache[cache_key] = (datetime.now(), result)
    return result
```

## Real-World Deployment: CLI Tool

Here's how to package your crew as a reusable CLI tool:

```python
#!/usr/bin/env python3
"""
research_crew.py — Run a multi-agent research crew from the command line.

Usage:
    python research_crew.py "Topic to research"
    python research_crew.py "Topic" --output ./reports/ --verbose
"""

import argparse
import sys
from crewai import Agent, Task, Crew, Process, LLM
from crewai_tools import SerperDevTool

def build_crew(output_dir: str = "output", verbose: bool = False) -> Crew:
    search_tool = SerperDevTool()

    researcher = Agent(
        role="Senior Research Analyst",
        goal="Find comprehensive, factual information on any given topic",
        backstory="You are a meticulous researcher with 15 years at top consulting firms.",
        tools=[search_tool],
        verbose=verbose,
        max_iter=8,
    )

    analyst = Agent(
        role="Strategic Analyst",
        goal="Synthesize research into actionable strategic insights",
        backstory="You are a former McKinsey consultant specializing in technology strategy.",
        verbose=verbose,
        max_iter=5,
    )

    writer = Agent(
        role="Technical Writer",
        goal="Produce clear, well-structured reports",
        backstory="You are an O'Reilly author known for making complex topics accessible.",
        verbose=verbose,
        max_iter=3,
        llm=LLM(model="gpt-4o-mini"),
    )

    research_task = Task(
        description="Research: {topic}. Cover current state, key players, recent developments, challenges.",
        expected_output="A structured research brief, 1500+ words with sources.",
        agent=researcher,
    )

    analysis_task = Task(
        description="Analyze research and identify strategic implications, risks, and opportunities.",
        expected_output="Analytical synthesis with evidence ratings and recommendations.",
        agent=analyst,
        context=[research_task],
    )

    writing_task = Task(
        description="Write a publication-ready markdown report on: {topic}.",
        expected_output="Complete markdown report with executive summary, findings, and recommendations.",
        agent=writer,
        context=[research_task, analysis_task],
        output_file=f"{output_dir}/report.md",
    )

    return Crew(
        agents=[researcher, analyst, writer],
        tasks=[research_task, analysis_task, writing_task],
        process=Process.sequential,
        verbose=verbose,
    )

def main():
    parser = argparse.ArgumentParser(description="Run AI Research Crew")
    parser.add_argument("topic", help="Topic to research")
    parser.add_argument("--output", default="output", help="Output directory")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()

    crew = build_crew(output_dir=args.output, verbose=args.verbose)
    result = crew.kickoff(inputs={"topic": args.topic})
    
    print(f"\n{'='*60}")
    print(f"Report saved to: {args.output}/report.md")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
```

Run it:

```bash
python research_crew.py "The Impact of AI Agents on Enterprise Software Development"
```

## Common Pitfalls and Debugging

### Agent Loops

If an agent keeps calling the same tool repeatedly (a common failure mode), reduce `max_iter` and make the task description more specific:

```python
# Bad: vague task leads to loops
task = Task(
    description="Research AI.",
    expected_output="Something about AI.",
    agent=researcher,
)

# Good: specific, bounded task
task = Task(
    description=(
        "Research exactly 3 recent developments in AI agent frameworks "
        "from 2026. For each, provide: framework name, key innovation, "
        "GitHub stars, and one code example."
    ),
    expected_output=(
        "A list of exactly 3 items, each with name, innovation, stars, and code."
    ),
    agent=researcher,
)
```

### Context Window Overflow

When chaining tasks, downstream agents can hit context limits. Mitigate this:

```python
from crewai import Task

summary_task = Task(
    description=(
        "Summarize the previous research output into a concise brief. "
        "Keep it under 500 words. Focus on the top 5 most important findings."
    ),
    expected_output="A 500-word summary with 5 key findings.",
    agent=summarizer,
    context=[research_task],  # Only receives what it needs
)
```

### Tool Permission Errors

Agents sometimes attempt actions outside their tool scope. Always validate tool inputs:

```python
@tool("Safe File Writer")
def safe_write_file(filepath: str, content: str) -> str:
    """Write content to a file within the allowed directory."""
    import os
    ALLOWED_DIR = "/app/output"
    
    # Resolve and validate path
    full_path = os.path.realpath(os.path.join(ALLOWED_DIR, filepath))
    if not full_path.startswith(ALLOWED_DIR):
        return f"Error: Access denied. Files must be within {ALLOWED_DIR}"
    
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    with open(full_path, "w") as f:
        f.write(content)
    
    return f"Successfully wrote {len(content)} chars to {filepath}"
```

## Comparison: CrewAI vs. Alternatives

| Feature | CrewAI | AutoGen | LangGraph |
|---------|--------|---------|-----------|
| Abstraction level | High (roles, tasks) | Medium (conversations) | Low (graph nodes) |
| Learning curve | Gentle | Moderate | Steep |
| Built-in tools | 50+ | Limited | Extensive (via LangChain) |
| Memory management | Automatic | Manual | Manual |
| Hierarchical orchestration | Built-in | Manual | Custom graph |
| Human-in-the-loop | Supported | Supported | Supported |
| Observability | Basic | Basic | Rich (LangSmith) |
| Best for | Rapid prototyping, business workflows | Research, multi-model chat | Complex state machines, production pipelines |

**Choose CrewAI when:** You want to iterate quickly on multi-agent workflows and prefer a role-based abstraction.

**Choose LangGraph when:** You need fine-grained control over state transitions, conditional routing, and production-grade observability.

**Choose AutoGen when:** You want flexible multi-model conversations with minimal abstraction overhead.

## Conclusion

Multi-agent systems represent a paradigm shift from "prompt engineering" to "team engineering." CrewAI makes this accessible with a clean, intuitive API that lets you focus on defining **what** each agent should do rather than wrestling with infrastructure.

The key takeaways from this tutorial:

1. **Decompose by expertise** — Give each agent a narrow, well-defined role with a rich backstory. This produces better results than generic prompts.
2. **Design the data flow** — Use `context` to explicitly connect task outputs. Think of it as designing a pipeline, not a chat.
3. **Match models to tasks** — Expensive models for reasoning-heavy tasks; cheap models for formatting and structure.
4. **Iterate on task descriptions** — The quality of your output is directly proportional to the specificity of your task descriptions.
5. **Add guardrails** — `max_iter`, tool validation, and human-in-the-loop are essential for production deployments.

The code in this tutorial is production-ready. Clone it, adapt the agents to your domain, and deploy. The multi-agent pattern scales from simple two-agent workflows to complex teams of 10+ agents handling end-to-end business processes.

As the AI ecosystem matures in 2026, the ability to design and orchestrate multi-agent systems is becoming a core skill — as fundamental as knowing how to write a REST API was in 2015. Start building now.

*For more CrewAI examples and documentation, visit [docs.crewai.com](https://docs.crewai.com).*
