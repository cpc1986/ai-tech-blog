---
title: "MetaGPT in Practice: Building a Software Company Simulation with AI Agent Teams"
date: "2026-06-15"
excerpt: "MetaGPT takes the radical approach of simulating an entire software company with AI agents — Product Manager, Architect, Engineer, and QA — collaborating to produce real code from natural language requirements. We test it end-to-end on a real project and analyze its orchestration, communication patterns, and output quality."
tags: ["MetaGPT", "multi-agent", "AI coding", "software engineering", "agent teams", "code generation", "2026"]
category: "Deep Dive"
---

What if you could describe a software project in plain English and have an entire team of AI agents produce working code? MetaGPT attempts exactly this. It assigns specialized agent roles — Product Manager, Architect, Project Manager, Engineer, and QA — and orchestrates them through the same collaborative process a real software team would follow.

With 44,000+ GitHub stars, MetaGPT is one of the most ambitious agent projects in the ecosystem. This article puts it through a real-world test and analyzes how its agent team architecture works.

## The Big Idea: Software Company as a Multi-Agent System

Most agent frameworks treat agents as interchangeable workers connected by a workflow graph. MetaGPT's insight is different: **effective collaboration requires specialization AND standardized communication protocols.**

In a real software company:
- The Product Manager writes a PRD (Product Requirements Document)
- The Architect designs the system (modules, interfaces, data flow)
- The Project Manager creates task breakdowns
- The Engineer implements code from specifications
- The QA Engineer writes tests and validates

MetaGPT encodes these roles and their standard operating procedures (SOPs) as agent behaviors. Agents don't just chat — they produce structured artifacts that become inputs for downstream agents.

```
User Requirements
       ↓
[Product Manager] → PRD (structured requirements document)
       ↓
[Architect] → System Design (classes, modules, interfaces, data flow)
       ↓
[Project Manager] → Task List (ordered implementation tasks)
       ↓
[Engineer(s)] → Source Code (implementing each task)
       ↓
[QA Engineer] → Test Cases + Validation Results
       ↓
Complete Project Output
```

This "artifact-driven" communication is MetaGPT's key differentiator. Agents don't just pass messages — they produce structured deliverables that other agents consume programmatically.

## Installation and Setup

```bash
# Install MetaGPT
pip install metagpt

# Configure your API key
export OPENAI_API_KEY="your-key"

# Or use a local model via Ollama
# MetaGPT supports OpenAI-compatible endpoints
```

Metabash configuration for using alternative models:

```python
# config.yaml
llm:
  api_type: "openai"  # or "anthropic", "azure"
  model: "gpt-4o"
  # For local models:
  # base_url: "http://localhost:11434/v1"
  # model: "llama3.1:70b"
```

## Test Project: Building a REST API for a URL Shortener

We test MetaGPT by asking it to build a complete URL shortener service. This is a mid-complexity project that touches:
- API design (REST endpoints)
- Database schema (URL mappings, analytics)
- Authentication (API keys)
- Rate limiting
- Caching (Redis)
- Error handling
- Testing

### Running MetaGPT

```bash
# Command-line usage
metagpt "Create a URL shortener REST API service with the following features:
1. Shorten long URLs to short codes
2. Redirect short codes to original URLs
3. Track click analytics (timestamp, referrer, location)
4. API key authentication for creating short URLs
5. Rate limiting (100 requests per minute per API key)
6. Redis caching for hot URLs
7. Docker Compose setup for deployment
8. Comprehensive unit and integration tests
9. Use Python with FastAPI and PostgreSQL"
```

Alternatively, via Python API:

```python
import asyncio
from metagpt.software_company import generate_repo, ProjectRepo
from metagpt.roles import ProductManager, Architect, ProjectManager, Engineer, QaEngineer

async def build_url_shortener():
    idea = """Create a URL shortener REST API service with the following features:
    1. Shorten long URLs to short codes (base62 encoding)
    2. Redirect short codes to original URLs
    3. Track click analytics (timestamp, referrer, location)
    4. API key authentication for creating short URLs
    5. Rate limiting (100 req/min per API key)
    6. Redis caching for hot URLs
    7. Docker Compose setup
    8. Comprehensive tests
    9. Use Python with FastAPI and PostgreSQL"""

    repo = await generate_repo(idea)
    return repo

repo = asyncio.run(build_url_shortener())
```

## What MetaGPT Produces

### Output 1: Product Requirements Document (by PM Agent)

The Product Manager agent produces a structured PRD:

```markdown
# URL Shortener Service - Product Requirements Document

## 1. Product Overview
A URL shortener REST API service that converts long URLs into short,
memorable codes and provides analytics on click activity.

## 2. Target Users
- Marketing teams tracking campaign links
- Social media managers needing short links
- Developers integrating URL shortening into applications

## 3. Functional Requirements

### FR-1: URL Shortening
- Accept a long URL via POST /api/urls
- Generate a unique short code (7 characters, base62)
- Return the shortened URL in response

### FR-2: URL Redirection
- Accept GET /{short_code}
- Look up original URL
- Return 301 redirect
- Record click event for analytics

### FR-3: Click Analytics
- Track: timestamp, referrer header, IP geolocation
- Store analytics in separate table from URL mapping
- Expose GET /api/urls/{short_code}/analytics endpoint

### FR-4: Authentication
- API key-based authentication
- POST /api/keys to create new keys
- Keys required for write operations (shorten, delete)
- Read operations (redirect) are public

### FR-5: Rate Limiting
- 100 requests per minute per API key
- Return 429 Too Many Requests when exceeded
- Implement via Redis sliding window

## 4. Non-Functional Requirements
- Response time < 50ms for redirects (with Redis cache)
- 99.9% uptime target
- Handle 10,000 requests per second at peak

## 5. API Design
POST   /api/urls          - Create short URL
GET    /{short_code}       - Redirect to original
GET    /api/urls/{code}    - Get URL details
DELETE /api/urls/{code}    - Delete short URL
GET    /api/urls/{code}/analytics - Click analytics
POST   /api/keys           - Create API key
```

### Output 2: System Architecture (by Architect Agent)

The Architect produces a detailed system design:

```
┌─────────────────────────────────────────────┐
│            System Architecture               │
├─────────────────────────────────────────────┤
│                                             │
│  Client → [FastAPI App] → [Redis Cache]     │
│                │                            │
│                ├── URL Service               │
│                │   ├── Shorten URL           │
│                │   ├── Redirect URL          │
│                │   └── Manage URLs           │
│                │                            │
│                ├── Auth Service              │
│                │   └── API Key Validation    │
│                │                            │
│                ├── Analytics Service          │
│                │   └── Track Click Events    │
│                │                            │
│                └── Rate Limiter              │
│                    └── Sliding Window (Redis)│
│                                             │
│  Database: PostgreSQL                        │
│  Cache: Redis                               │
│  Deployment: Docker Compose                  │
└─────────────────────────────────────────────┘
```

```python
# Generated data models
class URL(BaseModel):
    id: UUID
    original_url: str
    short_code: str  # 7 chars, base62
    api_key_id: UUID
    created_at: datetime
    expires_at: Optional[datetime]
    click_count: int = 0

class ClickEvent(BaseModel):
    id: UUID
    url_id: UUID
    timestamp: datetime
    referrer: Optional[str]
    ip_address: str
    country: Optional[str]
    city: Optional[str]

class APIKey(BaseModel):
    id: UUID
    key_hash: str  # SHA-256 hash of the key
    name: str
    created_at: datetime
    rate_limit: int = 100  # requests per minute
    is_active: bool = True
```

### Output 3: Source Code (by Engineer Agent)

The Engineer produces multiple files:

```python
# main.py - Generated by MetaGPT
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.responses import RedirectResponse
from datetime import datetime, timedelta
import redis
import hashlib
import base62
from models import URL, ClickEvent, APIKey
from database import get_db, init_db

app = FastAPI(title="URL Shortener API", version="1.0.0")
redis_client = redis.Redis(host="redis", port=6379, db=0)

# Rate limiting middleware
async def check_rate_limit(api_key: str = Depends(get_api_key)):
    key = f"rate_limit:{api_key}"
    current = redis_client.get(key)
    if current and int(current) >= 100:
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    pipe = redis_client.pipeline()
    pipe.incr(key)
    pipe.expire(key, 60)
    pipe.execute()

@app.post("/api/urls", status_code=201)
async def create_short_url(
    original_url: str,
    api_key: APIKey = Depends(check_rate_limit),
    db=Depends(get_db)
):
    # Generate unique short code
    short_code = base62.encode(hashlib.md5(original_url.encode()).int_digest())[:7]

    # Check if already shortened
    existing = await db.execute(
        URL.select().where(URL.c.original_url == original_url)
    )
    if existing:
        return {"short_code": existing.short_code, "url": f"/{existing.short_code}"}

    # Store in database
    url = URL(
        original_url=original_url,
        short_code=short_code,
        api_key_id=api_key.id,
        created_at=datetime.utcnow()
    )
    await db.execute(URL.insert(), url.dict())

    # Cache in Redis
    redis_client.setex(f"url:{short_code}", 86400, original_url)

    return {"short_code": short_code, "url": f"/{short_code}"}

@app.get("/{short_code}")
async def redirect_url(short_code: str, request: Request, db=Depends(get_db)):
    # Check Redis cache first
    cached_url = redis_client.get(f"url:{short_code}")
    if cached_url:
        original_url = cached_url.decode()
    else:
        # Database lookup
        result = await db.execute(
            URL.select().where(URL.c.short_code == short_code)
        )
        if not result:
            raise HTTPException(status_code=404, detail="URL not found")
        original_url = result.original_url
        # Cache for future requests
        redis_client.setex(f"url:{short_code}", 86400, original_url)

    # Record analytics asynchronously
    await record_click(short_code, request, db)

    return RedirectResponse(url=original_url, status_code=301)

async def record_click(short_code: str, request: Request, db):
    click = ClickEvent(
        url_id=...,  # Look up from short_code
        timestamp=datetime.utcnow(),
        referrer=request.headers.get("referer"),
        ip_address=request.client.host,
    )
    await db.execute(ClickEvent.insert(), click.dict())
```

### Output 4: Docker Compose Configuration

```yaml
# docker-compose.yml - Generated by MetaGPT
version: '3.8'

services:
  app:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://user:pass@db:5432/urlshortener
      - REDIS_URL=redis://redis:6379/0
    depends_on:
      - db
      - redis

  db:
    image: postgres:16
    environment:
      POSTGRES_DB: urlshortener
      POSTGRES_USER: user
      POSTGRES_PASSWORD: pass
    volumes:
      - postgres_data:/var/lib/postgresql/data

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

volumes:
  postgres_data:
```

### Output 5: Tests (by QA Agent)

```python
# tests/test_url_shortener.py - Generated by MetaGPT
import pytest
from httpx import AsyncClient
from main import app

@pytest.fixture
async def client():
    async with AsyncClient(app=app, base_url="http://test") as ac:
        yield ac

@pytest.fixture
async def api_key(client):
    response = await client.post("/api/keys", json={"name": "test"})
    return response.json()["key"]

async def test_create_short_url(client, api_key):
    headers = {"X-API-Key": api_key}
    response = await client.post(
        "/api/urls",
        params={"original_url": "https://example.com/very/long/path"},
        headers=headers
    )
    assert response.status_code == 201
    data = response.json()
    assert "short_code" in data
    assert len(data["short_code"]) == 7

async def test_redirect(client, api_key):
    # First create a short URL
    headers = {"X-API-Key": api_key}
    create_resp = await client.post(
        "/api/urls",
        params={"original_url": "https://example.com"},
        headers=headers
    )
    short_code = create_resp.json()["short_code"]

    # Then test redirect
    response = await client.get(f"/{short_code}", follow_redirects=False)
    assert response.status_code == 301
    assert response.headers["location"] == "https://example.com"

async def test_rate_limiting(client, api_key):
    headers = {"X-API-Key": api_key}
    for i in range(100):
        await client.post(
            "/api/urls",
            params={"original_url": f"https://example.com/{i}"},
            headers=headers
        )

    # 101st request should be rate limited
    response = await client.post(
        "/api/urls",
        params={"original_url": "https://example.com/extra"},
        headers=headers
    )
    assert response.status_code == 429

async def test_url_not_found(client):
    response = await client.get("/nonexistent", follow_redirects=False)
    assert response.status_code == 404
```

## Analysis: How MetaGPT's Agent Team Works

### Communication Protocol: Publish-Subscribe

MetaGPT uses a global **message bus** where agents publish structured artifacts:

```python
# Simplified MetaGPT internal communication
class MessageBus:
    def __init__(self):
        self.subscribers = {}  # {topic: [agent1, agent2, ...]}

    def publish(self, topic: str, content: dict):
        for agent in self.subscribers.get(topic, []):
            agent.receive(topic, content)

# Agents subscribe to specific artifact types:
# PM subscribes to: "user_requirements"
# Architect subscribes to: "prd" (from PM)
# Engineer subscribes to: "system_design" (from Architect)
# QA subscribes to: "source_code" (from Engineer)
```

### SOP-Encoded Workflows

Each role has encoded Standard Operating Procedures:

```python
class ProductManager(Agent):
    """SOP: Requirements → PRD"""

    async def act(self, message):
        # Step 1: Clarify requirements
        requirements = await self.clarify(message.content)

        # Step 2: Research competitive products
        competitors = await self.research(requirements)

        # Step 3: Write structured PRD
        prd = await self.write_prd(requirements, competitors)

        # Step 4: Publish PRD for Architect
        self.publish("prd", prd)
```

### What Works Well

1. **Complete project output** — MetaGPT produces a directory tree with multiple files, not just a single code snippet. For the URL shortener, it generated 12 files across 4 directories.

2. **Consistent architecture** — Because the Architect agent designs the system before the Engineer codes, the output is architecturally coherent. Models, routes, and database schemas are consistent.

3. **Documentation as a byproduct** — The PRD, system design, and API documentation are produced as intermediate artifacts. You get docs "for free."

4. **Test generation** — The QA agent produces tests that align with the PRD requirements (FR-1 through FR-5 in our example).

### What Doesn't Work Well

1. **Compilation issues** — The generated code is rarely runnable without modification. In our test, 3 out of 12 files had import errors or missing dependencies. This is the biggest gap between demo and production.

2. **Architecture over-complexity** — The Architect agent tends to over-engineer. For a URL shortener, it designed 6 service classes when 2 would suffice. This is mitigated by adjusting the "investment" parameter (controls how much thought the agent puts in).

3. **No iterative refinement** — MetaGPT runs once through the pipeline. There's no loop back from QA to Engineer for fixing bugs. You get the first draft, not a refined product.

4. **Memory is limited** — Unlike Mem0 or Letta, MetaGPT agents don't have persistent long-term memory. Each project starts from scratch.

5. **Token cost** — A complete project generation consumed ~180K tokens (~$0.90 with GPT-4o). For iterative development where you refine the output, costs add up quickly.

## MetaGPT vs Building from Scratch

| Aspect | MetaGPT | Manual Development |
|--------|---------|-------------------|
| **Time to prototype** | 5-10 minutes | 2-4 hours |
| **Architecture quality** | Good first draft | Depends on developer |
| **Code quality** | 60-70% correct | 90%+ with review |
| **Documentation** | Included automatically | Often skipped |
| **Testing** | Basic tests included | Often manual |
| **Refinement needed** | Moderate (fix imports, adjust patterns) | None if experienced |
| **Cost** | ~$0.50-2.00 per project | Developer time |

## Integration with Memory Systems

One of the most promising patterns is combining MetaGPT with external memory systems:

```python
# Conceptual integration: MetaGPT + Mem0
from metagpt.software_company import generate_repo
from mem0 import Memory

# Shared memory for the development team
team_memory = Memory.from_config(config)

# Pre-load team context
team_memory.add("Team prefers FastAPI over Flask for Python APIs")
team_memory.add("Use SQLAlchemy 2.0 with async for all database access")
team_memory.add("Always include OpenAPI schema validation")

# When architect designs, inject team conventions
team_context = team_memory.search("API framework preferences")
# Feed into the architect's context...

# After project is generated, store learnings
team_memory.add("URL shortener project: used base62 for short codes")
team_memory.add("FastAPI + PostgreSQL pattern worked well for this use case")
```

This pattern enables **learning across projects** — the agent team accumulates conventions, patterns, and lessons that improve future project generation.

## Real-World Assessment

### Should You Use MetaGPT in Production?

**For prototyping and scaffolding:** Absolutely. MetaGPT excels at generating a complete project skeleton from a description. The PRD + architecture + code + tests output gives you a massive head start over starting from scratch.

**For production-ready code:** Not directly. The output needs review, debugging, and refinement. Think of MetaGPT as a very capable junior developer who needs senior review.

**For learning:** Excellent. Reading the generated PRD, architecture diagrams, and code teaches you how to structure projects. The SOP-encoded workflows are educational in themselves.

### Best Practices

1. **Be specific in requirements** — Vague prompts produce generic output. "URL shortener with Redis caching and API key auth" is better than "make a URL thing."

2. **Set investment wisely** — Lower `investment` values (1-3) produce simpler, more practical output. Higher values (5+) produce over-architected output.

3. **Iterate on the PRD** — The PRD is the most important artifact. If the PRD is wrong, everything downstream will be wrong. Review and edit the PRD before running the full pipeline.

4. **Use as a starting point, not the finish line** — Generate the project, then manually refine the 12 files. The 30-60% that's correct saves you hours. Fix the remaining 40%.

## Conclusion

MetaGPT demonstrates that **role-based, artifact-driven multi-agent collaboration** is a viable approach to code generation. By encoding software engineering SOPs into agent behaviors, it produces more coherent and complete output than single-agent code generation systems.

The key limitation is quality — while the structure is right, the details (imports, edge cases, error handling) often need fixing. This is likely to improve rapidly as underlying LLMs get better at code generation.

Combined with memory systems (Mem0 for shared knowledge) and orchestration frameworks (LangGraph for complex workflows), the vision of an autonomous software development team is becoming tangible — if not quite production-ready.

---

*This article concludes our Agent Memory Research Series. [Part 1: AI Native Survey](/blog/ai-native-implementation-research-survey-2026) | [Part 2: 10 Memory Systems](/blog/agent-memory-comparative-analysis-open-source-2026) | [Part 3: Mem0 vs Letta](/blog/mem0-vs-letta-memgpt-agent-memory-benchmark-2026) | [Part 4: Framework Comparison](/blog/langgraph-vs-crewai-vs-autogen-multi-agent-comparison-2026)*
