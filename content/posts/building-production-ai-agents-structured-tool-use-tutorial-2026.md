---
title: "Building Production-Ready AI Agents with Structured Tool Use: A Complete 2026 Tutorial"
date: "2026-07-19"
excerpt: "A hands-on tutorial for building production-ready AI agents with structured tool use in 2026. Learn how to implement function calling, manage tool schemas, handle errors, add observability, and deploy agents that reliably interact with real-world APIs using OpenAI, Anthropic, and open-source frameworks."
tags: ["AI agents", "tool use", "function calling", "OpenAI", "Anthropic", "structured output", "agent tutorial", "production AI", "2026"]
category: "Tutorials"
---

AI agents that can call external tools — APIs, databases, file systems — have moved from research demos to production systems in 2026. But the gap between a prototype agent that *sometimes* calls the right function and a production agent that reliably handles edge cases, validates arguments, and recovers from failures is enormous.

This tutorial walks through building a production-grade AI agent with structured tool use from scratch. We cover function calling schema design, argument validation, error recovery, multi-step orchestration, observability, and deployment. Every pattern shown here is battle-tested in real production systems.

## What You'll Build

By the end of this tutorial, you'll have a fully functional AI agent that:

- Connects to real APIs (weather, calendar, web search) via structured tool definitions
- Validates all function arguments before execution
- Recovers gracefully from API errors and malformed LLM outputs
- Logs every step for full observability and debugging
- Handles multi-step reasoning with tool chains
- Deploys as a FastAPI service with health checks and rate limiting

**Prerequisites**: Python 3.11+, an OpenAI API key (or Anthropic — we show both), and basic familiarity with async Python.

## Why Structured Tool Use Is Hard

Before diving into code, let's understand why most agent implementations break in production:

| Failure Mode | Root Cause | Frequency in Prod |
|-------------|-----------|-------------------|
| Wrong argument types | LLM generates string instead of int | Very common |
| Missing required fields | LLM omits optional-ish but required params | Common |
| Hallucinated tool names | LLM invents a tool that doesn't exist | Common |
| Invalid enum values | LLM picks value outside allowed set | Common |
| Circular tool calls | Agent calls same tool with same args repeatedly | Moderate |
| Non-terminating loops | Agent never reaches a "finish" state | Moderate |
| Nested JSON escaping | LLM double-escapes JSON in arguments | Rare but painful |

The core insight: **LLMs are stochastic, but tool interfaces are deterministic**. The bridge between these two worlds is rigorous schema validation and defensive programming.

## Step 1: Define Tool Schemas with Pydantic

The foundation of reliable tool use is a strong schema. We use Pydantic models because they give us both JSON Schema generation (for the LLM) and runtime validation (for safety).

```python
# tools/schemas.py
from pydantic import BaseModel, Field
from typing import Literal, Optional
from enum import Enum

class WeatherUnit(str, Enum):
    celsius = "celsius"
    fahrenheit = "fahrenheit"

class GetWeatherInput(BaseModel):
    """Get current weather for a location."""
    city: str = Field(
        ...,
        description="City name, e.g. 'San Francisco'",
        min_length=1,
        max_length=100,
    )
    country: Optional[str] = Field(
        None,
        description="ISO 3166-1 alpha-2 country code, e.g. 'US'",
        pattern=r"^[A-Z]{2}$",
    )
    unit: WeatherUnit = Field(
        WeatherUnit.celsius,
        description="Temperature unit",
    )

class SearchWebInput(BaseModel):
    """Search the web for information."""
    query: str = Field(
        ...,
        description="Search query, be specific for better results",
        min_length=3,
        max_length=500,
    )
    max_results: int = Field(
        5,
        description="Maximum number of results to return",
        ge=1,
        le=20,
    )

class CreateCalendarEventInput(BaseModel):
    """Create a calendar event."""
    title: str = Field(
        ...,
        description="Event title",
        min_length=1,
        max_length=200,
    )
    date: str = Field(
        ...,
        description="Event date in ISO 8601 format, e.g. '2026-07-20'",
        pattern=r"^\d{4}-\d{2}-\d{2}$",
    )
    time: str = Field(
        ...,
        description="Event time in 24h format, e.g. '14:30'",
        pattern=r"^\d{2}:\d{2}$",
    )
    duration_minutes: int = Field(
        60,
        description="Duration in minutes",
        ge=15,
        le=480,
    )
    description: Optional[str] = Field(
        None,
        description="Optional event description",
        max_length=2000,
    )
```

The key patterns here:
- **`Field(...)` with `description`**: Every field has a description the LLM sees in the schema
- **`min_length`/`max_length`/`ge`/`le`**: Constraints that prevent degenerate inputs
- **`pattern`**: Regex validation for formatted strings (dates, country codes, times)
- **`Enum` classes**: Restrict the LLM to valid choices
- **`Optional` with `None` default**: Truly optional parameters

## Step 2: Build the Tool Registry

Next, we need a registry that maps tool definitions to their implementations and handles validation:

```python
# tools/registry.py
import json
import inspect
from typing import Callable, Dict, Type, Any
from pydantic import BaseModel, ValidationError

class ToolRegistry:
    """Registry for managing agent tools with schema validation."""

    def __init__(self):
        self._tools: Dict[str, Dict[str, Any]] = {}

    def register(
        self,
        name: str,
        description: str,
        input_model: Type[BaseModel],
        handler: Callable,
    ):
        """Register a tool with its schema and handler."""
        # Generate JSON Schema from Pydantic model
        schema = input_model.model_json_schema()

        self._tools[name] = {
            "name": name,
            "description": description,
            "input_model": input_model,
            "handler": handler,
            "schema": schema,
        }

    def get_openai_tools(self) -> list[dict]:
        """Return tools in OpenAI function calling format."""
        return [
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool["description"],
                    "parameters": tool["schema"],
                },
            }
            for tool in self._tools.values()
        ]

    def get_anthropic_tools(self) -> list[dict]:
        """Return tools in Anthropic tool use format."""
        return [
            {
                "name": tool["name"],
                "description": tool["description"],
                "input_schema": tool["schema"],
            }
            for tool in self._tools.values()
        ]

    async def execute(self, name: str, arguments: dict) -> dict:
        """Execute a tool by name with validated arguments."""
        if name not in self._tools:
            return {
                "error": f"Unknown tool: {name}",
                "available_tools": list(self._tools.keys()),
            }

        tool = self._tools[name]
        input_model = tool["input_model"]
        handler = tool["handler"]

        # Validate arguments against schema
        try:
            validated = input_model.model_validate(arguments)
        except ValidationError as e:
            return {
                "error": f"Invalid arguments for {name}",
                "details": json.loads(e.json()),
                "received": arguments,
            }

        # Execute the handler
        try:
            if inspect.iscoroutinefunction(handler):
                result = await handler(validated)
            else:
                result = handler(validated)
            return {"result": result}
        except Exception as e:
            return {
                "error": f"Tool execution failed: {type(e).__name__}: {str(e)}",
                "tool": name,
                "arguments": validated.model_dump(),
            }

    def has_tool(self, name: str) -> bool:
        return name in self._tools
```

The `execute` method is the critical safety layer. It:
1. Rejects unknown tool names (no hallucinated tools)
2. Validates every argument using Pydantic (no wrong types or missing fields)
3. Catches all handler exceptions (no unhandled crashes)
4. Returns structured error messages the LLM can understand and recover from

## Step 3: Implement Tool Handlers

Now let's implement the actual tool handlers. These are the functions that interact with real APIs:

```python
# tools/handlers.py
import httpx
import os
from .schemas import GetWeatherInput, SearchWebInput, CreateCalendarEventInput

async def get_weather(input: GetWeatherInput) -> dict:
    """Fetch weather from Open-Meteo (free, no API key required)."""
    async with httpx.AsyncClient(timeout=10.0) as client:
        # First, geocode the city
        geo_url = "https://geocoding-api.open-meteo.com/v1/search"
        geo_params = {"name": input.city, "count": 1}
        if input.country:
            geo_params["country_code"] = input.country

        geo_resp = await client.get(geo_url, params=geo_params)
        geo_data = geo_resp.json()

        if not geo_data.get("results"):
            return {"error": f"City not found: {input.city}"}

        location = geo_data["results"][0]
        lat, lon = location["latitude"], location["longitude"]

        # Fetch weather
        weather_url = "https://api.open-meteo.com/v1/forecast"
        weather_params = {
            "latitude": lat,
            "longitude": lon,
            "current": "temperature_2m,relative_humidity_2m,wind_speed_10m,weather_code",
            "temperature_unit": input.unit.value,
        }

        weather_resp = await client.get(weather_url, params=weather_params)
        weather_data = weather_resp.json()

        current = weather_data.get("current", {})
        return {
            "city": input.city,
            "location": f"{lat:.2f}, {lon:.2f}",
            "temperature": f"{current.get('temperature_2m', 'N/A')}°{input.unit.value[0].upper()}",
            "humidity": f"{current.get('relative_humidity_2m', 'N/A')}%",
            "wind_speed": f"{current.get('wind_speed_10m', 'N/A')} km/h",
            "weather_code": current.get("weather_code", "N/A"),
        }


async def search_web(input: SearchWebInput) -> list[dict]:
    """Search the web using SearXNG or a similar API."""
    # In production, use a real search API (Brave, Tavily, etc.)
    search_url = os.getenv("SEARCH_API_URL", "https://searx.be/search")
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.get(
            search_url,
            params={"q": input.query, "format": "json", "limit": input.max_results},
        )
        data = resp.json()
        results = []
        for r in data.get("results", [])[:input.max_results]:
            results.append({
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "snippet": r.get("content", "")[:300],
            })
        return results


async def create_calendar_event(input: CreateCalendarEventInput) -> dict:
    """Create a calendar event (mock implementation)."""
    # In production, connect to Google Calendar, Outlook, or Cal.com API
    event_id = f"evt_{hash(input.title + input.date + input.time) % 100000:05d}"
    return {
        "event_id": event_id,
        "title": input.title,
        "date": input.date,
        "time": input.time,
        "duration_minutes": input.duration_minutes,
        "status": "created",
    }
```

## Step 4: Build the Agent Loop

This is the heart of the agent — the loop that orchestrates LLM reasoning with tool execution:

```python
# agent/core.py
import json
import uuid
from datetime import datetime
from typing import Optional

from openai import AsyncOpenAI

from tools.registry import ToolRegistry

MAX_ITERATIONS = 10  # Prevent infinite loops

class AgentRun:
    """Tracks a single agent execution run."""
    def __init__(self, run_id: str):
        self.run_id = run_id
        self.start_time = datetime.utcnow()
        self.steps: list[dict] = []
        self.total_tokens = 0
        self.tool_calls = 0

    def log_step(self, step: dict):
        step["timestamp"] = datetime.utcnow().isoformat()
        step["step_number"] = len(self.steps) + 1
        self.steps.append(step)

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "start_time": self.start_time.isoformat(),
            "steps": self.steps,
            "total_tokens": self.total_tokens,
            "tool_calls": self.tool_calls,
        }


class StructuredAgent:
    """Production-ready AI agent with structured tool use."""

    def __init__(
        self,
        registry: ToolRegistry,
        model: str = "gpt-4o-2024-08-06",
        system_prompt: Optional[str] = None,
    ):
        self.client = AsyncOpenAI()
        self.registry = registry
        self.model = model
        self.system_prompt = system_prompt or self._default_system_prompt()

    def _default_system_prompt(self) -> str:
        return """You are a helpful AI assistant with access to tools for looking up
information and taking actions. Follow these rules:

1. ALWAYS use tools when the user's request requires external data or actions.
2. NEVER fabricate information — if a tool returns an error, report it honestly.
3. If a tool call fails, try to recover by adjusting your approach.
4. Be concise in your responses.
5. If you are unsure about the right tool arguments, ask the user for clarification
   rather than guessing.
6. When multiple steps are needed, plan them out and execute sequentially."""

    async def run(self, user_message: str) -> dict:
        """Execute a complete agent run with the user's message."""
        run = AgentRun(run_id=str(uuid.uuid4()))

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_message},
        ]

        tools = self.registry.get_openai_tools()

        for iteration in range(MAX_ITERATIONS):
            run.log_step({
                "type": "llm_call",
                "iteration": iteration,
                "message_count": len(messages),
            })

            # Call the LLM
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                temperature=0.1,  # Low temp for reliable tool use
            )

            choice = response.choices[0]
            run.total_tokens += response.usage.total_tokens if response.usage else 0

            # If no tool calls, the agent is done — return its message
            if choice.finish_reason == "stop" or not choice.message.tool_calls:
                run.log_step({
                    "type": "final_response",
                    "content": choice.message.content,
                })
                return {
                    "response": choice.message.content,
                    "run": run.to_dict(),
                }

            # Process tool calls
            messages.append(choice.message)

            for tool_call in choice.message.tool_calls:
                tool_name = tool_call.function.name
                run.tool_calls += 1

                # Parse arguments defensively
                try:
                    tool_args = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError as e:
                    tool_result = {
                        "error": f"Invalid JSON in tool arguments: {e}",
                        "raw_arguments": tool_call.function.arguments[:500],
                    }
                else:
                    # Execute the tool with validation
                    tool_result = await self.registry.execute(tool_name, tool_args)

                run.log_step({
                    "type": "tool_call",
                    "tool": tool_name,
                    "arguments_size": len(tool_call.function.arguments),
                    "result_has_error": "error" in tool_result,
                })

                # Add tool result to conversation
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(tool_result, ensure_ascii=False),
                })

        # If we hit max iterations, return what we have
        run.log_step({"type": "max_iterations_reached"})
        return {
            "response": "I've reached the maximum number of reasoning steps. Please try a more specific request.",
            "run": run.to_dict(),
        }
```

The critical design decisions:

| Decision | Rationale |
|----------|-----------|
| `MAX_ITERATIONS = 10` | Prevents infinite loops; 10 is generous for most tasks |
| `temperature = 0.1` | Low randomness = more reliable function calling |
| `tool_choice = "auto"` | Let the model decide when to use tools |
| Defensive JSON parsing | LLMs occasionally produce malformed JSON |
| Error as tool result | Feed errors back to the LLM so it can self-correct |
| Structured run tracking | Every step is logged for debugging and auditing |

## Step 5: Add Observability with OpenTelemetry

Production agents need production observability. Here's how to instrument your agent with OpenTelemetry:

```python
# agent/observability.py
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

def setup_tracing(service_name: str = "ai-agent", endpoint: str = "http://localhost:4317"):
    """Configure OpenTelemetry tracing for the agent."""
    provider = TracerProvider()
    provider.add_span_processor(
        BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint))
    )
    trace.set_tracer_provider(provider)
    return trace.get_tracer(service_name)


# Integrate into the AgentRun class
class ObservableAgentRun(AgentRun):
    def __init__(self, run_id: str, tracer):
        super().__init__(run_id)
        self.tracer = tracer
        self._spans = []

    def start_tool_span(self, tool_name: str, arguments: dict):
        span = self.tracer.start_span(
            f"tool.{tool_name}",
            attributes={
                "tool.name": tool_name,
                "tool.arguments_size": len(str(arguments)),
                "agent.run_id": self.run_id,
            },
        )
        self._spans.append(span)
        return span

    def end_tool_span(self, span, result: dict):
        span.set_attribute("tool.result_has_error", "error" in result)
        span.set_attribute("tool.result_size", len(str(result)))
        span.end()
```

This gives you traces in Jaeger, Grafana Tempo, or any OTLP-compatible backend showing each tool call as a span with timing, success/failure, and run context.

## Step 6: Handle Multi-Step Tool Chains

Real tasks often require multiple tool calls in sequence. Here's a pattern for tool chains — for example, "Check the weather in Paris and schedule an outdoor event if it's sunny":

```python
# agent/chains.py
from agent.core import StructuredAgent
from tools.registry import ToolRegistry

async def run_tool_chain_example(registry: ToolRegistry):
    """Example: weather-dependent calendar scheduling."""
    agent = StructuredAgent(
        registry=registry,
        model="gpt-4o-2024-08-06",
        system_prompt="""You are an event planning assistant. When the user asks
to plan an outdoor event:
1. First check the weather using get_weather
2. If the weather is good (weather code 0-3, or temperature > 15°C),
   create the calendar event
3. If the weather is bad, suggest an indoor alternative or a different date
4. Always report the weather conditions before making the calendar event""",
    )

    result = await agent.run(
        "Plan a team picnic in Paris on July 20th at 12:00 for 2 hours. "
        "Only schedule it if the weather looks good."
    )

    return result
```

The LLM handles the conditional logic naturally because:
1. The system prompt gives it a clear decision procedure
2. Tool results are fed back into the conversation
3. It can call multiple tools across multiple iterations

## Step 7: Supporting Anthropic Claude

If you're using Claude instead of OpenAI, the tool format differs slightly. Here's the adapted agent:

```python
# agent/anthropic_agent.py
import anthropic
import json
from agent.core import AgentRun

MAX_ITERATIONS = 10

class AnthropicAgent:
    """Agent using Anthropic Claude with tool use."""

    def __init__(self, registry, model="claude-sonnet-4-20250514", system_prompt=None):
        self.client = anthropic.AsyncAnthropic()
        self.registry = registry
        self.model = model
        self.system_prompt = system_prompt or "You are a helpful AI assistant with access to tools. Use them when needed. Be concise."

    async def run(self, user_message: str) -> dict:
        run = AgentRun(run_id=str(uuid.uuid4()))

        messages = [{"role": "user", "content": user_message}]
        tools = self.registry.get_anthropic_tools()

        for iteration in range(MAX_ITERATIONS):
            response = await self.client.messages.create(
                model=self.model,
                system=self.system_prompt,
                messages=messages,
                tools=tools,
                max_tokens=4096,
            )

            # Track input tokens
            run.total_tokens += response.usage.input_tokens + response.usage.output_tokens

            # Process response content blocks
            assistant_content = response.content
            tool_use_blocks = [b for b in assistant_content if b.type == "tool_use"]
            text_blocks = [b for b in assistant_content if b.type == "text"]

            # If no tool use, we're done
            if not tool_use_blocks:
                final_text = "\n".join(b.text for b in text_blocks)
                return {"response": final_text, "run": run.to_dict()}

            # Add assistant message with all content blocks
            messages.append({"role": "assistant", "content": assistant_content})

            # Execute each tool call
            tool_results = []
            for tool_block in tool_use_blocks:
                tool_name = tool_block.name
                tool_args = tool_block.input  # Already parsed by SDK!
                run.tool_calls += 1

                tool_result = await self.registry.execute(tool_name, tool_args)

                run.log_step({
                    "type": "tool_call",
                    "tool": tool_name,
                    "result_has_error": "error" in tool_result,
                })

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_block.id,
                    "content": json.dumps(tool_result, ensure_ascii=False),
                })

            # Add tool results to messages
            messages.append({"role": "user", "content": tool_results})

        return {
            "response": "Maximum reasoning steps reached.",
            "run": run.to_dict(),
        }
```

Key difference with Anthropic: the SDK already parses JSON arguments into a dict via `tool_block.input`, so you skip the manual JSON parsing step. However, you still need Pydantic validation in `registry.execute()`.

## Step 8: Deploy as a FastAPI Service

Finally, let's wrap everything in a production-ready FastAPI service:

```python
# main.py
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from slowapi import Limiter
from slowapi.util import get_remote_address

from agent.core import StructuredAgent
from tools.registry import ToolRegistry
from tools.handlers import get_weather, search_web, create_calendar_event
from tools.schemas import GetWeatherInput, SearchWebInput, CreateCalendarEventInput
from agent.observability import setup_tracing

# --- App Setup ---

def create_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(
        name="get_weather",
        description="Get current weather conditions for any city worldwide",
        input_model=GetWeatherInput,
        handler=get_weather,
    )
    registry.register(
        name="search_web",
        description="Search the web for information on any topic",
        input_model=SearchWebInput,
        handler=search_web,
    )
    registry.register(
        name="create_calendar_event",
        description="Create a new calendar event with title, date, time, and duration",
        input_model=CreateCalendarEventInput,
        handler=create_calendar_event,
    )
    return registry


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    app.state.registry = create_registry()
    tracer = setup_tracing(
        service_name="ai-agent",
        endpoint=os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317"),
    )
    app.state.tracer = tracer
    yield
    # Shutdown — cleanup if needed

app = FastAPI(
    title="AI Agent API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

limiter = Limiter(key_func=get_remote_address)


# --- Request/Response Models ---

class AgentRequest(BaseModel):
    message: str
    model: str = "gpt-4o-2024-08-06"

class AgentResponse(BaseModel):
    response: str
    run_id: str
    tool_calls: int
    total_tokens: int


# --- API Key Check ---

async def verify_api_key(x_api_key: str = Header(...)):
    expected = os.getenv("AGENT_API_KEY")
    if expected and x_api_key != expected:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return x_api_key


# --- Endpoints ---

@app.get("/health")
async def health():
    return {"status": "healthy", "tools": len(app.state.registry._tools)}


@app.post("/chat", response_model=AgentResponse)
@limiter.limit("20/minute")
async def chat(
    request: AgentRequest,
    api_key: str = Depends(verify_api_key),
):
    agent = StructuredAgent(
        registry=app.state.registry,
        model=request.model,
    )

    result = await agent.run(request.message)

    return AgentResponse(
        response=result["response"],
        run_id=result["run"]["run_id"],
        tool_calls=result["run"]["tool_calls"],
        total_tokens=result["run"]["total_tokens"],
    )


@app.get("/tools")
async def list_tools():
    return {"tools": app.state.registry.get_openai_tools()}
```

Run it:

```bash
# Install dependencies
pip install fastapi uvicorn openai anthropic pydantic httpx slowapi opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp

# Set environment variables
export OPENAI_API_KEY="sk-..."
export AGENT_API_KEY="your-secret-key"

# Start the server
uvicorn main:app --host 0.0.0.0 --port 8000
```

Test it:

```bash
# Health check
curl http://localhost:8000/health

# Chat with the agent
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-secret-key" \
  -d '{"message": "What is the weather in Tokyo right now?"}'
```

## Production Deployment Checklist

Before deploying to production, ensure you've covered these critical areas:

| Area | Implementation | Status |
|------|---------------|--------|
| **Input validation** | Pydantic schemas on all tool inputs | ✅ Built in |
| **Error recovery** | Errors fed back to LLM as tool results | ✅ Built in |
| **Loop prevention** | MAX_ITERATIONS cap | ✅ Built in |
| **Observability** | OpenTelemetry traces for every tool call | ✅ Built in |
| **Rate limiting** | SlowAPI rate limiter on endpoints | ✅ Built in |
| **Authentication** | API key header validation | ✅ Built in |
| **Timeouts** | httpx client timeouts on all API calls | ✅ Built in |
| **Logging** | Structured AgentRun tracking | ✅ Built in |
| **Graceful degradation** | Tool errors don't crash the agent | ✅ Built in |
| **Secrets management** | Environment variables, not hardcoded | ✅ Recommended |
| **Load testing** | k6 / Locust for concurrent users | ⬜ Before launch |
| **Cost monitoring** | Token usage tracking per run | ⬜ Before launch |
| **Persistence** | Store runs in Postgres/DynamoDB | ⬜ Before launch |
| **Streaming responses** | SSE for real-time output | ⬜ Nice to have |

## Common Pitfalls and Solutions

### Pitfall 1: The LLM Passes a String Where an Integer Is Expected

Even with schema constraints, LLMs sometimes send `"5"` instead of `5`. Pydantic handles this automatically for coercible types, but for strict validation:

```python
from pydantic import StrictInt, StrictStr

class SafeInput(BaseModel):
    count: StrictInt    # Will reject "5"
    name: StrictStr     # Will reject 123
```

### Pitfall 2: The Agent Gets Stuck in a Loop

If the agent keeps calling the same tool with the same arguments, track tool call history:

```python
# In the agent loop, before executing:
call_key = f"{tool_name}:{json.dumps(tool_args, sort_keys=True)}"
if call_key in recent_calls:
    recent_calls[call_key] += 1
    if recent_calls[call_key] >= 3:
        tool_result = {
            "error": f"You've called {tool_name} with the same arguments 3 times. "
                     "This suggests a loop. Try a different approach or tell the user "
                     "you cannot complete this task."
        }
else:
    recent_calls[call_key] = 1
```

### Pitfall 3: JSON Escaping in Nested Arguments

When a tool argument itself contains JSON (e.g., a configuration payload), the LLM may double-escape it. Fix this in the schema:

```python
class APICallInput(BaseModel):
    endpoint: str
    method: Literal["GET", "POST", "PUT", "DELETE"]
    body: Optional[str] = Field(
        None,
        description="JSON string of the request body. Must be a valid JSON string.",
    )

# In the handler, parse the body:
async def api_call_handler(input: APICallInput) -> dict:
    if input.body:
        try:
            body = json.loads(input.body)
        except json.JSONDecodeError:
            return {"error": "Invalid JSON in request body"}
```

## Conclusion

Building production-ready AI agents with structured tool use requires bridging the gap between stochastic LLM outputs and deterministic API interfaces. The key principles are:

1. **Strong schemas with Pydantic** — both for generating tool definitions the LLM can understand and for validating LLM outputs at runtime
2. **Defensive execution** — every tool call is wrapped in try/except, and errors are fed back to the LLM for self-correction
3. **Loop prevention** — iteration caps and duplicate call detection prevent runaway agents
4. **Full observability** — structured logging and OpenTelemetry tracing make debugging possible
5. **Production-grade API** — rate limiting, authentication, and health checks make the agent deployable

The patterns in this tutorial work with any LLM provider that supports function calling (OpenAI, Anthropic, Google Gemini, Mistral, and open-source models via vLLM or Ollama). The `ToolRegistry` abstraction means you can swap providers without changing your tool implementations.

The complete code from this tutorial is available with additional examples for streaming, persistent memory, and multi-agent coordination. Start with the single-agent pattern here, add complexity incrementally, and always validate the LLM's outputs before executing them.

---

*Found this tutorial helpful? Bookmark it and check back for updates as the tool use ecosystem continues to evolve rapidly in 2026.*
