---
title: "Building LLM-Powered Structured Data Extraction Pipelines: A Practical Tutorial for 2026"
date: "2026-07-16"
excerpt: "A hands-on tutorial for building production-grade structured data extraction pipelines using LLMs. Learn how to use Instructor, Outlines, DSPy, and LangChain to extract structured JSON from unstructured documents with high accuracy, validation, and retry logic."
tags: ["structured extraction", "LLM", "Instructor", "Outlines", "DSPy", "data pipeline", "ETL", "Pydantic", "JSON schema", "2026"]
category: "Tutorials"
---

Extracting structured data from unstructured text is one of the highest-ROI applications of large language models in production. Whether you're parsing invoices, normalizing medical records, extracting entity relationships from legal contracts, or turning messy CRM notes into structured databases — LLM-powered extraction pipelines have become the go-to solution in 2026.

Yet most teams stumble when moving from a demo that works on three examples to a pipeline that reliably processes 100,000 documents. This tutorial walks you through building a production-grade structured extraction pipeline from scratch, covering schema design, framework selection, validation strategies, error handling, and performance optimization.

## The Structured Extraction Landscape in 2026

Before diving into code, let's understand the current tooling landscape. The key question is: **how do you force an LLM to output valid JSON conforming to your schema?** Several approaches have emerged:

| Approach | Mechanism | Frameworks | Structured Output Quality | Speed |
|----------|-----------|------------|--------------------------|-------|
| **Prompt + JSON parse** | Instruction in prompt, parse output | Raw API calls | Low — frequent format errors | Fast |
| **Tool/Function calling** | Native API function calling | OpenAI, Anthropic, Gemini APIs | Medium — schema support limited | Fast |
| **Constrained decoding** | Grammar-based token filtering | Outlines, lm-format-enforcer | High — guaranteed valid | Slower |
| **Pydantic + retry** | Validate with Pydantic, re-prompt on failure | Instructor, Marvin, LangChain | High — converges with retries | Variable |
| **Hybrid (constrained + validate)** | Grammar filtering + Pydantic validation | Instructor + Outlines | Very high — both guarantees | Moderate |

For production pipelines, the **hybrid approach** gives you the best of both worlds: grammar-constrained decoding ensures structurally valid JSON, and Pydantic validation ensures semantic correctness (e.g., dates are real dates, enums match allowed values).

## Project Setup

Let's build a pipeline that extracts structured product information from e-commerce listing descriptions. Create a new project:

```bash
mkdir llm-extraction-pipeline && cd llm-extraction-pipeline
python3 -m venv .venv
source .venv/bin/activate
pip install instructor pydantic openai outlines dspy-ai langchain-core tqdm rich
```

Our test data consists of messy product descriptions like:

```python
SAMPLE_DATA = [
    {
        "id": "prod_001",
        "text": """
        Apple MacBook Pro 16-inch M4 Max
        Brand new 2025 model, Space Black finish. 48GB unified memory,
        1TB SSD. Comes with 140W USB-C power adapter. Retail price $3499
        but we're selling for $2899. SKU: APPL-MBP16-M4MAX-1TB.
        Ships from warehouse in CA. Condition: Sealed box.
        """
    },
    {
        "id": "prod_002",
        "text": """
        Sony WH-1000XM6 wireless noise cancelling headphones
        Color: Midnight Blue. 40hr battery. USB-C charging + 3.5mm cable included.
        List: $449.99 Our Price: $359.99 you save $90!!
        Model# WH1000XM6/B. Item condition: open box, like new, all accessories present.
        """
    },
    {
        "id": "prod_003",
        "text": """
        Samsung 65" QN90D Neo QLED 4K Smart TV (2025)
        pays $1,797. Was $2,199. Model QN65QN90DAFXZA.
        Dolby Atmos, Object Tracking Sound+, Gaming Hub.
        Condition: refurbished by manufacturer. 90 day warranty.
        Comes with stand + wall mount kit. Free shipping.
        """
    }
]
```

## Step 1: Define Your Schema with Pydantic

The foundation of any extraction pipeline is a well-defined schema. Pydantic v2 gives you validation, serialization, and JSON Schema generation in one package:

```python
from pydantic import BaseModel, Field, field_validator
from typing import Optional
from enum import Enum


class ProductCondition(str, Enum):
    NEW = "new"
    OPEN_BOX = "open_box"
    REFURBISHED = "refurbished"
    USED = "used"
    UNKNOWN = "unknown"


class PriceInfo(BaseModel):
    current_price: float = Field(
        ..., description="The current selling price in USD"
    )
    original_price: Optional[float] = Field(
        None, description="The original/retail/list price in USD"
    )
    discount_percentage: Optional[float] = Field(
        None, description="Discount percentage if mentioned (0-100)"
    )

    @field_validator("current_price", "original_price")
    @classmethod
    def price_must_be_positive(cls, v):
        if v is not None and v <= 0:
            raise ValueError("Price must be positive")
        return v

    @field_validator("discount_percentage")
    @classmethod
    def discount_in_range(cls, v):
        if v is not None and not (0 <= v <= 100):
            raise ValueError("Discount must be between 0 and 100")
        return v


class ProductInfo(BaseModel):
    """Extract structured product information from an e-commerce listing."""

    brand: str = Field(
        ..., description="Brand or manufacturer name (e.g., 'Apple', 'Sony')"
    )
    model_name: str = Field(
        ..., description="Full product model name"
    )
    model_number: Optional[str] = Field(
        None, description="Manufacturer model number or SKU"
    )
    condition: ProductCondition = Field(
        ..., description="Condition of the item"
    )
    pricing: PriceInfo = Field(
        ..., description="Pricing information"
    )
    key_features: list[str] = Field(
        default_factory=list,
        description="List of key features or specifications mentioned",
        max_length=10,
    )
    included_accessories: list[str] = Field(
        default_factory=list,
        description="Items included in the package"
    )
    color: Optional[str] = Field(
        None, description="Product color if specified"
    )
    warranty_info: Optional[str] = Field(
        None, description="Warranty information if mentioned"
    )
```

Key design principles for extraction schemas:

1. **Use `Field(description=...)` on every field** — the LLM reads these as instructions
2. **Use Enums for categorical fields** — constrains output and reduces ambiguity
3. **Use `Optional` for fields that may not be present** — prevents hallucinated fill-ins
4. **Add validators for domain constraints** — catches semantic errors that schema alone can't
5. **Use nested models for logical groupings** — improves LLM comprehension of field relationships

## Step 2: Build the Extraction Pipeline with Instructor

[Instructor](https://github.com/jxnl/instructor) is the most popular library for structured LLM output in Python. It patches the OpenAI client to add Pydantic model validation, automatic retries, and streaming support.

```python
import instructor
from openai import OpenAI
from rich import print as rprint

# Create the Instructor-patched client
client = instructor.from_openai(OpenAI())


def extract_product_info(
    text: str,
    model: str = "gpt-4o-2024-08-06",
    max_retries: int = 3,
) -> ProductInfo:
    """Extract structured product info from raw text using Instructor."""

    response = client.chat.completions.create(
        model=model,
        response_model=ProductInfo,
        max_retries=max_retries,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a precise data extraction assistant. "
                    "Extract product information from e-commerce listings. "
                    "Only include information explicitly stated in the text. "
                    "If a field is not mentioned, leave it null/empty. "
                    "Do not infer or hallucinate values."
                ),
            },
            {
                "role": "user",
                "content": f"Extract product info from this listing:\n\n{text}",
            },
        ],
    )
    return response


# Test on our sample data
for item in SAMPLE_DATA:
    result = extract_product_info(item["text"])
    print(f"\n=== {item['id']} ===")
    rprint(result.model_dump_json(indent=2))
```

The output for the MacBook listing would look like:

```json
{
  "brand": "Apple",
  "model_name": "MacBook Pro 16-inch M4 Max",
  "model_number": "APPL-MBP16-M4MAX-1TB",
  "condition": "new",
  "pricing": {
    "current_price": 2899.0,
    "original_price": 3499.0,
    "discount_percentage": 17.15
  },
  "key_features": [
    "16-inch display",
    "M4 Max chip",
    "48GB unified memory",
    "1TB SSD",
    "Space Black finish",
    "2025 model"
  ],
  "included_accessories": ["140W USB-C power adapter"],
  "color": "Space Black",
  "warranty_info": null
}
```

### How Instructor's Retry Loop Works

When the LLM produces output that fails Pydantic validation, Instructor automatically re-prompts with the error message:

```
Attempt 1: LLM outputs discount_percentage: 120 → Validation Error
Attempt 2: Instructor sends back "discount must be 0-100" → LLM corrects to 17.15 ✓
```

This retry loop is what makes Instructor so reliable for production use. You can monitor retries with a hook:

```python
def log_retry(details):
    print(f"Retry #{details['attempt']}: {details['error']}")

response = client.chat.completions.create(
    model="gpt-4o-2024-08-06",
    response_model=ProductInfo,
    max_retries=3,
    messages=[...],
    context={"max_tokens": 2000},
    # The validation_context is passed to Pydantic validators
)
```

## Step 3: Add Constrained Decoding with Outlines

While Instructor handles semantic validation, it doesn't guarantee structurally valid JSON on the first try. [Outlines](https://github.com/dottxt-ai/outlines) uses finite-state automata to constrain the LLM's output tokens so that only valid JSON matching your schema is ever generated.

```python
import outlines

# Load a local model with Outlines
model = outlines.models.transformers(
    "Qwen/Qwen2.5-7B-Instruct",
    device="auto",
)

# Create a generator constrained to your Pydantic model
generator = outlines.generate.json(model, ProductInfo)


def extract_with_outlines(text: str) -> ProductInfo:
    """Extract using locally-constrained generation."""
    prompt = f"""Extract product information from this e-commerce listing.
Only include information explicitly stated.

Listing:
{text}"""

    result = generator(prompt)
    return result
```

Outlines guarantees that every token produced is JSON-valid according to your schema. The tradeoff is that local inference is slower than API calls, and only certain model architectures are supported.

### When to Use Each Approach

| Criterion | Instructor (API) | Outlines (Local) | Hybrid |
|-----------|------------------|-------------------|--------|
| **Output guarantee** | Valid after retries | Always valid JSON | Always valid + semantic |
| **Latency per doc** | 1–3s | 5–15s (GPU-dependent) | 5–15s |
| **Cost per 1K docs** | ~$15–50 | Free (compute only) | Free (compute only) |
| **Model quality** | GPT-4o / Claude grade | Depends on local model | Best of both |
| **Privacy** | Data sent to API | Fully local | Fully local |
| **Setup complexity** | Low | Medium–High | High |

**Recommendation:** Use Instructor + API models for accuracy-critical extraction where data privacy allows local processing. Add Outlines when you need guaranteed JSON or must keep data on-premises.

## Step 4: Batch Processing with Error Handling

Production pipelines need robust batch processing. Here's a production-grade batch extractor:

```python
import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from tqdm.asyncio import tqdm_asyncio


@dataclass
class ExtractionResult:
    doc_id: str
    data: ProductInfo | None = None
    error: str | None = None
    attempts: int = 0
    latency_ms: float = 0.0
    timestamp: str = field(
        default_factory=lambda: datetime.utcnow().isoformat()
    )


async def extract_single(
    doc: dict,
    semaphore: asyncio.Semaphore,
    max_retries: int = 3,
) -> ExtractionResult:
    """Extract from a single document with concurrency control."""
    async with semaphore:
        start = asyncio.get_event_loop().time()
        attempts = 0

        try:
            # Instructor supports async natively
            async_client = instructor.from_openai(
                AsyncOpenAI(), mode=instructor.Mode.JSON_SCHEMA
            )

            for attempt in range(1, max_retries + 1):
                attempts = attempt
                try:
                    result = await async_client.chat.completions.create(
                        model="gpt-4o-2024-08-06",
                        response_model=ProductInfo,
                        max_retries=0,  # We handle retries ourselves
                        messages=[
                            {
                                "role": "system",
                                "content": (
                                    "Extract product information. "
                                    "Only include explicitly stated facts."
                                ),
                            },
                            {
                                "role": "user",
                                "content": doc["text"],
                            },
                        ],
                    )
                    elapsed = (asyncio.get_event_loop().time() - start) * 1000
                    return ExtractionResult(
                        doc_id=doc["id"],
                        data=result,
                        attempts=attempts,
                        latency_ms=elapsed,
                    )
                except instructor.exceptions.IncompleteOutput as e:
                    if attempt == max_retries:
                        raise
                    await asyncio.sleep(0.5 * attempt)  # Exponential backoff

        except Exception as e:
            elapsed = (asyncio.get_event_loop().time() - start) * 1000
            return ExtractionResult(
                doc_id=doc["id"],
                error=str(e),
                attempts=attempts,
                latency_ms=elapsed,
            )


async def batch_extract(
    documents: list[dict],
    concurrency: int = 10,
    checkpoint_path: str | None = None,
) -> list[ExtractionResult]:
    """Process a batch of documents with checkpointing."""

    # Resume from checkpoint if exists
    completed_ids: set[str] = set()
    results: list[ExtractionResult] = []

    if checkpoint_path and Path(checkpoint_path).exists():
        with open(checkpoint_path) as f:
            for line in f:
                r = ExtractionResult(**json.loads(line))
                completed_ids.add(r.doc_id)
                results.append(r)

    # Filter to only unprocessed documents
    pending = [d for d in documents if d["id"] not in completed_ids]

    semaphore = asyncio.Semaphore(concurrency)
    tasks = [extract_single(doc, semaphore) for doc in pending]

    # Process with progress bar and checkpoint saving
    cp_file = None
    if checkpoint_path:
        cp_file = open(checkpoint_path, "a")

    try:
        for coro in tqdm_asyncio.as_completed(
            tasks, total=len(tasks), desc="Extracting"
        ):
            result = await coro
            results.append(result)

            if cp_file:
                cp_file.write(
                    json.dumps(result.__dict__, default=str) + "\n"
                )
                cp_file.flush()
    finally:
        if cp_file:
            cp_file.close()

    return results


# Run the batch extraction
async def main():
    # Load your documents (could be from a file, database, etc.)
    documents = SAMPLE_DATA  # Replace with your actual data source

    results = await batch_extract(
        documents,
        concurrency=5,
        checkpoint_path="extraction_checkpoint.jsonl",
    )

    # Print summary
    success = sum(1 for r in results if r.data is not None)
    failed = sum(1 for r in results if r.error is not None)
    avg_latency = (
        sum(r.latency_ms for r in results) / len(results)
        if results
        else 0
    )

    print(f"\nResults: {success} success, {failed} failed")
    print(f"Average latency: {avg_latency:.0f}ms")

    for r in results:
        if r.data:
            print(f"\n{r.doc_id}: {r.data.brand} {r.data.model_name}")
            print(f"  Price: ${r.data.pricing.current_price}")
        else:
            print(f"\n{r.doc_id}: ERROR - {r.error}")


asyncio.run(main())
```

## Step 5: Improve Accuracy with DSPy Optimization

Raw prompting often leaves 10–20% accuracy on the table. [DSPy](https://github.com/stanfordnlp/dspy) lets you programmatically optimize your extraction prompts using labeled examples:

```python
import dspy


# Define the DSPy signature for extraction
class ProductExtraction(dspy.Signature):
    """Extract structured product information from an e-commerce listing.
    Only include information that is explicitly stated in the text."""

    listing_text: str = dspy.InputField(desc="Raw e-commerce listing text")
    brand: str = dspy.OutputField(desc="Brand/manufacturer name")
    model_name: str = dspy.OutputField(desc="Full model name")
    model_number: str = dspy.OutputField(desc="Model number or SKU, or 'N/A'")
    condition: str = dspy.OutputField(
        desc="One of: new, open_box, refurbished, used, unknown"
    )
    current_price: float = dspy.OutputField(desc="Current selling price in USD")
    original_price: float = dspy.OutputField(
        desc="Original/list price in USD, or 0 if not mentioned"
    )
    key_features: str = dspy.OutputField(
        desc="Comma-separated list of key features"
    )


# Create a module with chain-of-thought
class ExtractorModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.extractor = dspy.ChainOfThought(ProductExtraction)

    def forward(self, listing_text):
        return self.extractor(listing_text=listing_text)


# Define a metric for evaluation
def extraction_accuracy(example, prediction, trace=None):
    """Simple exact-match metric on key fields."""
    score = 0.0
    total = 0

    for field in ["brand", "model_name", "condition"]:
        total += 1
        if getattr(example, field, "").lower() == getattr(
            prediction, field, ""
        ).lower():
            score += 1

    # Price within 5% tolerance
    total += 1
    if example.current_price > 0 and prediction.current_price > 0:
        if abs(example.current_price - prediction.current_price) / example.current_price < 0.05:
            score += 1

    return score / total if total > 0 else 0


# Set up DSPy with your LLM
lm = dspy.LM("openai/gpt-4o-2024-08-06")
dspy.configure(lm=lm)

# Optimize with BootstrapFewShot
from dspy.teleprompt import BootstrapFewShot

optimizer = BootstrapFewShot(metric=extraction_accuracy, max_bootstrapped_demos=4)

# You need labeled examples for optimization
trainset = [
    dspy.Example(
        listing_text=SAMPLE_DATA[0]["text"],
        brand="Apple",
        model_name="MacBook Pro 16-inch M4 Max",
        condition="new",
        current_price=2899.0,
        original_price=3499.0,
    ).with_inputs("listing_text"),
    # Add more labeled examples...
]

optimized_module = optimizer.compile(ExtractorModule(), trainset=trainset)

# Use the optimized extractor
result = optimized_module(listing_text=SAMPLE_DATA[1]["text"])
print(f"Extracted: {result.brand} {result.model_name} - ${result.current_price}")
```

DSPy optimization typically improves extraction accuracy by 8–15% on challenging fields by automatically discovering the best few-shot examples and chain-of-thought patterns.

## Step 6: Validation and Quality Gates

Even with the best prompts, no pipeline achieves 100% accuracy. Production systems need quality gates:

```python
from pydantic import model_validator
import statistics


class ProductInfoV2(ProductInfo):
    """Extended schema with cross-field validation rules."""

    @model_validator(mode="after")
    def validate_price_consistency(self):
        """Check that current price makes sense relative to original."""
        if self.pricing.original_price and self.pricing.current_price:
            if self.pricing.current_price > self.pricing.original_price * 1.1:
                raise ValueError(
                    f"Current price (${self.pricing.current_price}) exceeds "
                    f"original price (${self.pricing.original_price}) by >10%. "
                    f"Likely extraction error — prices may be swapped."
                )
        return self


class ExtractionQualityGate:
    """Evaluate extraction quality across a batch of results."""

    def __init__(self, confidence_threshold: float = 0.7):
        self.confidence_threshold = confidence_threshold

    def check_field_completeness(
        self, result: ProductInfo
    ) -> dict[str, bool]:
        """Check which fields are populated vs. null/empty."""
        data = result.model_dump()
        return {
            k: v is not None and v != "" and v != []
            for k, v in data.items()
        }

    def compute_completeness_score(self, result: ProductInfo) -> float:
        """Fraction of fields that are populated."""
        checks = self.check_field_completeness(result)
        populated = sum(1 for v in checks.values() if v)
        return populated / len(checks) if checks else 0.0

    def flag_for_review(
        self, result: ExtractionResult
    ) -> tuple[bool, list[str]]:
        """Determine if a result needs manual review."""
        if result.error:
            return True, [f"Extraction error: {result.error}"]

        if result.data is None:
            return True, ["No data extracted"]

        issues = []
        completeness = self.compute_completeness_score(result.data)

        if completeness < self.confidence_threshold:
            issues.append(
                f"Low completeness: {completeness:.0%} of fields populated"
            )

        # Check for suspicious extractions
        if result.data.pricing.current_price == 0:
            issues.append("Current price is $0 — likely extraction failure")

        if not result.data.brand or len(result.data.brand) < 2:
            issues.append("Brand name seems incomplete")

        if result.data.condition == ProductCondition.UNKNOWN:
            issues.append("Condition could not be determined")

        return len(issues) > 0, issues

    def batch_report(
        self, results: list[ExtractionResult]
    ) -> dict:
        """Generate a quality report for a batch."""
        reviewed = [self.flag_for_review(r) for r in results]
        completeness_scores = [
            self.compute_completeness_score(r.data)
            for r in results
            if r.data
        ]

        return {
            "total": len(results),
            "success": sum(1 for r in results if r.data),
            "failed": sum(1 for r in results if r.error),
            "flagged_for_review": sum(1 for f, _ in reviewed if f),
            "avg_completeness": (
                statistics.mean(completeness_scores)
                if completeness_scores
                else 0
            ),
            "avg_latency_ms": (
                statistics.mean([r.latency_ms for r in results])
                if results
                else 0
            ),
            "common_issues": self._common_issues(reviewed),
        }

    @staticmethod
    def _common_issues(
        reviewed: list[tuple[bool, list[str]]]
    ) -> dict[str, int]:
        from collections import Counter

        all_issues = [issue for _, issues in reviewed for issue in issues]
        return dict(Counter(all_issues).most_common(10))


# Usage
gate = ExtractionQualityGate(confidence_threshold=0.75)
# ... after batch extraction ...
# report = gate.batch_report(results)
# print(json.dumps(report, indent=2))
```

## Step 7: Cost and Performance Optimization

At scale, extraction costs add up fast. Here are the key levers:

### Model Selection Strategy

Not every document needs GPT-4o. Use a tiered approach:

```python
import re


def classify_complexity(text: str) -> str:
    """Classify document complexity to route to appropriate model."""
    word_count = len(text.split())
    has_tables = bool(re.search(r"\|.*\|", text))
    has_multiple_products = len(re.findall(r"\$\d+", text)) > 2
    has_ambiguous_condition = bool(
        re.search(r"like new|minor|barely used|good condition", text, re.I)
    )

    if word_count < 50 and not has_ambiguous_condition:
        return "simple"  # → Use GPT-4o-mini ($0.15/1M input tokens)
    elif has_tables or has_multiple_products:
        return "complex"  # → Use GPT-4o ($2.50/1M input tokens)
    else:
        return "medium"  # → Use GPT-4o-mini with validation


MODEL_ROUTING = {
    "simple": "gpt-4o-mini",
    "medium": "gpt-4o-mini",
    "complex": "gpt-4o-2024-08-06",
}


async def smart_extract(
    doc: dict, semaphore: asyncio.Semaphore
) -> ExtractionResult:
    """Route to appropriate model based on document complexity."""
    complexity = classify_complexity(doc["text"])
    model = MODEL_ROUTING[complexity]

    result = await extract_single(doc, semaphore, model_override=model)

    # If simple/medium model failed validation, retry with stronger model
    if result.error and model != "gpt-4o-2024-08-06":
        result = await extract_single(
            doc, semaphore, model_override="gpt-4o-2024-08-06"
        )

    return result
```

### Caching and Deduplication

```python
import hashlib
import json


class ExtractionCache:
    """Cache extraction results to avoid re-processing identical documents."""

    def __init__(self, cache_dir: str = ".extraction_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

    def _cache_key(self, text: str, model: str) -> str:
        content = f"{model}:{text}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def get(self, text: str, model: str) -> ProductInfo | None:
        key = self._cache_key(text, model)
        path = self.cache_dir / f"{key}.json"
        if path.exists():
            return ProductInfo.model_validate_json(path.read_text())
        return None

    def set(self, text: str, model: str, result: ProductInfo):
        key = self._cache_key(text, model)
        path = self.cache_dir / f"{key}.json"
        path.write_text(result.model_dump_json())
```

### Cost Comparison

Here's a realistic cost comparison for processing 10,000 product listings (~200 words each):

| Strategy | Model | Cost | Accuracy | Throughput |
|----------|-------|------|----------|------------|
| All GPT-4o | gpt-4o-2024-08-06 | ~$75 | 95% | 8 docs/min (5 concurrent) |
| All GPT-4o-mini | gpt-4o-mini | ~$5 | 82% | 40 docs/min (10 concurrent) |
| Tiered routing | Mix | ~$18 | 93% | 20 docs/min |
| Tiered + retry | Mix | ~$22 | 96% | 18 docs/min |
| Tiered + cache | Mix | ~$12* | 96% | 30+ docs/min |

*\*Assuming 30% cache hit rate on repeated/similar documents*

## Putting It All Together: The Complete Pipeline

Here's the final architecture of our production extraction pipeline:

```
┌─────────────┐     ┌──────────────┐     ┌───────────────┐
│  Data Source │────►│   Preprocess │────►│  Classify     │
│  (DB/API/S3)│     │  (Clean/Chunk)│    │  Complexity   │
└─────────────┘     └──────────────┘     └───────┬───────┘
                                                 │
                                    ┌────────────┼────────────┐
                                    ▼            ▼            ▼
                              ┌──────────┐ ┌──────────┐ ┌──────────┐
                              │  Simple   │ │  Medium  │ │ Complex  │
                              │ 4o-mini  │ │ 4o-mini  │ │  4o      │
                              └────┬─────┘ └────┬─────┘ └────┬─────┘
                                   │            │            │
                                   └────────────┼────────────┘
                                                ▼
                                      ┌──────────────────┐
                                      │ Pydantic Validate │
                                      │ + Retry on Error  │
                                      └────────┬─────────┘
                                               │
                                    ┌──────────┼──────────┐
                                    ▼                     ▼
                              ┌───────────┐         ┌───────────┐
                              │  Success  │         │   Failed  │
                              │  Store    │         │  Quarantine│
                              └───────────┘         └───────────┘
                                    │
                                    ▼
                              ┌───────────┐
                              │ Quality   │
                              │ Gate      │
                              └─────┬─────┘
                                    │
                         ┌──────────┼──────────┐
                         ▼                     ▼
                   ┌───────────┐         ┌───────────┐
                   │  Auto-    │         │ Flag for  │
                   │  Approve  │         │  Review   │
                   └───────────┘         └───────────┘
```

## Common Pitfalls and Solutions

### 1. Schema Drift Between Documents

**Problem:** Different document formats have different field availability. A rigid schema either forces hallucinations (to fill required fields) or loses data (in Optional fields).

**Solution:** Use multiple specialized schemas and route documents to the right one:

```python
class InvoiceSchema(BaseModel):
    invoice_number: str
    vendor: str
    line_items: list[LineItem]
    total: float
    due_date: str

class ProductListingSchema(BaseModel):
    brand: str
    model_name: str
    pricing: PriceInfo
    condition: ProductCondition

SCHEMA_REGISTRY = {
    "invoice": InvoiceSchema,
    "product": ProductListingSchema,
    # Add more schema types...
}
```

### 2. Hallucinated Values in Optional Fields

**Problem:** LLMs tend to fill in Optional fields with plausible but incorrect values rather than leaving them null.

**Solution:** Be explicit in your system prompt and use `None` as the default with strong prompting:

```python
"NEVER guess or infer values for fields not explicitly stated. "
"If a piece of information is not clearly present in the text, "
"return null for that field. It is better to return null than to guess."
```

Also consider making more fields `Optional` — you can always merge information across multiple extraction passes.

### 3. Long Documents Exceeding Context

**Problem:** Some documents (e.g., full contracts) exceed the model's context window or degrade in accuracy when too long.

**Solution:** Chunk documents intelligently and merge results:

```python
def chunk_and_extract(text: str, max_chunk_tokens: int = 3000) -> ProductInfo:
    """Split long documents into overlapping chunks and merge results."""
    chunks = chunk_by_tokens(text, max_chunk_tokens, overlap=200)
    partial_results = [extract_product_info(chunk) for chunk in chunks]
    return merge_product_infos(partial_results)


def merge_product_infos(results: list[ProductInfo]) -> ProductInfo:
    """Merge multiple extraction results, preferring non-null values."""
    merged = {}
    for result in results:
        for field_name, value in result.model_dump().items():
            if value is not None and value != "" and value != []:
                if field_name not in merged:
                    merged[field_name] = value
                elif isinstance(value, list):
                    # Merge lists, deduplicate
                    existing = merged[field_name]
                    merged[field_name] = list(
                        set(existing + value)
                    ) if all(isinstance(x, str) for x in value) else value
    return ProductInfo(**merged)
```

## Production Checklist

Before deploying your extraction pipeline to production, verify:

- [ ] **Schema versioning:** Use a `schema_version` field so you can evolve your schema without breaking downstream consumers
- [ ] **Idempotency:** Extracting the same document twice should produce the same result (within model randomness)
- [ ] **Checkpointing:** Long batch runs should be resumable after failures
- [ ] **Monitoring:** Track extraction success rate, latency, retry rates, and completeness scores
- [ ] **Alerting:** Alert on drop in success rate or spike in latency
- [ ] **Human-in-the-loop:** Flag low-confidence extractions for manual review
- [ ] **Cost tracking:** Monitor token usage per document type and model
- [ ] **PII handling:** Sanitize documents before sending to API models if they contain sensitive data
- [ ] **Rate limiting:** Respect API rate limits with semaphore-based concurrency control
- [ ] **Testing:** Maintain a golden dataset of 50+ labeled examples for regression testing

## Conclusion

Building a reliable LLM-powered structured extraction pipeline requires more than just calling an API with a schema. The key ingredients are:

1. **Thoughtful Pydantic schemas** with descriptive fields, enums, and validators
2. **Robust retry logic** with Instructor's validation-reprompt loop
3. **Constrained decoding** (via Outlines) when you need JSON guarantees
4. **Prompt optimization** with DSPy to squeeze out the last 10–15% accuracy
5. **Quality gates** that flag problematic extractions for human review
6. **Cost optimization** through model routing, caching, and tiered processing

The frameworks covered — Instructor, Outlines, and DSPy — are all production-proven in 2026 and complement each other well. Start with Instructor for rapid prototyping, add Outlines when you need structural guarantees at scale, and layer in DSPy optimization when you've exhausted prompt engineering options.

The complete code from this tutorial is available with additional examples for invoice parsing, medical record extraction, and legal contract analysis. Happy extracting!
