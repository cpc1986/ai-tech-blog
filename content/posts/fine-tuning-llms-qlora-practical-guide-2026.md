---
title: "Fine-Tuning Open-Source LLMs with QLoRA: A Step-by-Step Practical Guide for 2026"
date: "2026-06-01"
excerpt: "A comprehensive hands-on tutorial covering QLoRA fine-tuning for open-source large language models. Learn how to train Llama 4, Mistral, and Qwen models on consumer GPUs with 4-bit quantization, LoRA adapters, and the Hugging Face ecosystem — complete with code, benchmarks, and deployment strategies."
tags: ["QLoRA", "fine-tuning", "LLM training", "LoRA", "Llama 4", "GPU training", "Hugging Face", "model adaptation", "2026"]
category: "Tutorials"
---

Fine-tuning large language models used to require a cluster of A100 GPUs and a six-figure budget. In 2026, thanks to **QLoRA** (Quantized Low-Rank Adaptation), you can adapt a capable 8B–14B parameter model to your specific domain on a single consumer GPU in under two hours. This tutorial walks you through the entire process — from dataset preparation to deployment — with real code, real benchmarks, and practical tips learned from dozens of fine-tuning runs.

## Why QLoRA Matters in 2026

The open-source LLM ecosystem has reached an inflection point. Models like Llama 4 Scout, Mistral Small 3.1, and Qwen 3 now match or exceed GPT-4-class performance on many benchmarks. But base models are generalists. To get specialist performance — whether for medical diagnosis, legal document analysis, code generation for a specific framework, or customer support — you need fine-tuning.

Traditional full fine-tuning of an 8B parameter model requires:

| Resource | Full Fine-Tuning | QLoRA Fine-Tuning |
|----------|-----------------|-------------------|
| GPU VRAM | 60–120 GB (multiple A100s) | 12–24 GB (single RTX 4090/5070) |
| Training time (10K samples) | 4–8 hours | 1–2 hours |
| Disk space for checkpoints | 50–100 GB | 2–5 GB (adapter only) |
| Cost (cloud GPU) | $50–200 per run | $5–15 per run |
| Trainable parameters | 100% (8B) | 0.5–2% (40M–160M) |

QLoRA works by freezing the base model in 4-bit precision and training only small "adapter" layers using Low-Rank Adaptation (LoRA). This collapses the resource requirements by 10–20x while retaining 95–99% of the quality of full fine-tuning.

## How QLoRA Works Under the Hood

Before diving into code, understanding the mechanics helps you make better hyperparameter decisions.

### Standard LoRA

LoRA adds trainable low-rank decomposition matrices to each attention layer. For a weight matrix `W` of shape `[d_model, d_model]`, LoRA introduces two smaller matrices:

```
W' = W + B · A
```

Where `B` is `[d_model, r]` and `A` is `[r, d_model]`, with rank `r` typically between 8 and 64. Only `A` and `B` are updated during training.

### QLoRA Enhancements 4-bit Quantization

QLoRA extends LoRA with three innovations:

1. **4-bit NormalFloat (NF4) quantization**: A data type optimized for normally-distributed neural network weights. It information-theoretically optimal for weight quantization, preserving more precision than naive INT4.

2. **Double quantization**: Quantizes the quantization constants themselves, saving an additional ~0.37 bits per parameter.

3. **Paged optimizers**: Uses CPU RAM as overflow for optimizer states during memory spikes (e.g., when processing long sequences), preventing out-of-memory crashes.

The result is dramatic: you can load an 8B model in ~5 GB of VRAM (4-bit) and use the remaining 7–19 GB for the forward pass, gradients of the adapters, and optimizer states.

## Step 1: Environment Setup

### Hardware Requirements

| GPU | VRAM | Max Model Size (QLoRA) | Recommended |
|-----|------|----------------------|-------------|
| RTX 3060 | 12 GB | 3B–7B (r=8) | Budget option |
| RTX 4070 Ti | 16 GB | 7B–8B (r=16) | Good |
| RTX 4090 | 24 GB | 8B–14B (r=32–64) | Excellent |
| RTX 5070 Ti | 16 GB | 8B (r=16–32) | Great efficiency |
| RTX 5090 | 32 GB | 14B–32B (r=64) | Best consumer |
| A100 80GB | 80 GB | 70B+ | Cloud/data center |

### Software Installation

Create a fresh environment and install the required packages:

```bash
# Create conda environment
conda create -n qlora python=3.11 -y
conda activate qlora

# Install PyTorch with CUDA support (adjust for your CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install the Hugging Face ecosystem
pip install transformers>=4.48.0
pip install datasets>=3.2.0
pip install peft>=0.14.0
pip install bitsandbytes>=0.45.0
pip install accelerate>=1.2.0
pip install trl>=0.15.0
pip install wandb evaluate scikit-learn
```

Verify your installation:

```python
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

import transformers, peft, bitsandbytes, trl
print(f"Transformers: {transformers.__version__}")
print(f"PEFT: {peft.__version__}")
print(f"Bitsandbytes: {bitsandbytes.__version__}")
print(f"TRL: {trl.__version__}")
```

## Step 2: Preparing Your Dataset

Fine-tuning quality is dominated by data quality. A well-curated dataset of 1,000 examples will outperform a noisy dataset of 100,000 examples.

### Data Format

For instruction-tuning (the most common use case), format your data as JSONL with `instruction`, `input` (optional), and `output` fields:

```json
{"instruction": "Classify the sentiment of this review.", "input": "The product quality exceeded my expectations!", "output": "Positive"}
{"instruction": "Summarize this meeting transcript in 3 bullet points.", "input": "John: We need to ship v2.1 by Friday... Sarah: The QA team found 3 bugs...", "output": "- Release deadline for v2.1 is Friday\n- QA identified 3 blocking bugs\n- Team to prioritize bug fixes over new features"}
```

### Loading and Formatting with Hugging Face Datasets

```python
from datasets import load_dataset

# Option 1: Load from Hugging Face Hub
dataset = load_dataset("your-username/your-dataset", split="train")

# Option 2: Load from local JSONL
dataset = load_dataset("json", data_files="training_data.jsonl", split="train")

# Format into the chat template expected by the model
def format_to_chat(example):
    """Convert instruction format to multi-turn chat format."""
    messages = []
    
    user_content = example["instruction"]
    if example.get("input"):
        user_content += f"\n\n{example['input']}"
    
    messages.append({"role": "user", "content": user_content})
    messages.append({"role": "assistant", "content": example["output"]})
    
    return {"messages": messages}

dataset = dataset.map(format_to_chat)
```

### Dataset Quality Checklist

Before training, apply these filters:

```python
# Filter examples by length (avoid extremely short or long samples)
def filter_by_length(example):
    total_len = len(example["messages"][0]["content"]) + len(example["messages"][1]["content"])
    return 50 < total_len < 4096

dataset = dataset.filter(filter_by_length)

# Check dataset statistics
lengths = [len(m["content"]) for ex in dataset for m in ex["messages"]]
print(f"Dataset size: {len(dataset)}")
print(f"Avg length: {sum(lengths)/len(lengths):.0f} chars")
print(f"Min/Max length: {min(lengths)} / {max(lengths)} chars")

# Split into train/validation
split = dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = split["train"]
eval_dataset = split["test"]
print(f"Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")
```

**Rule of thumb**: Start with 2,000–5,000 high-quality examples. More data helps, but only if quality is maintained.

## Step 3: Loading the Model with 4-bit Quantization

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

model_id = "unsloth/Llama-4-Scout-8B-Instruct-bnb-4bit"
# Alternatives:
# model_id = "unsloth/Mistral-Small-3.1-24B-Instruct-bnb-4bit"
# model_id = "unsloth/Qwen3-8B-Instruct-bnb-4bit"

# 4-bit quantization configuration
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",           # NormalFloat4 — optimal for weights
    bnb_4bit_compute_dtype=torch.bfloat16,  # Computation in bf16
    bnb_4bit_use_double_quant=True,        # Double quantization for memory savings
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load model with quantization
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",           # Automatically distribute across GPUs
    attn_implementation="flash_attention_2",  # Use Flash Attention 2 for speed
)

# Print memory footprint
print(f"Model memory: {model.get_memory_footprint() / 1e9:.2f} GB")
# Typical output for 8B model: ~4.8 GB
```

## Step 4: Configuring LoRA Adapters

The rank (`r`) parameter controls the expressiveness of the adapters. Higher rank means more trainable parameters and potentially better adaptation, but at higher memory cost.

```python
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType

# Prepare model for training with quantized weights
model = prepare_model_for_kbit_training(model)

# LoRA configuration
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=32,                          # Rank: 8, 16, 32, or 64
    lora_alpha=64,                 # Scaling factor (typically 2x rank)
    lora_dropout=0.05,             # Dropout for regularization
    target_modules=[               # Which layers to adapt
        "q_proj", "k_proj", "v_proj", "o_proj",  # Attention projections
        "gate_proj", "up_proj", "down_proj",       # MLP projections
    ],
    bias="none",
)

# Apply LoRA to model
model = get_peft_model(model, lora_config)

# Print trainable parameters
model.print_trainable_parameters()
# Example output: trainable params: 83,886,080 || all params: 8,030,261,248 || trainable%: 1.04%
```

### Choosing the Right Rank

| Rank | Trainable Params (8B model) | VRAM Overhead | Best For |
|------|-----------------------------|---------------|----------|
| r=8 | ~20M | +0.5 GB | Simple tasks, style transfer, small datasets |
| r=16 | ~40M | +1.0 GB | Moderate complexity, domain adaptation |
| **r=32** | **~80M** | **+1.5 GB** | **General purpose, good balance** |
| r=64 | ~160M | +2.5 GB | Complex tasks, larger datasets |
| r=128 | ~320M | +4.0 GB | Maximum capacity, large datasets (10K+) |

**Recommendation**: Start with r=32 and `lora_alpha=64`. Increase to r=64 only if evaluation loss is still decreasing at the end of training.

## Step 5: Training with SFTTrainer

The `SFTTrainer` from the TRL library handles the complexities of supervised fine-tuning, including chat template application and padding:

```python
from trl import SFTConfig, SFTTrainer

# Training configuration
training_args = SFTConfig(
    output_dir="./results/llama4-scout-qlora",
    
    # Training hyperparameters
    num_train_epochs=3,                    # 3 epochs is a good starting point
    per_device_train_batch_size=2,         # Small batch due to memory constraints
    gradient_accumulation_steps=8,         # Effective batch size = 2 * 8 = 16
    learning_rate=2e-4,                    # Higher LR is fine for LoRA
    weight_decay=0.01,                     # L2 regularization
    warmup_ratio=0.1,                      # 10% warmup steps
    lr_scheduler_type="cosine",            # Cosine decay schedule
    
    # Precision and memory
    bf16=True,                             # Use bfloat16 for stable training
    optim="paged_adamw_8bit",              # 8-bit Adam for memory savings
    
    # Sequence handling
    max_seq_length=2048,                   # Max sequence length
    packing=True,                          # Pack multiple examples into one sequence
    
    # Evaluation and saving
    eval_strategy="steps",
    eval_steps=50,
    save_strategy="steps",
    save_steps=50,
    save_total_limit=3,                    # Keep only 3 checkpoints
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    
    # Logging
    logging_steps=10,
    report_to="wandb",                     # Log to Weights & Biases
    run_name="llama4-scout-qlora-v1",
    
    # QLoRA-specific
    gradient_checkpointing=True,           # Trade compute for memory
    gradient_checkpointing_kwargs={"use_reentrant": False},
)

# Initialize trainer
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    processing_class=tokenizer,
)

# Start training
trainer.train()

# Save the final adapter
trainer.save_model("./results/llama4-scout-qlora/final-adapter")
```

### Understanding Key Hyperparameters

**Effective Batch Size**: Your true batch size is `per_device_train_batch_size × gradient_accumulation_steps × num_gpus`. With batch_size=2 and grad_accum=8 on one GPU, your effective batch size is 16. This is important for stable training — batch sizes of 8–32 generally work best for QLoRA.

**Learning Rate**: QLoRA typically uses higher learning rates (1e-4 to 3e-4) than full fine-tuning (1e-5 to 5e-5). This is because only a small number of parameters are being updated, so they need to move faster.

**Epochs**: 2–5 epochs is typical. Monitor eval loss — if it starts increasing while train loss decreases, you are overfitting. Stop training at the epoch before this happens.

## Step 6: Monitoring Training

### Using Weights & Biases

```python
import wandb

# Initialize W&B run (if not using report_to="wandb" in training args)
wandb.init(
    project="llm-finetuning",
    name="llama4-scout-qlora-v1",
    config={
        "model": model_id,
        "lora_rank": 32,
        "lora_alpha": 64,
        "learning_rate": 2e-4,
        "epochs": 3,
        "batch_size": 2,
        "grad_accum": 8,
        "dataset_size": len(train_dataset),
    }
)
```

### What to Watch For

| Metric | Healthy Range | Problem Signal |
|--------|--------------|----------------|
| Training loss | Steadily decreasing to 0.3–0.8 | Plateaus immediately or barely decreases |
| Eval loss | Decreasing, tracking train loss | Diverges upward while train loss drops (overfitting) |
| Learning rate | Following cosine schedule | Drops to zero too early (check warmup_ratio) |
| GPU utilization | >85% | <70% (check data loading bottleneck) |
| GPU memory | Stable after first few steps | Keeps growing (memory leak) |

### Manual Loss Tracking (No W&B)

```python
# Plot training loss from trainer state
import json
import matplotlib.pyplot as plt

with open("./results/llama4-scout-qlora/trainer_state.json") as f:
    state = json.load(f)

train_loss = [(e["step"], e["loss"]) for e in state["log_history"] if "loss" in e]
eval_loss = [(e["step"], e["eval_loss"]) for e in state["log_history"] if "eval_loss" in e]

plt.figure(figsize=(10, 5))
plt.plot(*zip(*train_loss), label="Train Loss")
if eval_loss:
    plt.plot(*zip(*eval_loss), label="Eval Loss", marker="o")
plt.xlabel("Step")
plt.ylabel("Loss")
plt.legend()
plt.title("QLoRA Training Progress")
plt.savefig("training_loss.png", dpi=150)
plt.show()
```

## Step 7: Evaluation and Testing

After training, evaluate your model on held-out data:

```python
from transformers import pipeline
import torch

# Load the fine-tuned model
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
)

model = PeftModel.from_pretrained(base_model, "./results/llama4-scout-qlora/final-adapter")

# Test with sample prompts
test_prompts = [
    "Explain the key differences between REST and GraphQL APIs.",
    "Write a Python function to calculate the Fibonacci sequence using memoization.",
    "Summarize the main advantages of microservices architecture.",
]

for prompt in test_prompts:
    messages = [{"role": "user", "content": prompt}]
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
        )
    
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    print(f"Prompt: {prompt[:60]}...")
    print(f"Response: {response[:200]}...")
    print("-" * 80)
```

### Quantitative Evaluation

For classification or structured output tasks, compute accuracy metrics:

```python
from evaluate import load
import numpy as np

def evaluate_on_dataset(model, tokenizer, eval_dataset, max_samples=100):
    """Simple accuracy evaluation for instruction-following tasks."""
    correct = 0
    total = 0
    
    for example in eval_dataset.select(range(min(max_samples, len(eval_dataset)))):
        # Get the expected output
        expected = example["messages"][1]["content"].strip()
        
        # Generate prediction
        messages = [{"role": "user", "content": example["messages"][0]["content"]}]
        input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=256, temperature=0.1)
        
        prediction = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        
        # Simple string matching (adjust for your use case)
        if expected.lower() in prediction.lower():
            correct += 1
        total += 1
    
    accuracy = correct / total * 100
    print(f"Accuracy: {accuracy:.1f}% ({correct}/{total})")
    return accuracy

evaluate_on_dataset(model, tokenizer, eval_dataset)
```

## Step 8: Merging and Exporting the Model

For deployment, merge the LoRA adapter back into the base model:

```python
from peft import AutoPeftModelForCausalLM

# Load in full precision for merging
model = AutoPeftModelForCausalLM.from_pretrained(
    "./results/llama4-scout-qlora/final-adapter",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

# Merge adapter weights into base model
merged_model = model.merge_and_unload()

# Save the merged model
output_dir = "./models/llama4-scout-finetuned-merged"
merged_model.save_pretrained(output_dir, safe_serialization=True)
tokenizer.save_pretrained(output_dir)

print(f"Merged model saved to {output_dir}")
print(f"Size: {sum(os.path.getsize(os.path.join(dp, f)) for dp, dn, fn in os.walk(output_dir) for f in fn) / 1e9:.2f} GB")
```

### Export to GGUF for Local Inference

If you want to run the model with Ollama or llama.cpp:

```bash
# Install llama.cpp
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp && make -j$(nproc)

# Convert to GGUF
python convert_hf_to_gguf.py ../models/llama4-scout-finetuned-merged \
    --outfile ../models/llama4-scout-finetuned-q4_k_m.gguf \
    --outtype q4_k_m

# Test with llama.cpp
./llama-cli -m ../models/llama4-scout-finetuned-q4_k_m.gguf \
    -p "Explain quantum computing in simple terms." \
    -n 256 --temp 0.7
```

### Upload to Hugging Face Hub

```python
from huggingface_hub import HfApi

api = HfApi()

# Upload the adapter (lightweight)
api.upload_folder(
    folder_path="./results/llama4-scout-qlora/final-adapter",
    repo_id="your-username/llama4-scout-qlora-finetuned",
    repo_type="model",
)

# Or upload the merged model
api.upload_folder(
    folder_path="./models/llama4-scout-finetuned-merged",
    repo_id="your-username/llama4-scout-finetuned-merged",
    repo_type="model",
)
```

## Step 9: Deployment Strategies

### Option A: vLLM for Production Serving

```bash
# Install vLLM
pip install vllm

# Serve the merged model
python -m vllm.serve \
    --model ./models/llama4-scout-finetuned-merged \
    --tensor-parallel-size 1 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.9 \
    --port 8000

# Or serve directly from LoRA adapter (vLLM supports dynamic LoRA loading)
python -m vllm.serve \
    --model unsloth/Llama-4-Scout-8B-Instruct-bnb-4bit \
    --enable-lora \
    --lora-modules my-adapter=./results/llama4-scout-qlora/final-adapter \
    --port 8000
```

### Option B: Ollama for Local Deployment

```bash
# Create a Modelfile
cat > Modelfile << 'EOF'
FROM llama4-scout-finetuned-q4_k_m.gguf

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER num_ctx 4096

SYSTEM You are a helpful AI assistant fine-tuned for [your domain] tasks.
EOF

# Build and run
ollama create my-finetuned-model -f Modelfile
ollama run my-finetuned-model
```

### Option C: Serverless Deployment

For cost-effective deployment without managing infrastructure:

```python
# Deploy to Hugging Face Inference Endpoints
from huggingface_hub import create_inference_endpoint

endpoint = create_inference_endpoint(
    name="llama4-scout-finetuned",
    repository="your-username/llama4-scout-finetuned-merged",
    framework="pytorch",
    accelerator="gpu",
    instance_size="x1",
    instance_type="nvidia-a10g",
    region="us-east-1",
)
print(f"Endpoint URL: {endpoint.url}")
```

## Common Pitfalls and Troubleshooting

### Out of Memory (OOM) Errors

```python
# If you get OOM during training, try these in order:

# 1. Reduce batch size
per_device_train_batch_size=1
gradient_accumulation_steps=16  # Maintain effective batch size

# 2. Reduce sequence length
max_seq_length=1024  # Instead of 2048

# 3. Reduce LoRA rank
r=16  # Instead of 32

# 4. Disable packing
packing=False

# 5. Use more aggressive gradient checkpointing
gradient_checkpointing_kwargs={"use_reentrant": False}
```

### Model Outputs Gibberish

This usually indicates a training configuration error:

- **Learning rate too high**: Reduce to 1e-4 or 5e-5
- **Wrong chat template**: Ensure `apply_chat_template` is used correctly
- **Insufficient training**: Increase epochs or check dataset quality
- **Pad token misconfiguration**: Verify `tokenizer.pad_token` is set

### Slow Training Speed

| Optimization | Speedup | How to Apply |
|-------------|---------|--------------|
| Flash Attention 2 | 2–3x | `attn_implementation="flash_attention_2"` |
| Packing | 1.5–3x | `packing=True` in SFTConfig |
| 8-bit optimizer | 1.2x memory savings | `optim="paged_adamw_8bit"` |
| Gradient checkpointing | Saves 40% VRAM | `gradient_checkpointing=True` |
| DataLoader workers | 10–20% | `dataloader_num_workers=4` |

## QLoRA vs Alternatives in 2026

| Method | VRAM (8B model) | Quality vs Full FT | Training Speed | Complexity |
|--------|------------------|-------------------|---------------|------------|
| **QLoRA** | **12–18 GB** | **95–99%** | **Fast** | **Medium** |
| Full Fine-Tuning | 60–80 GB | 100% (baseline) | Slow | High |
| LoRA (fp16) | 30–40 GB | 96–99% | Fast | Low |
| IA³ | 10–14 GB | 90–95% | Very fast | Low |
| Prompt Tuning | 6–8 GB | 80–90% | Very fast | Very low |
| DoRA | 16–22 GB | 97–100% | Medium | Medium |

QLoRA hits the sweet spot for most practitioners: near-full fine-tuning quality at a fraction of the cost.

## Best Practices Summary

1. **Data quality trumps quantity** — 2K clean examples beat 100K noisy ones
2. **Start conservative** — r=32, lr=2e-4, 3 epochs, then iterate
3. **Always hold out 10% for evaluation** — never train on your entire dataset
4. **Monitor eval loss closely** — stop before overfitting begins
5. **Use cosine scheduling with warmup** — stable convergence
6. **Pack sequences** — significant throughput improvement with packing=True
7. **Save adapters, not full models** — 100x smaller, just as usable
8. **Version everything** — use W&B or MLflow to track experiments
9. **Test on real examples** — automated metrics do not capture qualitative quality
10. **Iterate on data before hyperparameters** — most improvements come from better data

## Conclusion

QLoRA has democratized LLM fine-tuning. What once required a distributed GPU cluster now runs on a gaming PC. In 2026, with models like Llama 4 Scout and Mistral Small 3.1 providing strong open-source foundations, there has never been a better time to build custom AI solutions tailored to your specific domain.

The workflow is straightforward: curate high-quality data, configure 4-bit quantization with LoRA adapters, train with SFTTrainer, merge and deploy. The entire cycle — from data preparation to production deployment — can be completed in a single day.

Start small, iterate on your data, and scale your rank and dataset size as you validate results. Happy fine-tuning.
