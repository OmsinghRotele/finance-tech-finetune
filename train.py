"""
Fine-Tuning Script: Domain-Specific LLM for Finance & Tech
Model: mistralai/Mistral-7B-Instruct-v0.3
Method: LoRA (Low-Rank Adaptation) via PEFT + 4-bit Quantization
"""

import os
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer
from datasets import load_dataset

# ─────────────────────────────────────────────
# 1. CONFIGURATION
# ─────────────────────────────────────────────
MODEL_NAME       = "mistralai/Mistral-7B-Instruct-v0.3"
DATASET_PATH     = "data/finance_tech_dataset.jsonl"
OUTPUT_DIR       = "./output/finance-tech-mistral"
HF_REPO_NAME     = "finance-tech-mistral-7b"   # change to your HF username/repo

MAX_SEQ_LENGTH   = 512
NUM_EPOCHS       = 3
BATCH_SIZE       = 4
GRAD_ACCUM       = 4          # effective batch = 4 * 4 = 16
LEARNING_RATE    = 2e-4
LORA_RANK        = 16
LORA_ALPHA       = 32
LORA_DROPOUT     = 0.05

# ─────────────────────────────────────────────
# 2. LOAD TOKENIZER
# ─────────────────────────────────────────────
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# ─────────────────────────────────────────────
# 3. LOAD MODEL WITH 4-BIT QUANTIZATION
# ─────────────────────────────────────────────
print("Loading model in 4-bit quantization...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)
model.config.use_cache = False
model.config.pretraining_tp = 1

# ─────────────────────────────────────────────
# 4. APPLY LORA (PEFT)
# ─────────────────────────────────────────────
print("Applying LoRA configuration...")
lora_config = LoraConfig(
    r=LORA_RANK,
    lora_alpha=LORA_ALPHA,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# Expected output: ~1-2% of total 7B parameters are trainable

# ─────────────────────────────────────────────
# 5. LOAD AND FORMAT DATASET
# ─────────────────────────────────────────────
print("Loading dataset...")
dataset = load_dataset("json", data_files=DATASET_PATH, split="train")

def format_prompt(example):
    """Format each sample into Mistral instruction template."""
    return {
        "text": f"<s>[INST] {example['instruction']} [/INST] {example['response']} </s>"
    }

dataset = dataset.map(format_prompt)
print(f"Dataset size: {len(dataset)} examples")

# ─────────────────────────────────────────────
# 6. TRAINING ARGUMENTS
# ─────────────────────────────────────────────
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LEARNING_RATE,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    fp16=True,
    logging_steps=5,
    save_steps=50,
    save_total_limit=2,
    evaluation_strategy="no",
    report_to="none",
    optim="paged_adamw_32bit",  # memory-efficient optimizer
)

# ─────────────────────────────────────────────
# 7. TRAIN
# ─────────────────────────────────────────────
print("Starting training...")
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    tokenizer=tokenizer,
    args=training_args,
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LENGTH,
    packing=False,
)

trainer.train()

# ─────────────────────────────────────────────
# 8. SAVE MODEL
# ─────────────────────────────────────────────
print("Saving fine-tuned model...")
os.makedirs(OUTPUT_DIR, exist_ok=True)
trainer.model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Model saved to {OUTPUT_DIR}")
