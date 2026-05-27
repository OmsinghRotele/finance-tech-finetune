"""
Evaluation Script: Test the Fine-Tuned Finance & Tech Model
Run this after training to verify domain-specific responses.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
from peft import PeftModel

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
BASE_MODEL   = "mistralai/Mistral-7B-Instruct-v0.3"
ADAPTER_PATH = "./output/finance-tech-mistral"   # path to saved LoRA weights

# ─────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────
print("Loading base model...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

print("Loading fine-tuned LoRA adapter...")
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
model.eval()

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# ─────────────────────────────────────────────
# INFERENCE FUNCTION
# ─────────────────────────────────────────────
def ask(question: str, max_new_tokens: int = 300) -> str:
    prompt = f"<s>[INST] {question} [/INST]"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    # Strip the prompt from the output
    return response.split("[/INST]")[-1].strip()

# ─────────────────────────────────────────────
# TEST CASES
# ─────────────────────────────────────────────
finance_tests = [
    "What is EBITDA and how is it used in company valuation?",
    "Explain equity dilution with a practical example.",
    "How do derivatives help in risk management?",
    "What does a P/E ratio of 50x indicate about a stock?",
    "Summarize the key components of a balance sheet.",
]

tech_tests = [
    "What is Kubernetes and how does it manage Docker containers?",
    "Explain the difference between CI and CD in a DevOps pipeline.",
    "Why would a company use a vector database instead of PostgreSQL?",
    "How do microservices communicate with each other using REST APIs?",
    "What problem does a service mesh solve in microservices?",
]

print("\n" + "="*60)
print("FINANCE DOMAIN TESTS")
print("="*60)
for q in finance_tests:
    print(f"\nQ: {q}")
    print(f"A: {ask(q)}")
    print("-"*50)

print("\n" + "="*60)
print("TECH DOMAIN TESTS")
print("="*60)
for q in tech_tests:
    print(f"\nQ: {q}")
    print(f"A: {ask(q)}")
    print("-"*50)
