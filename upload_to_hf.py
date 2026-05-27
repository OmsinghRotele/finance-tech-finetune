"""
Upload fine-tuned model to Hugging Face Hub.
Run this after training is complete.
"""

from huggingface_hub import HfApi, login
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

# ─────────────────────────────────────────────
# STEP 1: LOGIN TO HUGGING FACE
# ─────────────────────────────────────────────
# Get your token from: https://huggingface.co/settings/tokens
HF_TOKEN     = "your_hf_token_here"   # replace with your token
HF_USERNAME  = "your_username"         # replace with your HF username
REPO_NAME    = "finance-tech-mistral-7b"
ADAPTER_PATH = "./output/finance-tech-mistral"

login(token=HF_TOKEN)

# ─────────────────────────────────────────────
# STEP 2: PUSH LORA ADAPTER TO HUB
# (Uploads only the small adapter weights, not the full 7B model)
# ─────────────────────────────────────────────
from peft import PeftModel
import os

# Load and push
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")
tokenizer.push_to_hub(f"{HF_USERNAME}/{REPO_NAME}", token=HF_TOKEN)

api = HfApi()
api.create_repo(repo_id=f"{HF_USERNAME}/{REPO_NAME}", exist_ok=True, token=HF_TOKEN)

# Upload the adapter files
api.upload_folder(
    folder_path=ADAPTER_PATH,
    repo_id=f"{HF_USERNAME}/{REPO_NAME}",
    token=HF_TOKEN,
)

print(f"Model uploaded to: https://huggingface.co/{HF_USERNAME}/{REPO_NAME}")
