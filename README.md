# Fine-Tuning a Domain-Specific AI Model for Finance & Tech

This project fine-tunes `mistralai/Mistral-7B-Instruct-v0.3` using LoRA (Low-Rank Adaptation) to understand domain-specific vocabulary and generate professional responses in the **Finance** and **Tech** industries.

---

## Project Structure

```
finetune-project/
├── data/
│   └── finance_tech_dataset.jsonl   # 20 domain-specific Q&A pairs
├── notebooks/
│   └── finetune_colab.ipynb         # Full pipeline for Google Colab
├── train.py                         # Training script (LoRA fine-tuning)
├── evaluate.py                      # Evaluation / inference script
├── upload_to_hf.py                  # Upload model to Hugging Face Hub
├── requirements.txt                 # Python dependencies
└── README.md
```

---

## Domain Coverage

### Finance Terms
| Term | Description |
|------|-------------|
| EBITDA | Earnings Before Interest, Taxes, Depreciation & Amortization |
| P/E Ratio | Price-to-Earnings valuation multiple |
| Derivatives | Options, futures, forwards, swaps |
| Equity Dilution | Ownership reduction on new share issuance |
| SIP Investment | Systematic Investment Plan in mutual funds |
| Balance Sheet Analysis | Evaluation of assets, liabilities, and equity |

### Tech Terms
| Term | Description |
|------|-------------|
| Kubernetes | Container orchestration platform |
| Microservices | Distributed architecture pattern |
| CI/CD | Continuous Integration and Delivery pipelines |
| Docker Containers | Lightweight, portable application packaging |
| REST APIs | HTTP-based web service architecture |
| Vector Databases | Embedding-based similarity search databases |

---

## Approach: LoRA Fine-Tuning

Instead of retraining all 7 billion parameters (which requires 80GB+ GPU memory), we use **LoRA (Low-Rank Adaptation)**:

- Freezes the original model weights
- Adds small trainable matrices (~0.1% of total parameters) to attention layers
- Achieves domain adaptation with **~15GB GPU memory** (fits in Google Colab T4)
- Training time: ~20 minutes on a free T4 GPU

```
Original Weights (frozen)   +   LoRA Adapter (trainable)
     7B parameters                  ~8M parameters
```

---

## How to Run

### Option A: Google Colab (Recommended — Free GPU)

1. Open [Google Colab](https://colab.research.google.com)
2. Upload `notebooks/finetune_colab.ipynb`
3. Set runtime: `Runtime > Change runtime type > T4 GPU`
4. Run all cells from top to bottom
5. Takes ~20-30 minutes total

### Option B: Local Setup

**Requirements:** NVIDIA GPU with 15GB+ VRAM (RTX 3090 / 4090 / A100)

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/finance-tech-finetune
cd finance-tech-finetune

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run training
python train.py

# 4. Evaluate the model
python evaluate.py

# 5. Upload to Hugging Face
# Edit upload_to_hf.py with your HF token, then:
python upload_to_hf.py
```

---

## Model Performance — Example Outputs

**Q: What is EBITDA and how is it used in valuation?**
> EBITDA (Earnings Before Interest, Taxes, Depreciation, and Amortization) measures core operational profitability. It is widely used in the EV/EBITDA valuation multiple, allowing analysts to compare companies across industries without the distortion of capital structure or tax differences...

**Q: How does Kubernetes handle application scaling?**
> Kubernetes uses the Horizontal Pod Autoscaler (HPA) to automatically scale the number of Pod replicas based on CPU usage, memory, or custom metrics. When load increases beyond a threshold, HPA adds new Pods to distribute traffic...

---

## Deployment

| Platform | What lives there |
|----------|-----------------|
| **GitHub** | This repo — code, scripts, dataset, notebook |
| **Hugging Face** | The trained LoRA adapter weights |

Model available at: `https://huggingface.co/OmsinghRotele/finance-tech-mistral-7b`

---

## Technical Stack

- **Base Model:** `mistralai/Mistral-7B-Instruct-v0.3`
- **Fine-tuning Method:** LoRA via HuggingFace PEFT
- **Quantization:** 4-bit NF4 (bitsandbytes)
- **Trainer:** TRL SFTTrainer
- **Dataset:** 20 custom Finance & Tech instruction-response pairs
- **Training Platform:** Google Colab (T4 GPU)
