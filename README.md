# DELoRA
We are architecting, implementing, and validating a novel parameter-efficient fine-tuning  strategy called DeLoRA.Our mission is to fundamentally solve the efficiency vs. performance dilemma in multi-task LLM adaptation.We aim to create a new architectural paradigm that allows a single LLM to master diverse tasks without the catastrophic trade-offs


# **Adapter Cascade: A Modular Multi-Task Fine-Tuning Framework (LoRA-Based)**

This repository contains the implementation and demonstration of **Adapter Cascade**, a modular, parameter-efficient architecture for multi-task fine-tuning of language models.
Adapter Cascade composes **a shared LoRA adapter** with **ultra-light task-specific residual adapters**, enabling specialization without catastrophic interference, while keeping storage and training cost extremely low.

The project includes:

* Training of a **shared adapter** on MNLI + SAMSum
* Training of **task residual adapters**
* A demonstrable **Cascade assembly**: Shared + Residual
* Evaluation showing interference in mismatched adapters
* A minimal prototype of the proposed architecture, implementable on a single GPU/Colab

This work serves as a proof-of-concept for scalable, modular, adapter-based multi-task learning.

---

# **1. Overview**

Standard LoRA training stores one adapter **per task**, causing:

* Redundant parameters
* Storage overhead
* Noisy multi-task behavior
* Poor cross-task generalization
* Task interference when merging adapters

**Adapter Cascade** solves this by splitting fine-tuning updates into:

### **1. Shared Adapter (Medium-Rank)**

A LoRA module trained on all tasks together.
Provides general, reusable, task-agnostic adaptation.

### **2. Residual Adapter (Ultra-Low-Rank per Task)**

A tiny adapter (rank 1–2) trained **on top of the shared adapter**, capturing only the task-specific "delta".

### **3. Cascade Composition**

During inference:

```
W_eff = W_base + LoRA_shared + LoRA_residual_task
```

This yields:

* Specialization without forgetting
* Modular task addition
* Minimal memory footprint
* Better-than-shared performance
* Lower interference than naive merging

---

# **2. Architecture**

```
                 ┌───────────────────────────┐
                 │     Pretrained LLM        │  (Frozen)
                 └───────────────┬───────────┘
                                 │
                    ┌────────────▼─────────────┐
                    │    Shared Adapter         │  (Rank: 8–16)
                    │   LoRA_shared(A_s,B_s)    │
                    └────────────┬──────────────┘
                                 │
                ┌────────────────▼────────────────┐
                │  Task Residual Adapter (Tiny)   │  (Rank: 1–2)
                │ LoRA_task(A_t,B_t)              │
                └────────────────┬────────────────┘
                                 │
                         ┌───────▼───────┐
                         │   Model Out    │
                         └───────────────┘
```

Adapters are stacked using Hugging Face PEFT and activated via `set_adapter()`.
The residual adapter is trained with the shared adapter frozen.

---

# **3. Features**

* **Parameter Efficient:**
  Shared adapter: medium-rank (e.g., r=12)
  Task adapter: extremely low rank (r=1–2)

* **Modular & Scalable:**
  Add a new task = train one tiny residual LoRA.

* **Avoids Interference:**
  Shared-only and residual-only produce incomplete results.
  Cascade = full performance.

* **Lightweight Training:**
  Runs on Google Colab / single GPU.

* **No Merging Problems:**
  No destructive merging like classical LoRA merge.

---

# **4. Repository Structure**

```
├── data/
│   ├── sampled_mnli.json
│   ├── sampled_samsum.json
│
├── shared_adapter/           # Trained shared LoRA
├── samsum_residual_adapter/  # Residual LoRA for SAMSum
│
├── notebooks/
│   ├── train_shared.ipynb
│   ├── train_residual.ipynb
│   ├── inference_cascade.ipynb
│
├── README.md
└── requirements.txt
```

---

# **5. Training**

### **5.1 Shared Adapter**

A LoRA adapter trained on a mixture of MNLI + SAMSum samples.

```
shared_model = get_peft_model(base_model, shared_config)
trainer.train()
shared_model.save_pretrained("./shared_adapter")
```

### **5.2 Residual Task Adapter**

Freeze shared adapter, add rank-2 task LoRA, train only residual weights.

```
model = PeftModel.from_pretrained(base_model, "./shared_adapter")
freeze(shared_loras)
model.add_adapter("samsum_residual", residual_config)
trainer.train()
model.save_adapter("./samsum_residual_adapter")
```

---

# **6. Inference**

### **Cascade Composition**

```
model = PeftModel.from_pretrained(base_model, "./shared_adapter")
model.load_adapter("./samsum_residual_adapter", "samsum_residual")
model.set_adapter("samsum_residual")
output = model.generate(...)
```

### **Other modes supported**

* Shared-only
* Residual-only
* Wrong-task adapter (MNLI on SAMSum)
  Useful for demonstrating interference patterns.

---

# **7. Results (Qualitative Summary)**

| Mode                        | Behavior                    |
| --------------------------- | --------------------------- |
| Shared-only                 | Under-generates; incomplete |
| Residual-only               | No structure; incoherent    |
| Wrong adapter (MNLI→SAMSum) | Numeric pollution, miscues  |
| **Adapter Cascade**         | Best: coherent summary      |

This verifies the architecture’s effectiveness.

---

# **8. Requirements**

```
torch>=2.0
transformers>=4.35
peft>=0.7
datasets
sentence-transformers
huggingface_hub
```

Install via:

```
pip install -r requirements.txt
```

---

# **9. Motivation**

The project seeks to demonstrate:

* A practical alternative to large-scale multi-task finetuning
* Reduction of adapter storage overhead
* Multi-task specialization without destructive merging
* Modular extensibility for scaling to dozens or hundreds of tasks

Adapter Cascade is compatible with:

* LoRA
* QLoRA
* Normalized LoRA variants (DeLoRA)
* Vision and multimodal architectures (future extension)

---




# **10. Acknowledgements**

* Hugging Face PEFT for LoRA infrastructure
* Datasets from MNLI and SAMSum
* Inspiration from modular neural architectures and multi-task PEFT research

---
