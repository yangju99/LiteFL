# LiteFL: Lightweight LLM-Based Fault Localization

This project provides a lightweight fault localization framework using large language models (LLMs). It performs method-level fault localization on Defects4J using source code and comment embeddings.

---

## 🚀 Getting Started

Follow the steps below to preprocess the data and run within-project and cross-project fault localization experiments.

---

### ✅ Step 1: Data Preprocessing

**Description:**  
For each buggy version in the `defects4j/` folder, the source code and comments of covered methods are embedded using an LLM. The resulting embeddings are stored in the `chunks/` directory.

**Command:**
```bash
python generate_embeddings_batch.py




