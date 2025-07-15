# 🧠 Perspective-Aware Healthcare Summarization

This project aims to generate high-quality summaries of healthcare-related Q&A data using multiple **perspectives** such as **CAUSE**, **INFORMATION**, **SUGGESTION**, **QUESTION**, and **EXPERIENCE**. The system combines both **supervised** and **weakly-supervised** methods using state-of-the-art NLP models to improve contextual relevance and diversity in summaries.
[🔗 Click here to view the paper](https://drive.google.com/file/d/1ebBvkEjKrzX4R1QnLnurbuVhHZb_ox2c/view?usp=sharing)


---

## 🔍 Key Objectives

- Analyze healthcare Q&A content from online forums like Reddit.
- Generate perspective-aware summaries rather than generic ones.
- Combine model outputs using ensemble techniques for improved performance.
- Explore weak supervision using Snorkel for limited or unlabeled data scenarios.

---

## 📁 Project Structure

- stacking-ensemble_nlp.ipynb # Fine-tuning Flan-T5, BART, Pegasus + ensemble
- snorkel_nlp.ipynb # Snorkel weak supervision + SVM/LR classification
- train.json # Training dataset
- valid.json # Validation dataset
- test.json # Testing dataset
- classified-test-output.json # Perspective-labeled sentences using Snorkel
- final_20_report.pdf # Detailed project report
- 20_PPT.pptx # Project presentation slides
- README.md # This file


---

## 🏗️ Methodology

### 🔹 Supervised Summarization (Stacked Ensemble)

- **Models**: Flan-T5 (w/ LoRA), BART, and Pegasus.
- **Approach**: Each model is fine-tuned using structured prompts with labeled perspectives.
- **Ensemble**: For each input, generate summaries from all models and select the most relevant and fluent one using BERTScore and instruction alignment.

### 🔹 Weak Supervision Pipeline

- **Embedding Models**: Sentence Transformers (all-mpnet-base-v2, MiniLM)
- **Classifiers**: Logistic Regression or SVM for perspective classification.
- **Snorkel Labeling Functions**: Keyword-based rules to assign perspective labels.
- **Fallback**: Zero-Shot Classification (facebook/bart-large-mnli) when models abstain.
- **Two-Stage Summarization**:
  1. Extractive (BART)
  2. Abstractive Refinement (Pegasus)

---

## 📊 Evaluation Metrics

- **BERTScore** – Measures semantic similarity
- **BLEU / METEOR** – Measure lexical overlap
- **Perspective-wise Performance** – Evaluates summary quality across CAUSE, INFO, etc.

| Model        | BERTScore F1 | BLEU  | METEOR |
|--------------|--------------|-------|--------|
| BART         | 0.8907       | 0.0883| 0.2544 |
| Flan-T5      | 0.8632       | 0.0363| 0.1856 |
| Stacked      | 0.8815       | 0.0747| 0.2386 |

---

## 🔗 Resources & Models

- **PLASMA Paper**: [arXiv](https://arxiv.org/pdf/2406.08881)
- **Flan-T5 (LoRA)**: [Google Drive](https://drive.google.com/file/d/1B7Y0v7PilShiwwZpYC9gqfX-c6dW5LeK/view?usp=drive_link)
- **Fine-tuned BART**: [Google Drive](https://drive.google.com/file/d/1gcOZbf_eemWJDFbYhMnZTkGXUgcB2cNu/view?usp=drive_link)
- **PLASMA Folder**: [Google Drive](https://drive.google.com/drive/folders/1fSkgWWQRqOLh9H4O-YfxwzM3rs7baTNH)

---

## 📄 License

This project is for educational and research use only. Attribution to original authors and datasets is required.
