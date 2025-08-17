
This project aims to generate high-quality summaries of healthcare-related Q&A data using multiple **perspectives** such as **CAUSE**, **INFORMATION**, **SUGGESTION**, **QUESTION**, and **EXPERIENCE**. The system combines both **supervised** and **weakly-supervised** methods using state-of-the-art NLP models to improve contextual relevance and diversity in summaries.
[🔗 Click here to view the paper](https://drive.google.com/file/d/1ebBvkEjKrzX4R1QnLnurbuVhHZb_ox2c/view?usp=sharing)
# Efficient Summarization of Healthcare Responses

[![Python](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Transformers](https://img.shields.io/badge/🤗-Transformers-orange)](https://huggingface.co/transformers/)

A comprehensive approach to perspective-aware healthcare summarization that generates structured summaries from community question-answering forums by capturing multiple viewpoints including causes, suggestions, experiences, and information.

## 🎯 Overview

Online healthcare forums like r/AskDocs, r/DiagnoseMe, and r/Medical_Advice generate vast amounts of responses ranging from expert insights to personal experiences. This project addresses the challenge of extracting reliable and meaningful information by developing models that generate perspective-aware summaries.

### Key Features

- **Multi-Perspective Summarization**: Captures different viewpoints (Cause, Information, Suggestion, Experience, Question)
- **Advanced Model Integration**: Utilizes FLAN-T5, BART, Pegasus, and PLASMA models
- **Weakly Supervised Learning**: Implements Snorkel-based labeling for unlabeled data
- **Ensemble Methods**: Combines multiple models for improved performance
- **Comprehensive Evaluation**: Uses BLEU, METEOR, and BERTScore metrics

## 📊 Dataset

The project uses the **PUMA dataset** for perspective-aware healthcare summarization:

- **Training**: 2,236 samples
- **Validation**: 959 samples  
- **Testing**: 640 samples

### Data Structure
```json
{
  "uri": "unique_identifier",
  "question": "user_question",
  "context": "background_context",
  "answers": ["list_of_community_answers"],
  "labelled_answer_spans": {
    "PERSPECTIVE_TYPE": [{"txt": "relevant_text_span"}]
  },
  "labelled_summaries": {
    "PERSPECTIVE_TYPE": "human_written_summary"
  }
}
```

## 🏗️ Architecture

### 1. Supervised Fine-Tuning Approach
- **FLAN-T5 with LoRA**: Memory-efficient fine-tuning using Low-Rank Adaptation
- **BART**: Fine-tuned encoder-decoder architecture
- **Pegasus**: Specialized for abstractive summarization
- **Stacking Ensemble**: Combines outputs from multiple models

### 2. Weakly Supervised Pipeline
- **Snorkel + Logistic Regression**: Weak labeling with keyword-based functions
- **SVM Classification**: Support Vector Machine for perspective prediction
- **Zero-Shot Classification**: Fallback classification using BART-MNLI
- **Two-Stage Summarization**: Extractive (BART) → Abstractive (Pegasus)

## 🚀 Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/healthcare-summarization.git
cd healthcare-summarization

# Install dependencies
pip install -r requirements.txt

# Install additional packages
pip install transformers torch sentence-transformers snorkel-metal
```

### Requirements
```
torch>=1.9.0
transformers>=4.20.0
sentence-transformers>=2.2.0
snorkel-metal>=0.9.0
scikit-learn>=1.0.0
pandas>=1.3.0
numpy>=1.21.0
datasets>=2.0.0
evaluate>=0.4.0
```

## 💻 Usage

### Training Models

#### 1. Fine-tune FLAN-T5 with LoRA
```python
from src.models.flan_t5_lora import FlanT5LoRATrainer

trainer = FlanT5LoRATrainer(
    model_name="google/flan-t5-base",
    lora_rank=8,
    learning_rate=5e-5
)
trainer.train(train_dataset, valid_dataset, epochs=5)
```

#### 2. Fine-tune BART
```python
from src.models.bart_trainer import BARTTrainer

trainer = BARTTrainer(
    model_name="facebook/bart-large-cnn",
    batch_size=8,
    learning_rate=5e-5
)
trainer.train(train_dataset, valid_dataset, epochs=5)
```

#### 3. Weakly Supervised Pipeline
```python
from src.weak_supervision.snorkel_pipeline import SnorkelPipeline

pipeline = SnorkelPipeline()
pipeline.create_labeling_functions()
pipeline.train_label_model(train_data)
pipeline.classify_test_data(test_data)
```

### Generating Summaries

```python
from src.inference.ensemble_predictor import EnsemblePredictor

predictor = EnsemblePredictor(
    bart_model_path="path/to/bart/model",
    t5_model_path="path/to/t5/model"
)

question = "What causes headaches?"
answers = ["Answer 1...", "Answer 2..."]
perspective = "CAUSE"

summary = predictor.predict(question, answers, perspective)
print(f"Generated Summary: {summary}")
```

## 📈 Results

### Model Performance Comparison

| Model | BERTScore F1 | BLEU | METEOR |
|-------|--------------|------|--------|
| BART | 0.8907 | 0.0883 | 0.2544 |
| FLAN-T5 | 0.8632 | 0.0363 | 0.1856 |
| **Stacked Ensemble** | **0.8815** | **0.0747** | **0.2386** |

### Perspective-wise Performance (FLAN-T5 + Pegasus)

| Perspective | BERTScore F1 | Examples |
|-------------|--------------|----------|
| Cause | 0.8666 | 102 |
| Information | 0.8662 | 484 |
| Suggestion | 0.8582 | 392 |
| Experience | 0.8487 | 205 |
| Question | 0.8395 | 64 |

## 🔧 Model Files

Pre-trained models are available for download:

- **FLAN-T5 with LoRA**: [Download](https://drive.google.com/file/d/1B7Y0v7PilShiwwZpYC9gqfX-c6dW5LeK/view?usp=drive_link)
- **Fine-tuned BART**: [Download](https://drive.google.com/file/d/1gcOZbf_eemWJDFbYhMnZTkGXUgcB2cNu/view?usp=drive_link)
- **PLASMA Model**: [Download](https://drive.google.com/drive/folders/1fSkgWWQRqOLh9H4O-YfxwzM3rs7baTNH)

## 📁 Project Structure

```
healthcare-summarization/
├── src/
│   ├── models/
│   │   ├── flan_t5_lora.py
│   │   ├── bart_trainer.py
│   │   └── plasma_model.py
│   ├── weak_supervision/
│   │   ├── snorkel_pipeline.py
│   │   └── labeling_functions.py
│   ├── inference/
│   │   └── ensemble_predictor.py
│   └── evaluation/
│       └── metrics.py
├── data/
│   ├── train.json
│   ├── valid.json
│   └── test.json
├── notebooks/
│   ├── data_exploration.ipynb
│   └── model_analysis.ipynb
├── requirements.txt
└── README.md
```

## 🎯 Key Innovations

1. **Perspective-Aware Architecture**: First comprehensive approach to multi-perspective healthcare summarization
2. **Hybrid Supervision**: Combines supervised fine-tuning with weakly supervised learning
3. **Efficient Training**: Uses LoRA for parameter-efficient fine-tuning
4. **Robust Ensemble**: Stacking approach that leverages complementary model strengths

## 📊 Evaluation Metrics

- **BERTScore**: Measures semantic similarity between generated and reference summaries
- **BLEU**: Evaluates n-gram overlap with reference text
- **METEOR**: Considers synonyms and paraphrases for better alignment assessment

## 🔮 Future Work

- **Reinforcement Learning**: Implement reward-based fine-tuning for human preference alignment
- **Automated Ensemble Selection**: Develop scoring mechanisms for automatic model selection
- **Data Augmentation**: Collect more perspective-labeled data for underrepresented categories
- **Real-time Deployment**: Create API endpoints for live healthcare forum integration

## 👥 Team

- **Dhairya** (Roll No: 2022157) - dhairya22157@iiitd.ac.in
- **Harsh Vishwakarma** (Roll No: 2022205) - harsh22205@iiitd.ac.in  
- **Pandillapelly Harshvardhini** (Roll No: 2022345) - pandillapelly22345@iiitd.ac.in

## 📚 References

1. [Perspective-aware Healthcare Answer Summarization](https://arxiv.org/abs/example)
2. [FLAN-T5 Model](https://huggingface.co/google/flan-t5-base)
3. [BART for Summarization](https://huggingface.co/facebook/bart-large-cnn)
4. [Snorkel Weak Supervision](https://www.snorkel.org/)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

We welcome contributions! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## ⭐ Citation

If you use this work in your research, please cite:

```bibtex
@article{healthcare_summarization_2024,
  title={Efficient Summarization of Healthcare Responses},
  author={Dhairya and Harsh Vishwakarma and Pandillapelly Harshvardhini},
  journal={GROUP-20 Project Report},
  year={2024}
}
```
