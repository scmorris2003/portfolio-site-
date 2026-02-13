# Amazon Reviews Sentiment Analysis with DistilBERT

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![Transformers](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Transformers-orange)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?logo=pytorch&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-green)

Fine-tuned **DistilBERT** for binary sentiment classification (positive/negative) on Amazon product reviews. Demonstrates end-to-end NLP workflow: data loading, preprocessing, fine-tuning, evaluation, and deployment.

Live demo: [Hugging Face Space → Try it here!](https://huggingface.co/spaces/MathProblem/amazon-sentiment-demo)

## 🎯 Project Goals
- Build a practical, production-ready sentiment analyzer for customer reviews.
- Highlight math/ML engineering skills: transformer fine-tuning, custom metrics, probabilistic uncertainty via Pyro (Bayesian layer on top of embeddings).
- Create a portfolio piece showing clean code, deployment, and communication.

## 📊 Results & Performance
Achieved **~93% accuracy** on held-out test set (after 3 epochs).  
Training time: ~2 min on 3070 ti GPU with mixed precision.

Strengths: Handles clear positive/negative reviews well.  
Weaknesses: Occasional struggles with neutral/sarcastic text (common in sentiment tasks).

## 🛠️ Key Features & Math Twist
- **Base model**: DistilBERT-base-uncased (lightweight, ~40% smaller/faster than BERT).
- **Dataset**: Subset of Amazon Reviews (multi-domain English reviews with star ratings).
- **Preprocessing**: Tokenization, binary labeling (≥4 stars → positive).
- **Training**: Hugging Face Trainer API + mixed precision (fp16) for efficiency.
- **Deployment**: Gradio/Streamlit demo on HF Spaces.

