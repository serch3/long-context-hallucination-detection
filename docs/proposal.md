# Project Proposal: Long-Context Hallucination Detection in LLM Outputs

Mandujano, Sueing 1

## Introduction and Motivation

Large language models (LLMs) such as ChatGPT, Claude, Gemini (private), LLaMA, Qwen, and DeepSeek (open weights) have demonstrated remarkable capabilities in generating human-like text with noticeable improvement in recent years. However, even today, they are prone to hallucinations: producing text that is fluent but factually incorrect, especially when processing long-context documents or generating citations with fabricated sources. Detecting such hallucinations is critical for applications in medicine, law, education, and scientific research, where errors can have serious consequences.

Recent work has focused on automated detection of LLM hallucinations using supervised models and structured evaluation frameworks trained on labeled datasets of AI-generated text. For example, Li et al. (2023) introduced HaluEval, a large-scale benchmark dataset for evaluating hallucination in LLMs across multiple domains, which serves as a primary resource for training and evaluating classifiers like those proposed in this project. More recently, Shelmanov et al. (2025) demonstrated that supervised transformer-based modules substantially outperform classical unsupervised methods in claim-level hallucination detection, with strong generalization across domains and languages, providing direct motivation for the fine-tuning approach taken here.

This project aims to develop a hallucination detection system utilizing ModernBERT as the primary model and DistilBERT as a baseline. The system will classify whether a piece of LLM-generated text is factually correct or hallucinated. By comparing the two models, we will study the tradeoff between accuracy, efficiency, and context length, providing insights into practical LLM evaluation techniques. This project also explicitly applies course concepts such as preprocessing, supervised learning, and foundational neural network techniques to improve classification performance and reduce overfitting.

## Proposed Work

### Dataset

- Primary: [HaluEval1.0](https://huggingface.co/datasets/pminervini/HaluEval) or [LibreEval](https://github.com/Arize-ai/LibreEval)
- Contains LLM-generated answers labeled as factual or hallucinated.
- Inputs include prompts/questions, context, and generated responses.

### Models

1. Baseline: DistilBERT (lightweight, fast)
2. Main: ModernBERT (long-context, modern)

### Pipeline

1. Preprocess text:
   - Combine context + answer.
   - Tokenization with Hugging Face tokenizers.
   - Input truncation: 512 tokens for DistilBERT, up to 2048 tokens for ModernBERT.
2. Train classifiers using Hugging Face Trainer with fine-tuning techniques (such as learning rate scheduling, early stopping, mixed-precision, and dropout), or a manual PyTorch training loop if time allows.
3. Evaluate with metrics:
   - Accuracy, precision, recall, F1-score.
   - Analysis by input length and error type.

## Team

- Sergio M Mandujano | U00823379
- Garrett Q Sueing | U00918836

## References

- Li, J., Cheng, X., Zhao, W. X., Nie, J.-Y., & Wen, J.-R. (2023). HaluEval: A large-scale hallucination evaluation benchmark for large language models. In H. Bouamor, J. Pino, & K. Bali (Eds.), *Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing* (pp. 6449-6464). Association for Computational Linguistics. https://doi.org/10.18653/v1/2023.emnlp-main.397
- Shelmanov, A., Fadeeva, E., Tsvigun, A., Tsvigun, I., Xie, Z., Kiselev, I., Daheim, N., Zhang, C., Vazhentsev, A., Sachan, M., Nakov, P., & Baldwin, T. (2025). A head to predict and a head to question: Pre-trained uncertainty quantification heads for hallucination detection in LLM outputs. In *Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing* (pp. 35712-35731). Association for Computational Linguistics. https://aclanthology.org/2025.emnlp-main.1809/

