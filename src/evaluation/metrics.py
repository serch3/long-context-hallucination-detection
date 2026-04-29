import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_f1_score_support, roc_auc_score
from transformers import EvalPrediction

def compute_metrics(eval_pred: EvalPrediction) -> dict:
    """
    Compute metrics for evaluation in HF Trainer.
    Expects predictions (logits) and label_ids.
    Calculates accuracy, precision, recall, f1, and AUROC.
    """
    logits, labels = eval_pred
    
    # In case of multiple outputs/tuples (dependent on model)
    if isinstance(logits, tuple):
        logits = logits[0]
        
    predictions = np.argmax(logits, axis=-1)
    
    # Softmax probabilities for positive class (AUROC)
    exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
    
    # Assuming positive class is index 1
    # If binary classification
    if logits.shape[-1] == 2:
        positive_probs = probs[:, 1]
        try:
            auroc = roc_auc_score(labels, positive_probs)
        except ValueError:
            # Handle cases where only one class is present in the batch
            auroc = float('nan')
    else:
        # Multiclass AUROC
        try:
            auroc = roc_auc_score(labels, probs, multi_class="ovr")
        except ValueError:
            auroc = float('nan')

    precision, recall, f1, _ = precision_recall_f1_score_support(
        labels, predictions, average='binary' if logits.shape[-1] == 2 else 'macro', zero_division=0
    )
    acc = accuracy_score(labels, predictions)

    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'auroc': auroc,
    }
