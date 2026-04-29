import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
from datasets import load_from_disk
from tqdm import tqdm

def analyze_errors_by_length(model_path: str, dataset_path: str, split: str = "test", output_dir: str = "results/error_analysis"):
    """
    Runs model predictions on the dataset split and identifies false positives / false negatives.
    Buckets mistakes by tokenized input length to analyze if performance degrades on long contexts.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    classifier = pipeline("text-classification", model=model, tokenizer=tokenizer, truncation=True)
    
    dataset = load_from_disk(dataset_path)[split]
    
    results = []
    
    # Process dataset
    print("Running inference for error analysis...")
    for idx, item in enumerate(tqdm(dataset)):
        # Assuming your processed dataset has 'input_text' and 'label'
        # Fallback to 'text' if 'input_text' isn't available
        text = item.get('input_text', item.get('text', ''))
        true_label = int(item['labels']) if 'labels' in item else int(item.get('label', 0))
        
        # We need length. If input_ids exists we use it, else tokenize
        if 'input_ids' in item:
            length = len(item['input_ids'])
        else:
            length = len(tokenizer.encode(text, truncation=False))
        
        # Pipeline prediction (classifier takes raw text)
        pred = classifier(text, truncation=True, max_length=model.config.max_position_embeddings)[0]
        # label maps like "LABEL_1" -> 1. Adapt mapping if appropriate
        pred_label = int(pred['label'].split('_')[-1]) if 'LABEL' in pred['label'] else int(pred['label'])
        
        error_type = "Correct"
        if true_label == 1 and pred_label == 0:
            error_type = "False Negative (Missed Hal)"
        elif true_label == 0 and pred_label == 1:
            error_type = "False Positive (False Hal)"
            
        results.append({
            "idx": idx,
            "length": length,
            "true_label": true_label,
            "pred_label": pred_label,
            "error_type": error_type
        })
        
    df = pd.DataFrame(results)
    
    # Define length buckets
    bins = [0, 512, 1024, 2048, 4096, 8192, float('inf')]
    labels = ["<512", "512-1024", "1024-2048", "2048-4096", "4096-8192", ">8192"]
    df['length_bucket'] = pd.cut(df['length'], bins=bins, labels=labels, right=False)
    
    # Calculate error rates per bucket
    error_summary = df.groupby(['length_bucket', 'error_type'], observed=False).size().unstack(fill_value=0)
    
    # Plotting
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='length_bucket', hue='error_type', multiple="stack", shrink=0.8)
    plt.title('Prediction Distribution by Input Length')
    plt.xlabel('Token Length Bucket')
    plt.ylabel('Count')
    
    model_name = Path(model_path).name
    plot_file = out_dir / f"{model_name}_errors_by_length.png"
    plt.savefig(plot_file)
    plt.close()
    
    # Save CSV of errors for manual inspection
    errors_df = df[df['error_type'] != "Correct"]
    csv_file = out_dir / f"{model_name}_misclassifications.csv"
    errors_df.to_csv(csv_file, index=False)
    
    print(f"Analysis saved! Plot: {plot_file}, Data: {csv_file}")
    
    return error_summary
