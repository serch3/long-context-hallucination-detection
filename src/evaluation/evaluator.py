import os
import json
import logging
from pathlib import Path
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_from_disk
from .metrics import compute_metrics

logger = logging.getLogger(__name__)

class Evaluator:
    def __init__(self, model_path: str, dataset_path: str, output_dir: str = "results/metrics"):
        self.model_path = model_path
        self.dataset_path = dataset_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Loading tokenizer and model from {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        
    def evaluate(self, split: str = "test", batch_size: int = 8):
        """Runs evaluation on the specified data split and saves a JSON report."""
        logger.info(f"Loading dataset from {self.dataset_path}...")
        dataset = load_from_disk(self.dataset_path)
        
        if split not in dataset:
            raise ValueError(f"Split {split} not found in the dataset.")
            
        eval_dataset = dataset[split]
        
        # Simple TrainingArguments just for evaluation (Trainer acts as inferencer)
        args = TrainingArguments(
            output_dir=str(self.output_dir / "tmp"),
            per_device_eval_batch_size=batch_size,
            do_train=False,
            do_eval=True,
            report_to="none"
        )
        
        trainer = Trainer(
            model=self.model,
            args=args,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
        )
        
        logger.info("Starting evaluation...")
        metrics = trainer.evaluate()
        
        # Save metrics to JSON
        model_name = Path(self.model_path).name
        out_file = self.output_dir / f"{model_name}_eval_metrics.json"
        
        with open(out_file, "w") as f:
            json.dump(metrics, f, indent=4)
            
        logger.info(f"Evaluation complete. Report saved to {out_file}")
        return metrics
