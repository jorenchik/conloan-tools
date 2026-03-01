import json
import argparse
import torch
from pathlib import Path

from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    DataCollatorForTokenClassification,
)
from transformers.trainer import (
    Trainer,
)
from datasets import Dataset, DatasetDict

from conloan_tools.conloan_data import (
    load_conloan,
    tokenize_and_align_labels,
    LoanLabel,
    build_and_save_splits
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", help="JSON dataset files")
    parser.add_argument(
        "--model", default="distilbert-base-multilingual-cased"
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--output_dir", default="./results")
    parser.add_argument(
        "--splits_dir",
        default="./splits",
        help="Directory to save/load train/dev/test splits",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    splits_dir = Path(args.splits_dir)

    # 1. Setup Data
    label_to_id = {
        label.name.replace("_", "-"): label.value for label in LoanLabel
    }

    if splits_dir.exists():
        print(f"Loading existing splits from {splits_dir}")
        splits = DatasetDict.load_from_disk(str(splits_dir))
    else:
        raw_data = load_conloan(args.inputs)
        splits = build_and_save_splits(raw_data, splits_dir, seed=args.seed)

    # 2. Tokenize
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenized_splits = splits.map(
        lambda x: tokenize_and_align_labels(x, tokenizer, label_to_id),
        batched=True,
    )

    # 3. Initialize Model
    model = AutoModelForTokenClassification.from_pretrained(
        args.model, num_labels=len(label_to_id)
    )

    # 4. Training Arguments (matching Paper 3.3 where possible)
    train_args = TrainingArguments(
        output_dir=args.output_dir,
        eval_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=16,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        save_strategy="epoch",
        use_cpu=not torch.cuda.is_available(),
    )

    # 5. Trainer — evaluate on dev, hold out test for eval_classifier.py
    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=tokenized_splits["train"],
        eval_dataset=tokenized_splits["dev"],
        data_collator=DataCollatorForTokenClassification(tokenizer),
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
