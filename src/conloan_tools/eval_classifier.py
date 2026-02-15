import json
import argparse
import torch
import numpy as np
from pathlib import Path

from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
)
from datasets import DatasetDict
from seqeval.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)

from conloan_tools.conloan_data import (
    tokenize_and_align_labels,
    LoanLabel,
)


def align_predictions(predictions, label_ids, id_to_label):
    preds = np.argmax(predictions, axis=2)
    true_labels = []
    pred_labels = []

    for pred_seq, label_seq in zip(preds, label_ids):
        true_seq = []
        pred_seq_out = []
        for p, l in zip(pred_seq, label_seq):
            if l == -100:
                continue
            true_seq.append(id_to_label[l])
            pred_seq_out.append(id_to_label[p])
        true_labels.append(true_seq)
        pred_labels.append(pred_seq_out)

    return true_labels, pred_labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--splits_dir",
        required=True,
        help="Directory containing saved train/dev/test splits",
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=["train", "dev", "test"],
        help="Which split to evaluate on",
    )
    parser.add_argument(
        "--model", required=True, help="Path to trained model checkpoint"
    )
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument(
        "--output", default=None, help="Optional JSON output path"
    )
    args = parser.parse_args()

    # 1. Label mappings
    label_to_id = {
        label.name.replace("_", "-"): label.value for label in LoanLabel
    }
    id_to_label = {v: k for k, v in label_to_id.items()}

    # 2. Load split
    splits = DatasetDict.load_from_disk(args.splits_dir)
    dataset = splits[args.split]
    print(f"Evaluating on '{args.split}' split ({len(dataset)} examples)")

    # 3. Tokenize
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenized_ds = dataset.map(
        lambda x: tokenize_and_align_labels(x, tokenizer, label_to_id),
        batched=True,
    )

    # 4. Load model
    model = AutoModelForTokenClassification.from_pretrained(args.model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # 5. Predict
    collator = DataCollatorForTokenClassification(tokenizer)
    dataloader = torch.utils.data.DataLoader(
        tokenized_ds.remove_columns(
            [
                c
                for c in tokenized_ds.column_names
                if c not in ("input_ids", "attention_mask", "labels")
            ]
        ),
        batch_size=args.batch_size,
        collate_fn=collator,
    )

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            all_preds.append(outputs.logits.cpu().numpy())
            all_labels.append(batch["labels"].cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # 6. Evaluate
    true_labels, pred_labels = align_predictions(
        all_preds, all_labels, id_to_label
    )

    report = classification_report(true_labels, pred_labels, digits=4)
    f1 = f1_score(true_labels, pred_labels)
    prec = precision_score(true_labels, pred_labels)
    rec = recall_score(true_labels, pred_labels)

    print(report)
    print(f"Overall  P={prec:.4f}  R={rec:.4f}  F1={f1:.4f}")

    # 7. Optional JSON dump
    if args.output:
        results = {
            "split": args.split,
            "num_examples": len(dataset),
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "report": report,
        }
        Path(args.output).write_text(json.dumps(results, indent=2))
        print(f"Results written to {args.output}")


if __name__ == "__main__":
    main()
