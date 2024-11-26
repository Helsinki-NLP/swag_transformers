import argparse
import logging
import os
import random

import torch
import torch.nn as nn
import transformers
import datasets
import evaluate

import numpy as np

from datasets import DatasetDict, load_dataset

from torch.nn import KLDivLoss

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    EarlyStoppingCallback,
    Trainer,
)

from swag_transformers.swag_bert import SwagBertForSequenceClassification
from swag_transformers.trainer_utils import SwagUpdateCallback

logger = logging.getLogger(__name__)


class TrainerWithCustomLoss(Trainer):
    def __init__(self, *args, loss_function=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_function = loss_function or KLDivLoss(reduction="batchmean")

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        confidence = inputs.pop("confidence", None) # Handle optional confidence values
        outputs = model(**inputs)

        # Compute the custom KL-Divergence loss
        logits = outputs.logits

        if isinstance(self.loss_function, KLDivLoss):
            print(f"Distributions in KLDivLoss:\nLogits (after log_softmax): {torch.log_softmax(logits, dim=-1)}\nLabels: {labels}")
            loss = self.loss_function(torch.log_softmax(logits, dim=-1), labels)
        elif isinstance(self.loss_function, CustomCrossEntropyLoss):
            loss = self.loss_function(logits, labels, confidence)
        else:
            ValueError("Unsupported loss function provided.")

        return (loss, outputs) if return_outputs else loss


class CustomCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(CustomCrossEntropyLoss, self).__init__()

    def forward(self, logits, labels, confidence):
        loss_fn = nn.CrossEntropyLoss(reduction='none')
        loss = loss_fn(logits, labels.argmax(dim=-1))

        print(f"Weighting cross-entropy loss with labels: {labels} and confidence: {confidence}")

        if confidence is not None:
            confidence_weights = confidence * (labels.argmax(dim=-1) == 1).float() + \
                                (1 - confidence) * (labels.argmax(dim=-1) == 0).float()
            loss = loss * confidence_weights

        return loss.mean()


def annotation_to_label(annotation):
    if (annotation == 0) or (annotation > 2.5):
        return 1
    elif (annotation == -1) or (annotation <= 2.5):
        return 0


def read_data(path_to_train_data, language):
    '''
    Read an existing training data file
    and add validation and test sets.
    '''
    train_data = load_dataset("json", data_files=path_to_train_data)
    train_data = train_data.remove_columns("rank_scores") # Remove rank_scores
    # Download only the validation and test sets for the given language.
    dataset = load_dataset("GEM/opusparcus", f"{language}.100", trust_remote_code=True, cache_dir="./tmp")

    dataset = dataset.rename_column("input", "sentence1")
    dataset = dataset.rename_column("target", "sentence2")
    dataset = dataset.remove_columns("references")


    dataset["validation.full"] = dataset["validation.full"].filter(
        lambda x: x["annot_score"] != 2.5
    )
    dataset["test.full"] = dataset["test.full"].filter(
        lambda x: x["annot_score"] != 2.5
    )

    dataset["validation"] = dataset["validation.full"]
    dataset["test"] = dataset["test.full"]

    def dev_confidence(example):
        confidence = round((example["annot_score"] - 1.0) / (4.0 - 1.0), 4)
        return {"label_dist": [round(1 - confidence, 4), confidence]}

    dev_processed = dataset["validation.full"].map(dev_confidence)
    test_processed = dataset["test.full"].map(dev_confidence)

    dataset_dict = DatasetDict({
        "train": train_data["train"],
        # "validation": dataset["validation"],
        # "test": dataset["test"],
        "validation": dev_processed,
        "test": test_processed
    })

    print(f"Dataset dict after processing: {dataset_dict}")

    return dataset_dict


def download_data(negative_examples, negatives, language, quality, num_negatives=None):
    '''
    Downloads Opusparcus training and dev/test sets from Huggingface transformers.
    '''
    dataset = load_dataset(
        "GEM/opusparcus",
        lang=language,
        quality=quality,
        cache_dir="./tmp",
        trust_remote_code=True
    )
    dataset = dataset.rename_column("input", "sentence1")
    dataset = dataset.rename_column("target", "sentence2")
    dataset = dataset.remove_columns("references")

    num_positives = len(dataset["train"])
    num_negatives = num_negatives if num_negatives is not None else num_positives

    # Sampling negative examples
    if negatives == "same":
        # sample from the same data distribution
        # (sentence1 randomly from all data, sentence2 from the current training set)
        negative_samples = load_dataset("json", data_files=f"{negative_examples}")
        negative_samples = negative_samples["train"].select(range(num_negatives))

        # Assign a label distribution of [1.0, 0.0] for all negative examples
        negative_samples = negative_samples.map(lambda example: {"label_dist": [1.0, 0.0]})

    else:
        raise ValueError(f"Unsupported value for 'negatives': {negatives}")

    dataset["test.full"] = dataset["test.full"].filter(
        lambda x: x["annot_score"] != 2.5
    )
    dataset["validation.full"] = dataset["validation.full"].filter(
        lambda x: x["annot_score"] != 2.5
    )

    dataset["validation"] = dataset["validation.full"]
    dataset["test"] = dataset["test.full"]

    dataset = dataset.cast_column("annot_score", datasets.Value("float32"))
    negative_samples = negative_samples.cast_column("annot_score", datasets.Value("float32"))

    dataset["train"] = datasets.concatenate_datasets(
        [dataset["train"], negative_samples]
    )

    return dataset


def main():
    parser = argparse.ArgumentParser(description="Fine-tune pre-trained BERT for paraphrase detection using SWAG")
    parser.add_argument("--base_model", type=str, default="bert-base-uncased")
    parser.add_argument("--save_folder", type=str, default="save_folder")
    parser.add_argument("--limit_training", type=int, help="limit training data to N first samples")
    parser.add_argument("--num_positives", type=int, help="Number of positive examples if limit_training")
    parser.add_argument("--num_negatives", type=int, help="Number of negative examples")
    parser.add_argument("--negatives", type=str, default="same", help="Type of negative sampling (options: same, random, after)")
    parser.add_argument("--negative_data", type=str, help="Data directory of negative examples")
    parser.add_argument("--train_data", type=str, help="Path to training dataset (json)")
    parser.add_argument("--language", type=str, help="Language of the data (en, fi, fr, de, ru, sv)")
    parser.add_argument("--quality", type=int, help="Estimated clean label proportion for the Opusparcus dataset (95, 90, 85, 80, 75, 70, 65, 60)")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_steps", type=int, help="Max steps for training")
    parser.add_argument("--train_epochs", type=int, help="Number of training epochs")
    parser.add_argument("--collect_steps", type=int, default=500,
                        help="number of steps between collecting parameters; set to zero for per epoch updates")
    parser.add_argument("--eval_strategy", type=str, default="steps", help="Evaluation strategy, either steps or epoch.")
    parser.add_argument("--cache", type=str, default="./tmp", help="Temporary directory for storing models and data downloaded from HF.")
    parser.add_argument("--no_cov_mat", action="store_true", help="If active, train without covariance matrix calculation (SWA model training)")

    # Loss function:
    parser.add_argument("--loss_function", type=str, default="kl", help="Loss function: kl or cross_entropy")

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, cache_dir="./tmp")
    model = AutoModelForSequenceClassification.from_pretrained(args.base_model, num_labels=2, cache_dir="./tmp")
    swag_model = SwagBertForSequenceClassification.from_base(model, no_cov_mat=args.no_cov_mat) # True = SWA, False = SWAG
    swag_model.to(device)

    if args.train_data:
        dataset = read_data(args.train_data, args.language)
    else:
        dataset = download_data(
            negative_examples=args.negative_data,
            negatives=args.negatives,
            language=args.language,
            quality=args.quality,
            num_negatives=args.num_negatives,
        )

    if args.limit_training:
        dataset = DatasetDict({
            "train": dataset["train"].select(range(args.limit_training)),
            "validation": dataset["validation"],
            "test": dataset["test"]
        })

    logging.info(f"Training with:\n{dataset}")


    def tokenize_dataset(examples):
        processed = tokenizer(
            examples["sentence1"],
            examples["sentence2"],
            padding=False,
            max_length=tokenizer.model_max_length,
            truncation=True,
        )

        # Use distributions as labels
        processed["labels"] = [np.array(val) for val in examples["label_dist"]]
        # processed["labels"] = [annotation_to_label(val) for val in examples["annot_score"]]
        processed["confidence"] = [val[1] for val in examples["label_dist"]] # Collect confidence for positive class
        return processed

    metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        accuracy = metric.compute(predictions=predictions, references=np.argmax(labels, axis=-1)) # Take argmax when training with label distributions
        return {"accuracy": accuracy["accuracy"]}

    seed = random.randint(1, 10000000)

    training_args = transformers.TrainingArguments(
        output_dir=args.save_folder,
        seed=seed,
        learning_rate=2e-5,
        weight_decay=0.1,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        evaluation_strategy=args.eval_strategy,
        save_strategy=args.eval_strategy,
        max_steps=args.max_steps if args.eval_strategy == "steps" else -1,
        num_train_epochs=args.train_epochs if args.eval_strategy == "epoch" else None,
        eval_steps=500 if args.eval_strategy == "steps" else None,
        save_steps=500 if args.eval_strategy == "steps" else None,
        save_total_limit=1,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        load_best_model_at_end=True,
        remove_unused_columns=False,
    )

    if args.loss_function == "kl":
        loss_function = KLDivLoss(reduction="batchmean")
    elif args.loss_function == "cross_entropy":
        loss_function = CustomCrossEntropyLoss()
    else:
        raise ValueError(f"Unknown loss function: {args.loss_function}")

    cols_to_remove = [k for k in list(dataset["train"][0].keys()) if k not in ["labels", "confidence"]]

    processed_train = dataset["train"].map(
        tokenize_dataset, batched=True, remove_columns=cols_to_remove) # dataset["train"].column_names)
    processed_eval = dataset["validation"].map(
        tokenize_dataset, batched=True, remove_columns=cols_to_remove) # dataset["validation"].column_names)
    processed_test = dataset["test"].map(
        tokenize_dataset, batched=True, remove_columns=cols_to_remove) # dataset["test"].column_names)

    logger.info(f"Example tokenized train sample: {processed_train[0]}")
    logger.info(f"Dataset train label distributions: {processed_train['labels'][:5]}")

    data_collator = transformers.DataCollatorWithPadding(tokenizer=tokenizer)

    if args.eval_strategy == "steps":
        callbacks = [SwagUpdateCallback(swag_model, collect_steps=args.collect_steps), EarlyStoppingCallback(early_stopping_patience=10)]
    else:
        callbacks = [SwagUpdateCallback(swag_model, collect_steps=args.collect_steps)]

    trainer = TrainerWithCustomLoss(
        model=model,
        args=training_args,
        train_dataset=processed_train,
        eval_dataset=processed_eval,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
        loss_function=loss_function,
    )
    trainer.train()
    trainer.save_model(os.path.join(args.save_folder, "final_base"))
    trainer.save_state()

    # Save the full model + tokenizer + training arguments (similar to trainer.save_model)
    final_out = os.path.join(args.save_folder, "final")
    swag_model.save_pretrained(final_out)
    trainer.tokenizer.save_pretrained(final_out)
    torch.save(trainer.args, os.path.join(final_out, transformers.trainer.TRAINING_ARGS_NAME))
    trainer.state.save_to_json(os.path.join(final_out, "trainer_state.json"))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
