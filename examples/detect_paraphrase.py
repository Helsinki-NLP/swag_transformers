import argparse
import logging
import os
import bz2
import random
import json

import torch
import transformers
import datasets
import evaluate

import numpy as np

from datasets import Dataset, DatasetDict, load_dataset

from transformers import EarlyStoppingCallback

from swag_transformers.swag_bert import SwagBertForSequenceClassification
from swag_transformers.trainer_utils import SwagUpdateCallback

logger = logging.getLogger(__name__)


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
    # Download only the validation and test sets for the given language.
    dataset = load_dataset("GEM/opusparcus", f"{language}.100", trust_remote_code=True, cache_dir="./tmp")

    dataset = dataset.rename_column("input", "sentence1")
    dataset = dataset.rename_column("target", "sentence2")
    dataset = dataset.remove_columns("references")

    dataset["validation"] = dataset["validation.full"]
    dataset["test"] = dataset["test.full"]

    dataset_dict = DatasetDict({
        "train": train_data["train"],
        "validation": dataset["validation"],
        "test": dataset["test"],
    })

    dataset_dict["train"] = dataset_dict["train"].shuffle()

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

    # negative_sources, negative_targets = [], []

    # Sampling negative examples
    if negatives == "same":
        # sample from the same data distribution
        # (sentence1 randomly from all data, sentence2 from the current training set)
        negative_samples = load_dataset("json", data_files=f"{negative_examples}")
        negative_samples = negative_samples["train"].select(range(num_negatives))

        # with open(negative_examples, "rt") as f:
        #     for i, line in enumerate(f):
        #         _, s1, s2, *_ = line.split("\t")
        #         negative_sources.append(s1)
        #         negative_targets.append(s2)
        #         if len(negative_sources) == num_negatives:
        #             break

    # elif negatives == "random":
    #     # sample randomly from all data
    #     with bz2.open(negative_examples, "rt") as f:
    #         choices = random.sample(f.readlines(), num_negatives)
    #         for choice in choices:
    #             _, s1, s2, *_ = choice.split("\t")
    #             negative_sources.append(s1)
    #             negative_targets.append(s2)

    # elif negatives == "after":
    #     # sample from data after the positive examples
    #     with bz2.open(negative_examples, "rt") as f:
    #         for i, line in enumerate(f):
    #             if i < num_negatives:
    #                 continue
    #             _, s1, s2, *_ = line.split("\t")
    #             negative_sources.append(s1)
    #             negative_targets.append(s2)
    #             if len(negative_sources) == num_negatives:
    #                 break

        # missing = num_negatives - len(negative_sources)
        # if missing > 0:
        #     startind = num_positives - missing
        #     with bz2.open(negative_examples, "rt") as f:
        #         for i, line in enumerate(f):
        #             if i < startind:
        #                 continue
        #             if len(negative_sources) == num_negatives:
        #                 break
        #             _, s1, s2, *_ = line.split("\t")
        #             negative_sources.append(s1)
        #             negative_targets.append(s2)

    else:
        ValueError(negatives)

    dataset["test.full"] = dataset["test.full"].filter(
        lambda x: x["annot_score"] != 2.5
    )
    dataset["validation.full"] = dataset["validation.full"].filter(
        lambda x: x["annot_score"] != 2.5
    )

    dataset["validation"] = dataset["validation.full"]
    dataset["test"] = dataset["test.full"]

    # random.shuffle(negative_sources)

    # num_negatives = len(negative_sources)
    # negative_data = datasets.Dataset.from_dict({
    #     "lang": ["en"]*num_negatives,
    #     "sentence1": negative_sources,
    #     "sentence2": negative_targets,
    #     "annot_score": [-1]*num_negatives,
    #     "gem_id": [f"neg{i}" for i in range(num_negatives)],
    # }, features=dataset["train"].features)

    dataset["train"] = datasets.concatenate_datasets(
        [dataset["train"], negative_samples]
    )

    # cols_to_remove = [k for k in list(dataset["train"][0].keys()) if k not in ["labels"]]

    dataset["train"] = dataset["train"].shuffle()

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
    parser.add_argument("--eval_data", type=str, help="Path to validation dataset (json)")
    parser.add_argument("--test_data", type=str, help="Path to test dataset (json)")
    parser.add_argument("--language", type=str, help="Language of the data (en, fi, fr, de, ru, sv)")
    parser.add_argument("--quality", type=int, help="Estimated clean label proportion for the Opusparcus dataset (95, 90, 85, 80, 75, 70, 65, 60)")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_steps", type=int, help="Max steps for training")
    parser.add_argument("--train_epochs", type=int, help="Number of training epochs")
    parser.add_argument("--collect_steps", type=int, default=100,
                        help="number of steps between collecting parameters; set to zero for per epoch updates")
    parser.add_argument("--eval_strategy", type=str, default="steps", help="Evaluation strategy, either steps or epoch.")
    parser.add_argument("--cache", type=str, default="./tmp", help="Temporary directory for storing models and data downloaded from HF.")
    parser.add_argument("--no_cov_mat", action="store_true", help="If active, train without covariance matrix calculation (SWA model training)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = transformers.AutoTokenizer.from_pretrained(args.base_model, cache_dir="./tmp")
    model = transformers.AutoModelForSequenceClassification.from_pretrained(args.base_model, num_labels=2, cache_dir="./tmp")
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

        processed["labels"] = [annotation_to_label(val) for val in examples["annot_score"]]
        return processed

    metric = evaluate.load("accuracy")


    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        accuracy = metric.compute(predictions=predictions, references=labels)
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
    )

    processed_train = dataset["train"].map(
        tokenize_dataset, batched=True, remove_columns=dataset["train"].column_names)
    processed_eval = dataset["validation"].map(
        tokenize_dataset, batched=True, remove_columns=dataset["validation"].column_names)
    processed_test = dataset["test"].map(
        tokenize_dataset, batched=True, remove_columns=dataset["test"].column_names)

    data_collator = transformers.DataCollatorWithPadding(tokenizer=tokenizer)

    if args.eval_strategy == "steps":
        callbacks = [SwagUpdateCallback(swag_model, collect_steps=args.collect_steps), EarlyStoppingCallback(early_stopping_patience=10)]
    else:
        callbacks = [SwagUpdateCallback(swag_model, collect_steps=args.collect_steps)]

    trainer = transformers.Trainer(
        model=model,
        args=training_args,
        train_dataset=processed_train,
        eval_dataset=processed_eval,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
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
