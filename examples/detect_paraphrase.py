import argparse
import collections
import itertools
import logging
import os
import sys
import json

import torch
import transformers
import datasets

from datasets import Dataset
from datasets import DatasetDict

from swag_transformers.swag_bert import SwagBertForSequenceClassification
from swag_transformers.trainer_utils import SwagUpdateCallback


logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Fine-tune pre-trained BERT for paraphrase detection using SWAG")
    parser.add_argument("--base_model", type=str, default="bert-base-uncased")
    parser.add_argument("--save_folder", type=str, default="save_folder")
    parser.add_argument("--limit_training", type=int, help="limit training data to N first samples")
    parser.add_argument("--train_data", type=str, help="Path to training dataset (json)")
    parser.add_argument("--eval_data", type=str, help="Path to validation dataset (json)")
    parser.add_argument("--test_data", type=str, help="Path to test dataset (json)")
    args = parser.parse_args()

    tokenizer = transformers.AutoTokenizer.from_pretrained(args.base_model)
    model = transformers.AutoModelForSequenceClassification.from_pretrained(args.base_model, num_labels=3)
    swag_model = SwagBertForSequenceClassification.from_base(model)

    # train_data = [json.loads(line) for line in itertools.islice(args.train_data, args.limit_training)]
    train_data = [json.loads(line) for line in args.train_data]
    eval_data = [json.loads(line) for line in args.eval_data]
    test_data = [json.loads(line) for line in args.test_data]

    train_dataset = Dataset.from_list(train_data)
    eval_dataset = Dataset.from_list(eval_data)
    test_dataset = Dataset.from_list(test_data)

    dataset = DatasetDict({
        "train": train_dataset,
        "validation": eval_dataset,
        "test": test_dataset,
    })

    if args.limit_training:
        dataset = dataset.select(range(args.limit_training))

    def tokenize_dataset(dataset):
        processed = tokenizer(dataset["input"], dataset["target"],
                              padding="max_length", max_length=128, truncation=True)
        processed["labels"] = dataset["annot_score"]
        return processed

    training_args = transformers.TrainingArguments(
        output_dir="save_folder",
        learning_rate=2e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=3,
    )

    processed_train = dataset["train"].map(
        tokenize_dataset, batched=True, remove_columns=dataset.column_names)
    processed_eval = dataset["validation"].map(
        tokenize_dataset, batched=True, remove_columns=dataset.column_names)
    processed_test = dataset["test"].map(
        tokenize_dataset, batched=True, remove_columns=dataset.column_names)

    data_collator = transformers.DataCollatorWithPadding(tokenizer=tokenizer)
    trainer = transformers.Trainer(
        model=model,
        args=training_args,
        train_dataset=processed_train,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[SwagUpdateCallback(swag_model)]
    )
    trainer.train()
    trainer.save_model(os.path.join(args.save_folder, "final_base"))

    # Save the full model + tokenizer + training arguments (similar to trainer.save_model)
    final_out = os.path.join(args.save_folder, "final")
    swag_model.save_pretrained(final_out)
    trainer.tokenizer.save_pretrained(final_out)
    torch.save(trainer.args, os.path.join(final_out, transformers.trainer.TRAINING_ARGS_NAME))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
