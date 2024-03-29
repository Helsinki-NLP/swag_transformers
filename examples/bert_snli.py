import argparse
import collections
import logging
import os
import sys

import torch
import transformers
import datasets

from swag_transformers.swag_bert import SwagBertForSequenceClassification
from swag_transformers.trainer_utils import SwagUpdateCallback


logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Fine-tune pre-trained BERT for SNLI using SWAG")
    parser.add_argument("--base-model", type=str, default="bert-base-uncased")
    parser.add_argument("--save-folder", type=str, default="save_folder")
    parser.add_argument("--limit-training", type=int, help="limit training data to N first samples")
    args = parser.parse_args()

    tokenizer = transformers.AutoTokenizer.from_pretrained(args.base_model)
    model = transformers.AutoModelForSequenceClassification.from_pretrained(args.base_model, num_labels=3)
    swag_model = SwagBertForSequenceClassification.from_base(model)

    dataset = datasets.load_dataset("snli", split="train")
    dataset = dataset.filter(lambda example: example["label"] != -1)
    if args.limit_training:
        dataset = dataset.select(range(args.limit_training))

    def tokenize_dataset(dataset):
        processed = tokenizer(dataset["premise"], dataset["hypothesis"],
                              padding="max_length", max_length=128, truncation=True)
        processed["labels"] = dataset["label"]
        return processed

    training_args = transformers.TrainingArguments(
        output_dir="save_folder",
        learning_rate=2e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=3,
    )

    processed_dataset = dataset.map(
        tokenize_dataset, batched=True, remove_columns=dataset.column_names)

    data_collator = transformers.DataCollatorWithPadding(tokenizer=tokenizer)
    trainer = transformers.Trainer(
        model=model,
        args=training_args,
        train_dataset=processed_dataset,
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
