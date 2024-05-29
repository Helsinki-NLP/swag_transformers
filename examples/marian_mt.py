import argparse
import collections
import logging
import os
import sys

import torch
import transformers
import datasets

from swag_transformers.swag_marian import SwagMarianMTModel
from swag_transformers.trainer_utils import SwagUpdateCallback


logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Fine-tune pre-trained BERT for SNLI using SWAG")
    parser.add_argument("--base-model", type=str, default="Helsinki-NLP/opus-mt-de-en")
    parser.add_argument("--device", type=str, help="set device (default: cuda if available, otherwise cpu)")
    parser.add_argument("--save-folder", type=str, default="save_folder")
    parser.add_argument("--limit-training", type=int, help="limit training data to N first samples")
    parser.add_argument("--epochs", type=int, default=3, help="number of training epochs")
    args = parser.parse_args()

    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = transformers.AutoTokenizer.from_pretrained(args.base_model)
    model = transformers.MarianMTModel.from_pretrained(args.base_model)
    model.to(device)
    swag_model = SwagMarianMTModel.from_base(model)
    swag_model.to(device)

    dataset = datasets.load_dataset("Helsinki-NLP/opus-100", "de-nl", split="test")
    if args.limit_training:
        dataset = dataset.select(range(args.limit_training))
    logger.info(dataset)

    max_input_length = 128
    max_target_length = 128

    def tokenize_function(example):
        inputs = [pair['de'] for pair in example['translation']]
        targets = [pair['nl'] for pair in example['translation']]
        model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)

        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, max_length=max_target_length, truncation=True)

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    data_collator = transformers.DataCollatorForSeq2Seq(
        tokenizer=tokenizer, model=model
    )

    training_args = transformers.TrainingArguments(
        output_dir=args.save_folder,
        learning_rate=2e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=args.epochs,
        use_cpu=True if device == "cpu" else False
    )

    trainer = transformers.Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets,
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
