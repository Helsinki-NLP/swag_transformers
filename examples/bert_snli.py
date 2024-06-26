import argparse
import logging
import os

import torch
import transformers
import datasets

from swag_transformers.swag_bert import SwagBertForSequenceClassification
from swag_transformers.trainer_utils import SwagUpdateCallback


logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Fine-tune pre-trained BERT for SNLI using SWAG")
    parser.add_argument("--base-model", type=str, default="bert-base-uncased")
    parser.add_argument("--device", type=str, help="set device (default: cuda if available, otherwise cpu)")
    parser.add_argument("--save-folder", type=str, default="save_folder")
    parser.add_argument("--limit-training", type=int, help="limit training data to N first samples")
    parser.add_argument("--data-cache-dir", type=str, help="folder to cache HF datasets")
    parser.add_argument("--model-cache-dir", type=str, help="folder to cache HF models and tokenizers")
    parser.add_argument("--batch-size", type=int, default=4, help="batch size")
    parser.add_argument("--epochs", type=int, default=3, help="number of training epochs")
    parser.add_argument("--collect-steps", type=int, default=100, help="number of steps between collecting parameters")
    parser.add_argument("--learning-rate", type=float, default=2e-5, help="learning rate")
    args = parser.parse_args()

    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = transformers.AutoTokenizer.from_pretrained(args.base_model, cache_dir=args.model_cache_dir)
    model = transformers.AutoModelForSequenceClassification.from_pretrained(
        args.base_model, num_labels=3, cache_dir=args.model_cache_dir)
    model.to(device)
    swag_model = SwagBertForSequenceClassification.from_base(model)
    swag_model.to(device)

    dataset = datasets.load_dataset("snli", split="train", cache_dir=args.data_cache_dir)
    dataset = dataset.filter(lambda example: example["label"] != -1)
    if args.limit_training:
        dataset = dataset.select(range(args.limit_training))

    def tokenize_dataset(dataset):
        processed = tokenizer(dataset["premise"], dataset["hypothesis"],
                              padding="max_length", max_length=128, truncation=True)
        processed["labels"] = dataset["label"]
        return processed

    training_args = transformers.TrainingArguments(
        output_dir=args.save_folder,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        use_cpu=True if device == "cpu" else False
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
        callbacks=[SwagUpdateCallback(swag_model, collect_steps=args.collect_steps)]
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
