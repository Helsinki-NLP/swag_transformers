import argparse
import logging
import os
import bz2
import random

import torch
import transformers
import datasets

from datasets import DatasetDict
from datasets import load_dataset

from swag_transformers.swag_bert import SwagBertForSequenceClassification
from swag_transformers.trainer_utils import SwagUpdateCallback

logger = logging.getLogger(__name__)


def annotation_to_label(annotation):
    if (annotation == 0) or (annotation > 2.5):
        return 1
    elif (annotation == -1) or (annotation <= 2.5):
        return 0


def download_data(source_data, negatives, language, quality, num_negatives=None):
    '''
    Downloads Opusparcus training and dev/test sets from Huggingface transformers.
    '''
    # if args.use_absolute_data_size:
    #     # Download this to get dev/test and features
    #     dataset = load_dataset("GEM/Opusparcus", f"{lang}.{quality}", cache_dir="./tmp")
    #     dataset = dataset.rename_column("input", "sent1")
    #     dataset = dataset.rename_column("target", "sent2")
    #     dataset = dataset.remove_columns("references")

    #     sent1, sent2, ids = [], [], []
    #     with bz2.open(args.source_data, "rt") as f:
    #         for i, line in enumerate(f):
    #             id, s1, s2, score, *_ = line.split("\t")
    #             sent1.append(s1)
    #             sent2.append(s2)
    #             ids.append(id)
    #             if len(sent1) >= args.num_positives:
    #                 break

    #     train_dataset = datasets.Dataset.from_dict({
    #         "lang": [lang]*args.num_positives,
    #         "sent1": sent1,
    #         "sent2": sent2,
    #         "annot_score": [0.0]*args.num_positives,
    #         "gem_id": [f"pos{i}" for i in range(args.num_positives)],
    #     }, features=dataset["train"].features)
    #     dataset["train"] = train_dataset

    # else:
    dataset = load_dataset(
        "GEM/opusparcus",
        lang=language,
        quality=quality,
        cache_dir="./tmp",
        trust_remote_code=True
    )
    dataset = dataset.rename_column("input", "sent1")
    dataset = dataset.rename_column("target", "sent2")
    dataset = dataset.remove_columns("references")

    num_train_samples = len(dataset["train"])
    num_negatives = num_negatives if num_negatives is not None else num_train_samples
        
    negative_sources, negative_targets = [], []
        
    # Sampling negative examples
    if negatives == "same":
        # sample from the same data distribution
        with bz2.open(source_data, "rt") as f:
            for i, line in enumerate(f):
                _, s1, s2, *_ = line.split("\t")
                negative_sources.append(s1)
                negative_targets.append(s2)
                if len(negative_sources) == num_negatives:
                    break
        
    elif negatives == "random":
        # sample randomly from all data
        with bz2.open(source_data, "rt") as f:
            choices = random.sample(f.readlines(), num_negatives)
            for choice in choices:
                _, s1, s2, *_ = choice.split("\t")
                negative_sources.append(s1)
                negative_targets.append(s2)
        
    elif negatives == "after":
        # sample from data after the positive examples
        with bz2.open(source_data, "rt") as f:
            for i, line in enumerate(f):
                if i < num_negatives:
                    continue
                _, s1, s2, *_ = line.split("\t")
                negative_sources.append(s1)
                negative_targets.append(s2)
                if len(negative_sources) == num_negatives:
                    break

        missing = num_negatives - len(negative_sources)
        if missing > 0:
            startind = num_train_samples - missing
            with bz2.open(source_data, "rt") as f:
                for i, line in enumerate(f):
                    if i < startind:
                        continue
                    if len(negative_sources) == num_negatives:
                        break
                    _, s1, s2, *_ = line.split("\t")
                    negative_sources.append(s1)
                    negative_targets.append(s2)

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

    random.shuffle(negative_sources)

    num_negatives = len(negative_sources)
    negative_data = datasets.Dataset.from_dict({
        "lang": ["en"]*num_negatives,
        "sent1": negative_sources,
        "sent2": negative_targets,
        "annot_score": [-1]*num_negatives,
        "gem_id": [f"neg{i}" for i in range(num_negatives)],
    }, features=dataset["train"].features)

    dataset["train"] = datasets.concatenate_datasets(
        [dataset["train"], negative_data]
    )

    cols_to_remove = [k for k in list(dataset["train"][0].keys()) if k not in ["labels"]]

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
    parser.add_argument("--source_data", type=str, help="Data directory of Opusparcus data")
    parser.add_argument("--train_data", type=str, help="Path to training dataset (json)")
    parser.add_argument("--eval_data", type=str, help="Path to validation dataset (json)")
    parser.add_argument("--test_data", type=str, help="Path to test dataset (json)")
    parser.add_argument("--language", type=str, help="Language of the data (en, fi, fr, de, ru, sv)")
    parser.add_argument("--quality", type=int, help="Estimated clean label proportion for the Opusparcus dataset (95, 90, 85, 80, 75, 70, 65, 60)")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--cache", type=str, default="./tmp", help="Temporary directory for storing models and data downloaded from HF.")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = transformers.AutoTokenizer.from_pretrained(args.base_model, cache_dir="./tmp")
    model = transformers.AutoModelForSequenceClassification.from_pretrained(args.base_model, num_labels=2, cache_dir="./tmp")
    swag_model = SwagBertForSequenceClassification.from_base(model)
    swag_model.to(device)

    dataset = download_data(
        source_data=args.source_data,
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

    
    def tokenize_dataset(examples):
        processed = tokenizer(
            examples["sent1"],
            examples["sent2"],
            padding=False,
            max_length=tokenizer.model_max_length,
            truncation=True,
        )

        processed["labels"] = [annotation_to_label(val) for val in examples["annot_score"]]
        return processed


    training_args = transformers.TrainingArguments(
        output_dir=args.save_folder,
        learning_rate=2e-5,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=3,
        seed=42,
        save_steps=500,
        save_total_limit=1,
    )

    processed_train = dataset["train"].map(
        tokenize_dataset, batched=True, remove_columns=dataset["train"].column_names)
    processed_eval = dataset["validation"].map(
        tokenize_dataset, batched=True, remove_columns=dataset["validation"].column_names)
    processed_test = dataset["test"].map(
        tokenize_dataset, batched=True, remove_columns=dataset["test"].column_names)

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
