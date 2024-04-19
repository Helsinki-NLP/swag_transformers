import json
import random
import datasets

from datasets import load_dataset


def change_test_annot(data):
    # In the future, we might want to keep 2.5s in.
    for example in data:
        if example["annot_score"] > 2.5:
            example["annot_score"] = int(1)
        else:
            example["annot_score"] = int(0)

    return data


def create_train_examples(data):
    pos_data = []
    neg_data = []
    for d in data:
        d["annot_score"] = int(1)
        pos_data.append(d)

    targets = [i["target"] for i in data]
    random.shuffle(targets)

    for d, target in zip(data, targets):
        neg_example = {
            "lang": d["lang"],
            "input": d["input"],
            "target": target,
            "annot_score": int(0),
        }

        neg_data.append(neg_example)

    return pos_data, neg_data


def remove_unnecessary_items(data):
    for d in data:
        d.pop("gem_id", None)
        d.pop("references", None)

    return data


def main():
    # Load datasets:
    fi_datasets = []
    # qualities = [95, 90, 85, 80, 75, 70, 65, 60]
    qualities = [95]
    for v in qualities:
        fi_datasets.append(load_dataset("GEM/opusparcus", f"en.{v}", cache_dir="./tmp", trust_remote_code=True))

    fi_dict = {key: value for key, value in zip(qualities, fi_datasets)}

    # Process validation and test sets:
    valid_set = fi_dict[95]["validation.full"]
    test_set = fi_dict[95]["test.full"]

    valid_set = [d for d in valid_set if d["annot_score"] != 2.5]
    test_set = [d for d in test_set if d["annot_score"] != 2.5]

    valid_data = change_test_annot(valid_set)
    test_data = change_test_annot(test_set)
    valid_data_processed = remove_unnecessary_items(valid_data)
    test_data_processed = remove_unnecessary_items(test_data)

    with open("data/opusparcus.en.dev.json", "w") as fout:
        for d in valid_data:
            fout.write(f"{json.dumps(d, ensure_ascii=False)}\n")

    with open("data/opusparcus.en.test.json", "w") as fout:
        for d in test_data:
            fout.write(f"{json.dumps(d, ensure_ascii=False)}\n")

    # Process training sets
    trainsets = [fi_dict[key]["train"] for key in qualities]

    for k, s in zip(qualities, trainsets):
        pos_examples, neg_examples = create_train_examples(s)
        trainset = pos_examples + neg_examples

        trainset_processed = remove_unnecessary_items(trainset)

        with open(f"data/opusparcus.en.{k}.train.json", "w") as fout:
            for d in trainset_processed:
                fout.write(f"{json.dumps(d, ensure_ascii=False)}\n")


if __name__ == "__main__":
    main()
