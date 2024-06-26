import argparse
import logging

import transformers

from swag_transformers.swag_bert import SwagBertConfig, SwagBertForSequenceClassification


def main():
    parser = argparse.ArgumentParser(description="Load pre-trained SwagBertForSequenceClassification")
    parser.add_argument("path", type=str)
    args = parser.parse_args()

    transformers.AutoConfig.register("swag_bert", SwagBertConfig)
    transformers.AutoModelForSequenceClassification.register(SwagBertConfig, SwagBertForSequenceClassification)

    model = transformers.AutoModelForSequenceClassification.from_pretrained(args.path)
    print(model)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
