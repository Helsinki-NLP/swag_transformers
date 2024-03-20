import argparse
import collections
import logging
import os
import sys

import transformers

from swag_transformers.swag_bert import SwagBertConfig, SwagBertForSequenceClassification
from swag_transformers.trainer_utils import SwagUpdateCallback


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
