# SWAG transformers

This repository provides functionality for Stochastic Weight
Averaging-Gaussian training for Transformer models. The implementation
is tied into two libraries:

* [transformers](https://github.com/huggingface/transformers)
  (maintained by Hugging Face)
* [swa_gaussian](https://github.com/Helsinki-NLP/swa_gaussian)
  (maintained by the Language Technology Research Group at the University of Helsinki)

The goal is to make an implementation that works directly with the
convenience tools in the `transformers` library (e.g. `Pipeline` and
`Trainer`) as well as `evaluator` from the related `evaluate` library.

## Usage

See also [examples](./examples).

### Fine-tuning

BERT model, sequence classification task:

1. Load pretrained Bert model by `base_model = AutoModelForSequenceClassification.from_pretrained(name_or_path)`
2. Initialize SWAG model by `swag_model = SwagBertForSequenceClassification.from_base(base_model)`
3. Initialize SWAG callback object `swag_callback = SwagUpdateCallback(swag_model)`
4. Initialize `transformers.Trainer` with the `base_model` as model and `swag_callback` in callbacks.
5. Train the model (`trainer.train()`)
6. Store the complete model using `swag_model.save_pretrained(path)`

### Currently supported models

* BERT
  * `BertPreTrainedModel` -> `SwagBertPreTrainedModel`
  * `BertModel` -> `SwagBertModel`
  * `BertForSequenceClassification` -> `SwagBertForSequenceClassification`
