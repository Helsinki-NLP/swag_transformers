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

### Fine-tuning with SWAG

BERT model, sequence classification task:

1. Load pretrained Bert model by `base_model = AutoModelForSequenceClassification.from_pretrained(name_or_path)`
2. Initialize SWAG model by `swag_model = SwagBertForSequenceClassification.from_base(base_model)`
3. Initialize SWAG callback object `swag_callback = SwagUpdateCallback(swag_model)`
4. Initialize `transformers.Trainer` with the `base_model` as model and `swag_callback` in callbacks.
5. Train the model (`trainer.train()`)
6. Store the complete model using `swag_model.save_pretrained(path)`

Note that `trainer.save_model(path)` will save only the base model without the distribution parameters from SWAG.

For collecting the SWAG parameters, two possible schedules are supported:

* After the end of each training epoch (default, `collect_steps = 0` for `SwagUpdateCallback`)
* After each N training steps (set `collect_steps > 0` for `SwagUpdateCallback`)

### Sampling model parameters

After `swag_model` is trained or fine-tuned as described above,
`swag_model.sample_parameters()` should be called to sample new model
parameters. After that, `swag_model.forward()` can be used to predict
new output from classifiers and `swag_model.generate()` to generate
new output from generative LMs. In order to get a proper distribution
of outputs, `sample_parameters()` needs to be called each time before
`forward()` or `generate()`. For classifiers, the `SampleLogitsMixin`
class provides the convenience method `get_logits()` that samples the
parameters and makes a new prediction `num_predictions` times, and
returns the logit values in a tensor.

### Currently supported models

* BERT (bidirectional encoder)
  * `BertPreTrainedModel` -> `SwagBertPreTrainedModel`
  * `BertModel` -> `SwagBertModel`
  * `BertLMHeadModel` -> `SwagBertLMHeadModel`
  * `BertForSequenceClassification` -> `SwagBertForSequenceClassification`
* BART (bidirectional encoder + causal decoder)
  * `BartPreTrainedModel` -> `SwagBartPreTrainedModel`
  * `BartModel` -> `SwagBartModel`
  * `BartForConditionalGeneration` -> `SwagBartForConditionalGeneration`
  * `BartForSequenceClassification` -> `SwagBartForSequenceClassification`
* MarianMT (bidirectional encoder + causal decoder)
  * `MarianPreTrainedModel` -> `SwagMarianPreTrainedModel`
  * `MarianModel` -> `SwagMarianModel`
  * `MarianMTModel` -> `SwagMarianMTModel`
