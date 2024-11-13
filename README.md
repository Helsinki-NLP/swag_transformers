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
2. Initialize SWAG model by `swag_model = SwagBertForSequenceClassification.from_base(base_model, no_cov_mat=False)`
3. Initialize SWAG callback object `swag_callback = SwagUpdateCallback(swag_model)`
4. Initialize `transformers.Trainer` with the `base_model` as model and `swag_callback` in callbacks.
5. Train the model (`trainer.train()`)
6. Store the complete model using `swag_model.save_pretrained(path)`

Note that `trainer.save_model(path)` will save only the base model without the distribution parameters from SWAG.

For collecting the SWAG parameters, two possible schedules are supported:

* After the end of each training epoch (default, `collect_steps = 0` for `SwagUpdateCallback`)
* After each N training steps (set `collect_steps > 0` for `SwagUpdateCallback`)

### SWA, SWAG-Diagonal, and SWAG

The library supports both SWA (stochastic weight averaging) and two
variants of SWAG (SWA-Gaussian): SWAG-Diagonal that uses diagonal
covariance and "full" SWAG that does low-rank covariance matrix
estimation.

The method is selected by the `no_cov_mat` attribute when initializing
the model (e.g. `SwagModel.from_base(model, no_cov_mat=True)`). The
default value `True` works only with SWAG-Diagonal and SWA, and you
need to explicitly set `no_cov_mat=False` to activate the low-rank
covariance estimation of SWAG. Note that you can also test SWA and
SWAG-Diagonal methods when the model is trained with
`no_cov_mat=False` (see the next section).

With SWAG, the `max_num_models` option controls the maximum rank of
the covariance matrix. The rank is increased by each parameter
collection step until the maximum is reached. The current rank is
stored in `model.swag.cov_mat_rank` and automatically updated to
`model.config.cov_mat_rank` when using `SwagUpdateCallback`. If you
call `model.swag.collect_model()` manually, you should also update the
configuration accordingly before saving the model.

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

Note that both for `sample_parameters()` and `get_logits()` the
default keyword arguments are suitable only for SWAG-Diagonal. For
SWAG, you should use `cov=True` (required to use the covariance
matrix) and `scale=0.5` (recommended). For SWA, you should use
`cov=False` and `scale=0`. To summarize:

* SWA: `scale=0`, `cov=False`
* SWAG-Diagonal: `scale=1`, `cov=False` (defaults)
* SWAG: `scale=0.5`, `cov=True` (`no_cov_mat=False` required for the model)

### Currently supported models

* BERT (bidirectional encoder)
  * `BertPreTrainedModel` -> `SwagBertPreTrainedModel`
  * `BertModel` -> `SwagBertModel`
  * `BertLMHeadModel` -> `SwagBertLMHeadModel`
  * `BertForSequenceClassification` -> `SwagBertForSequenceClassification`
  * `BertForQuestionAnswering` -> `SwagBertForQuestionAnswering`
* BART (bidirectional encoder + causal decoder)
  * `BartPreTrainedModel` -> `SwagBartPreTrainedModel`
  * `BartModel` -> `SwagBartModel`
  * `BartForConditionalGeneration` -> `SwagBartForConditionalGeneration`
  * `BartForSequenceClassification` -> `SwagBartForSequenceClassification`
* MarianMT (bidirectional encoder + causal decoder)
  * `MarianPreTrainedModel` -> `SwagMarianPreTrainedModel`
  * `MarianModel` -> `SwagMarianModel`
  * `MarianMTModel` -> `SwagMarianMTModel`
