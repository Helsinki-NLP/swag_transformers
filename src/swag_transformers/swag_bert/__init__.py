"""PyTorch SWAG wrapper for BERT"""

import copy
import functools
import logging
from typing import Union

import torch

from transformers import PreTrainedModel, PretrainedConfig, BertConfig, BertLMHeadModel, BertModel, \
    BertPreTrainedModel, BertForSequenceClassification

from swag.posteriors.swag import SWAG


logger = logging.getLogger(__name__)


class SwagBertConfig(PretrainedConfig):
    """Config for BERT model averaging with SWAG"""

    model_type = 'swag_bert'
    internal_config_class = BertConfig

    def __init__(
            self,
            internal_model_config: dict = None,
            no_cov_mat: bool = True,
            max_num_models: int = 20,
            var_clamp: float = 1e-30,
            **kwargs
    ):
        super().__init__()
        if internal_model_config:
            self.internal_model_config = internal_model_config
        else:
            internal_config = self.internal_config_class(**kwargs)
            self.internal_model_config = internal_config.to_dict()
        self.no_cov_mat = no_cov_mat
        self.max_num_models = max_num_models
        self.var_clamp = var_clamp

    @classmethod
    def from_config(cls, base_config: BertConfig, **kwargs):
        """Initialize from existing BertConfig"""
        config = cls(**kwargs)
        config.internal_model_config = base_config.to_dict()
        return config

    def update_internal_config(self, base_config: BertConfig):
        """Update internal config from base_config"""
        self.internal_model_config = base_config.to_dict()
        # Copy some things to the top level
        if base_config.problem_type is not None:
            self.problem_type = base_config.problem_type


class SwagBertPreTrainedModel(PreTrainedModel):
    """Pretrained SWAG BERT model"""

    config_class = SwagBertConfig
    base_model_prefix = 'swag_bert'
    internal_model_class = BertModel

    def __init__(self, config):
        super().__init__(config)
        self.swag = SWAG(
            base=self.new_base_model,
            no_cov_mat=config.no_cov_mat,
            max_num_models=config.max_num_models,
            var_clamp=config.var_clamp,
            config=config.internal_config_class(**config.internal_model_config)
        )
        self.post_init()

    @classmethod
    def new_base_model(cls, *args, **kwargs):
        """Return new model of the base class

        Any arguments are passed to the base class constructor.

        """
        model = cls.internal_model_class(*args, **kwargs)
        model.tie_weights()
        return model

    def _init_weights(self, module):
        self.swag.base._init_weights(module)


class SwagBertModel(SwagBertPreTrainedModel):
    """SWAG BERT model"""

    def __init__(self, config, base_model=None):
        super().__init__(config)
        if base_model:
            self.swag = SWAG(
                base=functools.partial(self._base_model_copy, base_model),
                no_cov_mat=config.no_cov_mat,
                max_num_models=config.max_num_models,
                var_clamp=config.var_clamp
            )

    @staticmethod
    def _base_model_copy(model, *args, **kwargs):
        """Return deep copy of the model ignoring other arguments"""
        # Has to be copied, otherwise SWAG would initialize parameters
        # of the original model to zero
        model = copy.deepcopy(model)
        model.tie_weights()
        return model

    @classmethod
    def from_base(cls, base_model: BertPreTrainedModel, **kwargs):
        """Initialize from existing BertPreTrainedModel"""
        config = SwagBertConfig.from_config(base_model.config, **kwargs)
        swag_model = cls(config, base_model=base_model)
        return swag_model

    def forward(self, *args, **kwargs):
        return self.swag.forward(*args, **kwargs)


class SwagBertForSequenceClassification(SwagBertModel):
    """SWAG BERT model for sequence classification"""

    internal_model_class = BertForSequenceClassification

    def get_logits(
        self, *args, num_predictions=None, scale=1.0, cov=True, block=False, **kwargs
    ):
        """Sample model parameters num_predictions times and get logits for the input

        Results in a tensor of size batch_size x num_predictions x output_size.

        """
        if num_predictions is None:
            sample = False
            num_predictions = 1
        else:
            sample = True
        logits = []
        for _ in range(num_predictions):
            if sample:
                self.swag.sample(scale=scale, cov=cov, block=block)
            out = self.forward(*args, **kwargs)
            logits.append(out.logits)
        logits = torch.permute(torch.stack(logits), (1, 0, 2))  # [batch_size, num_predictions, output_size]
        return logits


class SwagBertLMHeadModel(SwagBertModel):
    """SWAG BERT model with LM head"""

    internal_model_class = BertLMHeadModel

    def prepare_inputs_for_generation(self, *args, **kwargs):
        return self.swag.base.prepare_inputs_for_generation(*args, **kwargs)
