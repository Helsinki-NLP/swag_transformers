"""SWAG wrapper base classes"""

from abc import ABCMeta, abstractmethod
import copy
import functools
import logging
from typing import Union, Type

import torch

from transformers import PreTrainedModel, PretrainedConfig

from swag.posteriors.swag import SWAG


logger = logging.getLogger(__name__)


class SwagConfig(PretrainedConfig):
    """Base configuration class for SWAG models

    For using this class, inherit it and define the following class
    attributes:

    - model_type: string
    - internal_config_class: class inherited from PretrainedConfig

    """

    internal_config_class: Type[PretrainedConfig] = PretrainedConfig

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
    def from_config(cls, base_config: PretrainedConfig, **kwargs):
        """Initialize from existing PretrainedConfig"""
        config = cls(**kwargs)
        config.internal_model_config = base_config.to_dict()
        return config

    def update_internal_config(self, base_config: PretrainedConfig):
        """Update internal config from base_config"""
        self.internal_model_config = base_config.to_dict()
        # Copy some things to the top level
        if base_config.problem_type is not None:
            self.problem_type = base_config.problem_type


class SwagPreTrainedModel(PreTrainedModel):
    """Base class for SWAG models wrapping PreTrainedModel

    For using this class, inherit it and define the following class
    attributes:

    - base_model_prefix: string
    - config_class: class inherited from PretrainedConfig
    - internal_model_class: class inherited from PreTrainedModel

    """

    config_class: Type[SwagConfig] = SwagConfig
    internal_model_class: Type[PreTrainedModel] = PreTrainedModel

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

    def sample_parameters(self):
        """Sample new model parameters"""
        self.swag.sample()


class SwagModel(SwagPreTrainedModel):
    """Base class for SWAG models

    For using this class, inherit it and define the following class
    attributes:

    - base_model_prefix: string
    - config_class: class inherited from PretrainedConfig
    - internal_model_class: class inherited from PreTrainedModel

    """

    def __init__(self, config, base_model=None):
        super().__init__(config)
        if base_model:
            self.swag = SWAG(
                base=functools.partial(self._base_model_copy, base_model),
                no_cov_mat=config.no_cov_mat,
                max_num_models=config.max_num_models,
                var_clamp=config.var_clamp
            )
        self.prepare_inputs_for_generation = self.swag.base.prepare_inputs_for_generation
        self.generate = self.swag.base.generate

    @staticmethod
    def _base_model_copy(model, *args, **kwargs):
        """Return deep copy of the model ignoring other arguments"""
        # Has to be copied, otherwise SWAG would initialize parameters
        # of the original model to zero
        model = copy.deepcopy(model)
        model.tie_weights()
        return model

    @classmethod
    def from_base(cls, base_model: PreTrainedModel, **kwargs):
        """Initialize from existing PreTrainedModel"""
        config = cls.config_class.from_config(base_model.config, **kwargs)
        swag_model = cls(config, base_model=base_model)
        return swag_model

    def forward(self, *args, **kwargs):
        """Call forward pass from the base model"""
        return self.swag.forward(*args, **kwargs)

    @classmethod
    def can_generate(cls) -> bool:
        return cls.internal_model_class.can_generate()

    def prepare_inputs_for_generation(self, *args, **kwargs):
        return self.swag.base.prepare_inputs_for_generation(*args, **kwargs)

    def generate(self, *args, **kwargs):
        return self.swag.base.generate(*args, **kwargs)


class SampleLogitsMixin:
    """Mixin class for classification models providing get_logits() method using SWAG"""

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
                self.sample_parameters(scale=scale, cov=cov, block=block)
            out = self.forward(*args, **kwargs)
            logits.append(out.logits)
        logits = torch.permute(torch.stack(logits), (1, 0, 2))  # [batch_size, num_predictions, output_size]
        return logits
