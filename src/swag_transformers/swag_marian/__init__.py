"""PyTorch SWAG wrapper for Marian models"""

import copy
import functools
import logging
from typing import Union

import torch

from transformers import PreTrainedModel, PretrainedConfig, MarianConfig, \
    MarianModel, MarianMTModel
from transformers.models.marian import MarianPreTrainedModel

from swag.posteriors.swag import SWAG


logger = logging.getLogger(__name__)


class SwagMarianConfig(PretrainedConfig):
    """Config for Marian model averaging with SWAG"""

    model_type = 'swag_marian'
    internal_config_class = MarianConfig

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
    def from_config(cls, base_config: MarianConfig, **kwargs):
        """Initialize from existing MarianConfig"""
        config = cls(**kwargs)
        config.internal_model_config = base_config.to_dict()
        return config

    def update_internal_config(self, base_config: MarianConfig):
        """Update internal config from base_config"""
        self.internal_model_config = base_config.to_dict()
        # Copy some things to the top level
        if base_config.problem_type is not None:
            self.problem_type = base_config.problem_type


class SwagMarianPreTrainedModel(PreTrainedModel):
    """Pretrained SWAG Marian model"""

    config_class = SwagMarianConfig
    base_model_prefix = 'swag_marian'
    internal_model_class = MarianPreTrainedModel

    def __init__(self, config):
        super().__init__(config)
        self.swag = SWAG(
            base=self.new_base_model,
            no_cov_mat=config.no_cov_mat,
            max_num_models=config.max_num_models,
            var_clamp=config.var_clamp,
            config=config.internal_config_class(**config.internal_model_config),
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


class SwagMarianModel(SwagMarianPreTrainedModel):
    """SWAG Marian model"""

    internal_model_class = MarianModel

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
    def from_base(cls, base_model: MarianPreTrainedModel, **kwargs):
        """Initialize from existing MarianPreTrainedModel"""
        config = SwagMarianConfig.from_config(base_model.config, **kwargs)
        swag_model = cls(config, base_model=base_model)
        return swag_model

    def forward(self, *args, **kwargs):
        return self.swag.forward(*args, **kwargs)


class SwagMarianMTModel(SwagMarianModel):
    """SWAG MarianMT model"""

    internal_model_class = MarianMTModel

    def generate(self, *args, **kwargs):
        return self.swag.base.generate(*args, **kwargs)
