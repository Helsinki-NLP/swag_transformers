"""PyTorch SWAG wrapper for Marian models"""

import logging

from transformers import MarianConfig, MarianModel, MarianMTModel
from transformers.models.marian import MarianPreTrainedModel

from .base import SwagConfig, SwagPreTrainedModel, SwagModel


logger = logging.getLogger(__name__)


MODEL_TYPE = 'swag_marian'


class SwagMarianConfig(SwagConfig):
    """Config for Marian model averaging with SWAG"""

    model_type = MODEL_TYPE
    internal_config_class = MarianConfig


class SwagMarianPreTrainedModel(SwagPreTrainedModel):
    """Pretrained SWAG Marian model"""

    config_class = SwagMarianConfig
    base_model_prefix = MODEL_TYPE
    internal_model_class = MarianPreTrainedModel


class SwagMarianModel(SwagModel):
    """SWAG Marian model"""

    config_class = SwagMarianConfig
    base_model_prefix = MODEL_TYPE
    internal_model_class = MarianModel


class SwagMarianMTModel(SwagMarianModel):
    """SWAG MarianMT model"""

    internal_model_class = MarianMTModel
