"""PyTorch SWAG wrapper for BART"""

import logging

from transformers import BartConfig, BartModel, BartPreTrainedModel, BartForConditionalGeneration, \
    BartForSequenceClassification

from .base import SwagConfig, SwagPreTrainedModel, SwagModel, SampleLogitsMixin


logger = logging.getLogger(__name__)


MODEL_TYPE = 'swag_bart'


class SwagBartConfig(SwagConfig):
    """Config for BART model averaging with SWAG"""

    model_type = MODEL_TYPE
    internal_config_class = BartConfig


class SwagBartPreTrainedModel(SwagPreTrainedModel):
    """Pretrained SWAG BART model"""

    config_class = SwagBartConfig
    base_model_prefix = MODEL_TYPE
    internal_model_class = BartPreTrainedModel


class SwagBartModel(SwagModel):
    """SWAG BART model"""

    config_class = SwagBartConfig
    base_model_prefix = MODEL_TYPE
    internal_model_class = BartModel


class SwagBartForConditionalGeneration(SwagBartModel):
    """SWAG BART model for sequence classification"""

    internal_model_class = BartForConditionalGeneration


class SwagBartForSequenceClassification(SampleLogitsMixin, SwagBartModel):
    """SWAG BART model for sequence classification"""

    internal_model_class = BartForSequenceClassification
