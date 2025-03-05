"""PyTorch SWAG wrapper for ModernBERT"""

import logging

from transformers import ModernBertConfig, ModernBertModel, ModernBertPreTrainedModel, \
    ModernBertForMaskedLM, ModernBertForSequenceClassification, ModernBertForTokenClassification

from .base import SwagConfig, SwagPreTrainedModel, SwagModel, SampleLogitsMixin


logger = logging.getLogger(__name__)


MODEL_TYPE = 'swag_modernbert'


class SwagModernBertConfig(SwagConfig):
    """Config for ModernBERT model averaging with SWAG"""

    model_type = MODEL_TYPE
    internal_config_class = ModernBertConfig


class SwagModernBertPreTrainedModel(SwagPreTrainedModel):
    """Pretrained SWAG ModernBERT model"""

    config_class = SwagModernBertConfig
    base_model_prefix = MODEL_TYPE
    internal_model_class = ModernBertPreTrainedModel


class SwagModernBertModel(SwagModel):
    """SWAG ModernBERT model"""

    config_class = SwagModernBertConfig
    base_model_prefix = MODEL_TYPE
    internal_model_class = ModernBertModel


class SwagModernBertForMaskedLM(SwagModernBertModel):
    """SWAG ModernBERT model for masked LM task"""

    internal_model_class = ModernBertForMaskedLM


class SwagModernBertForSequenceClassification(SampleLogitsMixin, SwagModernBertModel):
    """SWAG ModernBERT model for sequence classification"""

    internal_model_class = ModernBertForSequenceClassification


class SwagModernBertForTokenClassification(SwagModernBertModel):
    """SWAG ModernBERT model for token classification"""

    internal_model_class = ModernBertForTokenClassification
