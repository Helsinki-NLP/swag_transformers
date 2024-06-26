"""PyTorch SWAG wrapper for BERT"""

import logging

from transformers import BertConfig, BertLMHeadModel, BertModel, BertPreTrainedModel, \
    BertForSequenceClassification

from .base import SwagConfig, SwagPreTrainedModel, SwagModel, SampleLogitsMixin


logger = logging.getLogger(__name__)


MODEL_TYPE = 'swag_bert'


class SwagBertConfig(SwagConfig):
    """Config for BERT model averaging with SWAG"""

    model_type = MODEL_TYPE
    internal_config_class = BertConfig


class SwagBertPreTrainedModel(SwagPreTrainedModel):
    """Pretrained SWAG BERT model"""

    config_class = SwagBertConfig
    base_model_prefix = MODEL_TYPE
    internal_model_class = BertPreTrainedModel


class SwagBertModel(SwagModel):
    """SWAG BERT model"""

    config_class = SwagBertConfig
    base_model_prefix = MODEL_TYPE
    internal_model_class = BertModel


class SwagBertForSequenceClassification(SampleLogitsMixin, SwagBertModel):
    """SWAG BERT model for sequence classification"""

    internal_model_class = BertForSequenceClassification


class SwagBertLMHeadModel(SwagBertModel):
    """SWAG BERT model with LM head"""

    internal_model_class = BertLMHeadModel
