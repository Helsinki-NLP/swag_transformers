"""PyTorch SWAG wrapper for RoBERTa"""

import logging

from transformers import RobertaConfig, RobertaModel, RobertaPreTrainedModel, \
    RobertaForCausalLM, RobertaForMaskedLM, RobertaForSequenceClassification, \
    RobertaForMultipleChoice, RobertaForTokenClassification, RobertaForQuestionAnswering

from .base import SwagConfig, SwagPreTrainedModel, SwagModel, SampleLogitsMixin


logger = logging.getLogger(__name__)


MODEL_TYPE = 'swag_roberta'


class SwagRobertaConfig(SwagConfig):
    """Config for RoBERTa model averaging with SWAG"""

    model_type = MODEL_TYPE
    internal_config_class = RobertaConfig


class SwagRobertaPreTrainedModel(SwagPreTrainedModel):
    """Pretrained SWAG RoBERTa model"""

    config_class = SwagRobertaConfig
    base_model_prefix = MODEL_TYPE
    internal_model_class = RobertaPreTrainedModel


class SwagRobertaModel(SwagModel):
    """SWAG RoBERTa model"""

    config_class = SwagRobertaConfig
    base_model_prefix = MODEL_TYPE
    internal_model_class = RobertaModel


class SwagRobertaForCausalLM(SwagRobertaModel):
    """SWAG RoBERTa model for causal LM task"""

    internal_model_class = RobertaForCausalLM


class SwagRobertaForMaskedLM(SwagRobertaModel):
    """SWAG RoBERTa model for masked LM task"""

    internal_model_class = RobertaForMaskedLM


class SwagRobertaForSequenceClassification(SampleLogitsMixin, SwagRobertaModel):
    """SWAG RoBERTa model for sequence classification"""

    internal_model_class = RobertaForSequenceClassification


class SwagRobertaForMultipleChoice(SwagRobertaModel):
    """SWAG RoBERTa model for multiple choice task"""

    internal_model_class = RobertaForMultipleChoice


class SwagRobertaForTokenClassification(SwagRobertaModel):
    """SWAG RoBERTa model for token classification"""

    internal_model_class = RobertaForTokenClassification


class SwagRobertaForQuestionAnswering(SwagRobertaModel):
    """SWAG RoBERTa model for question answering"""

    internal_model_class = RobertaForQuestionAnswering
