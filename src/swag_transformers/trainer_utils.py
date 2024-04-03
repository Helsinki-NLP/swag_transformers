"""Utils for training the SWAG models"""

import logging

from swag.posteriors.swag import SWAG
from transformers import TrainerCallback


logger = logging.getLogger(__name__)


class SwagUpdateCallback(TrainerCallback):
    """Callback for updating SWAG parameters during training

    Use with transformers.Trainer to collect the SWAG parameters
    during training of the base model. Example:

    swag_model = SwagBertForSequenceClassification.from_base(base_model)
    trainer = transformers.Trainer(
        model=base_model,
        ...
        callbacks=[SwagUpdateCallback(swag_model)]
    )
    trainer.train()

    """

    def __init__(self, swag_model):
        self.main_model = swag_model

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        if model is None:
            logger.error("No model provided for SWAG update")
            return
        logger.debug("Updating SWAG parameters from %s", type(model).__name__)
        self.main_model.swag.collect_model(model)
        self.main_model.config.update_internal_config(model.config)
