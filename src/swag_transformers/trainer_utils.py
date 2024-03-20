"""Utils for training the SWAG models"""

import logging

from swag.posteriors.swag import SWAG
from transformers import TrainerCallback


logger = logging.getLogger(__name__)


class SwagUpdateCallback(TrainerCallback):

    def __init__(self, swag_model):
        if isinstance(swag_model, SWAG):
            self.swag = swag_model
        else:
            # expect SwagBertPreTrainedModel etc.
            self.swag = swag_model.swag

    def on_epoch_end(self, args, state, control, logs=None, model=None, **kwargs):
        if model is None:
            logger.error("No model provided for SWAG update")
            return
        logger.info("Updating SWAG parameters from %s", type(model).__name__)
        self.swag.collect_model(model)
