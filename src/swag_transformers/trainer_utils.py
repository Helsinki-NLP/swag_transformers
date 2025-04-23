"""Utils for training the SWAG models"""

import logging
import os

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

    Two possible schedules for the updates are currently supported: If
    collect_steps > 0 is provided, the parameters are collected after
    each collect_steps training steps. Otherwise, and as default, the
    parameters are collected on the end of each training epoch.

    The skip_first option can be set to N > 0 for skipping the first N
    collect steps that would have been otherwise done.

    If save_folder is provided, after each update the model is saved
    under it in the subdirectory swag-checkpoint-{steps}.

    """

    def __init__(self, swag_model, collect_steps=None, skip_first=0, save_folder=None):
        self.main_model = swag_model
        self.collect_steps = collect_steps
        self.last_collect_step = None
        self.skips = skip_first
        self.save_folder = save_folder

    def _update(self, model):
        """Update SWAG model parameters and config"""
        if self.skips > 0:
            logger.debug("Skipping SWAG update, %s skips left", self.skips)
            self.skips -= 1
            return
        self.main_model.swag.collect_model(model)
        self.main_model.config.update_internal_config(model.config)
        self.main_model.config.cov_mat_rank = self.main_model.swag.cov_mat_rank

    def on_train_end(self, args, state, control, model=None, **kwargs):
        if self.last_collect_step == state.global_step:
            return
        if model is None:
            logger.error("No model provided for SWAG update")
            return
        logger.debug("Updating SWAG parameters from %s after train end (steps %s)",
                     type(model).__name__, state.global_step)
        self._update(model)

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        if self.collect_steps:
            return
        if model is None:
            logger.error("No model provided for SWAG update")
            return
        logger.debug("Updating SWAG parameters from %s after epoch end (steps %s)",
                     type(model).__name__, state.global_step)
        self._update(model)
        self.last_collect_step = state.global_step
        if self.save_folder:
            self.main_model.save_pretrained(os.path.join(self.save_folder, f'swag-checkpoint-{state.global_step}'))

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if not self.collect_steps:
            return
        if not state.global_step:
            return
        if state.global_step % self.collect_steps != 0:
            return
        if model is None:
            logger.error("No model provided for SWAG update")
            return
        logger.debug("Updating SWAG parameters from %s after step %s",
                     type(model).__name__, state.global_step)
        self._update(model)
        self.last_collect_step = state.global_step
        if self.save_folder:
            self.main_model.save_pretrained(os.path.join(self.save_folder, f'swag-checkpoint-{state.global_step}'))
