import inspect
import logging

from transformers import PreTrainedModel, PretrainedConfig, BertConfig, BertModel

from swag.posteriors.swag import SWAG


logger = logging.getLogger(__name__)


class SwagBertConfig(PretrainedConfig):

    model_type = 'swag_bert'
    internal_config_class = BertConfig

    def __init__(
            self,
            no_cov_mat: bool = True,
            max_num_models: int = 20,
            var_clamp: float = 1e-30,
            **kwargs
    ):
        internal_params = list(inspect.signature(self.internal_config_class).parameters)
        internal_kwargs = {
            name: kwargs.pop(name)
            for name in internal_params
            if name in kwargs
        }
        super().__init__(**kwargs)
        internal_config = self.internal_config_class(**internal_kwargs)
        self.no_cov_mat = no_cov_mat
        self.max_num_models = max_num_models
        self.var_clamp = var_clamp
        self.internal_model_config = internal_config.to_dict()


class SwagBertPreTrainedModel(PreTrainedModel):

    config_class = SwagBertConfig
    base_model_prefix = 'swag_bert'
    internal_model_class = BertModel

    def __init__(self, config):
        super().__init__(config)
        self.model = SWAG(
            base=self.internal_model_class,
            no_cov_mat=config.no_cov_mat,
            max_num_models=config.max_num_models,
            var_clamp=config.var_clamp,
            config=config.internal_config_class(**config.internal_model_config),
            device='cpu'  # FIXME: how to deal with device
        )

    def _init_weights(self, module):
        # What should be here?
        raise NotImplementedError("TODO")


class SwagBertModel(SwagBertPreTrainedModel):

    def forward(self, *args, **kwargs):
        return self.model.forward(*args, **kwargs)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    config = SwagBertConfig(no_cov_mat=False, hidden_size=240)
    print(config)
    model = SwagBertPreTrainedModel(config)
    print(model)
