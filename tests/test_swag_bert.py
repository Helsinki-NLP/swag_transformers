import logging
import unittest

import torch

from transformers import AutoConfig

from swag_transformers.swag_bert import *


class TestSwagBert(unittest.TestCase):

    def test_untrained(self):
        hidden_size = 240
        config = SwagBertConfig(no_cov_mat=False, hidden_size=hidden_size)
        logging.debug(config)
        swag_model = SwagBertPreTrainedModel(config)
        logging.debug(swag_model)
        swag_model = SwagBertModel(config)
        logging.debug(swag_model)
        out = swag_model.forward(input_ids=torch.tensor([[3, 14]]))
        logging.debug(out)
        swag_model.model.sample()
        out = swag_model.forward(input_ids=torch.tensor([[3, 14]]))
        logging.debug(out)
        self.assertEqual(out.last_hidden_state.shape, (1, 2, hidden_size))

    def test_untrained_classifier(self):
        hidden_size = 240
        num_labels = 3
        config = SwagBertConfig(no_cov_mat=False, hidden_size=hidden_size, num_labels=num_labels)
        logging.debug(config)
        swag_model = SwagBertForSequenceClassification(config)
        swag_model.model.sample()
        logging.debug(swag_model)
        logging.debug(swag_model.model.base.config)
        out = swag_model.forward(input_ids=torch.tensor([[3, 14]]))
        logging.debug(out)
        self.assertEqual(out.logits.shape, (1, num_labels))




if __name__ == "__main__":
    unittest.main()
