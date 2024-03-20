import logging
import unittest

import torch

from transformers import AutoConfig, AutoModel, AutoModelForSequenceClassification, AutoTokenizer

from swag_transformers.swag_bert import SwagBertConfig, SwagBertModel, SwagBertPreTrainedModel, \
    SwagBertForSequenceClassification


class TestSwagBert(unittest.TestCase):

    pretrained_model_name = 'prajjwal1/bert-tiny'

    def test_untrained(self):
        hidden_size = 240
        config = SwagBertConfig(no_cov_mat=False, hidden_size=hidden_size)
        logging.debug(config)
        swag_model = SwagBertPreTrainedModel(config)
        logging.debug(swag_model)
        swag_model = SwagBertModel(config)
        logging.debug(swag_model)
        with self.assertLogs(level='WARNING') as cm:
            out = swag_model.forward(input_ids=torch.tensor([[3, 14]]))
            # Warning from using forward before sampling parameters
            self.assertTrue(any(msg.startswith('WARNING') for msg in cm.output))
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

    def test_pretrained_bert_tiny(self):
        model = AutoModel.from_pretrained(self.pretrained_model_name)
        hidden_size = model.config.hidden_size
        config = SwagBertConfig.from_config(model.config, no_cov_mat=False)
        logging.debug(config)
        swag_model = SwagBertModel.from_base(model)
        logging.debug(swag_model)
        swag_model.model.sample()
        out = swag_model.forward(input_ids=torch.tensor([[3, 14]]))
        logging.debug(out)
        self.assertEqual(out.last_hidden_state.shape, (1, 2, hidden_size))

    def test_pretrained_bert_tiny_classifier(self):
        num_labels = 4
        model = AutoModelForSequenceClassification.from_pretrained(self.pretrained_model_name, num_labels=num_labels)
        swag_model = SwagBertForSequenceClassification.from_base(model)
        logging.debug(swag_model)
        swag_model.model.sample()
        out = swag_model.forward(input_ids=torch.tensor([[3, 14]]))
        logging.debug(out)
        self.assertEqual(out.logits.shape, (1, num_labels))


if __name__ == "__main__":
    unittest.main()
