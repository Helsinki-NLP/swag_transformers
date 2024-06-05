import logging
import unittest
import tempfile

import numpy as np
import torch

from datasets import Dataset, DatasetDict
from transformers import AutoConfig, AutoModel, AutoModelForSequenceClassification, \
    AutoTokenizer, DataCollatorWithPadding, Trainer, TrainingArguments, BartForConditionalGeneration

from swag_transformers.swag_bart import SwagBartConfig, SwagBartModel, SwagBartPreTrainedModel, \
    SwagBartForSequenceClassification, SwagBartForConditionalGeneration
from swag_transformers.trainer_utils import SwagUpdateCallback


class TestSwagBart(unittest.TestCase):

    pretrained_model_name = 'Finnish-NLP/bart-small-finnish'
    # pretrained_model_name = 'sshleifer/bart-tiny-random'

    def test_untrained(self):
        hidden_size = 240
        config = SwagBartConfig(no_cov_mat=False, hidden_size=hidden_size)
        logging.debug(config)
        swag_model = SwagBartPreTrainedModel(config)
        swag_model = SwagBartModel(config)
        logging.debug(swag_model)
        self.assertEqual(swag_model.device.type, 'cpu')
        with self.assertLogs(level='WARNING') as cm:
            out = swag_model.forward(input_ids=torch.tensor([[3, 14]]))
            # Warning from using forward before sampling parameters
            self.assertTrue(any(msg.startswith('WARNING') for msg in cm.output))
        logging.debug(out)
        swag_model.swag.sample()
        out = swag_model.forward(input_ids=torch.tensor([[3, 14]]))
        logging.debug(out)
        self.assertEqual(out.last_hidden_state.shape, (1, 2, hidden_size))

    def test_pretrained_bart_generative(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = BartForConditionalGeneration.from_pretrained(self.pretrained_model_name)
        model.to(device)
        self.assertEqual(model.device.type, device)
        swag_model = SwagBartForConditionalGeneration.from_base(model)
        swag_model.to(device)
        self.assertEqual(swag_model.device.type, device)
        tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model_name)

        swag_model.swag.collect_model(model)
        swag_model.swag.sample()

        # Test forward
        base_out = model.forward(input_ids=torch.tensor([[3, 14]]), decoder_input_ids=torch.tensor([[1, 2, 4]]))
        out = swag_model.forward(input_ids=torch.tensor([[3, 14]]), decoder_input_ids=torch.tensor([[1, 2, 4]]))
        self.assertTrue(torch.allclose(base_out.logits, out.logits))

        # Test generate
        example = "I have no BART and I must generate"
        batch = tokenizer(example, return_tensors="pt")
        base_generated_ids = model.generate(batch["input_ids"])
        base_out = tokenizer.batch_decode(base_generated_ids, skip_special_tokens=True)
        generated_ids = swag_model.generate(batch["input_ids"])
        out = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        logging.info(base_out)
        logging.info(out)
        self.assertEqual(base_out, out)

        # Test saving & loading
        with tempfile.TemporaryDirectory() as tempdir:
            swag_model.save_pretrained(tempdir)
            stored_model = SwagBartForConditionalGeneration.from_pretrained(tempdir).to(device)

        generated_ids = stored_model.generate(batch["input_ids"])
        out = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        logging.info(out)
        self.assertEqual(base_out, out)



if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()
