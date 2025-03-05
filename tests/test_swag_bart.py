import logging
import os
import unittest
import tempfile

import torch

from transformers import AutoTokenizer, BartForConditionalGeneration, GenerationConfig

from swag_transformers.swag_bart import SwagBartConfig, SwagBartModel, SwagBartPreTrainedModel, \
    SwagBartForConditionalGeneration


class TestSwagBart(unittest.TestCase):

    # pretrained_model_name = 'Finnish-NLP/bart-small-finnish'
    pretrained_model_name = 'sshleifer/bart-tiny-random'

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
        swag_model.sample_parameters()
        out = swag_model.forward(input_ids=torch.tensor([[3, 14]]))
        logging.debug(out)
        self.assertEqual(out.last_hidden_state.shape, (1, 2, hidden_size))

    def pretrained_bart_generative(self, no_cov_mat):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = BartForConditionalGeneration.from_pretrained(self.pretrained_model_name)
        model.to(device)
        self.assertEqual(model.device.type, device)
        swag_model = SwagBartForConditionalGeneration.from_base(model, no_cov_mat=no_cov_mat)
        swag_model.to(device)
        self.assertEqual(swag_model.device.type, device)
        tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model_name, clean_up_tokenization_spaces=False)

        gen_config = GenerationConfig.from_model_config(model.config)
        gen_config.max_new_tokens = 10
        logging.debug(gen_config)

        swag_model.swag.collect_model(model)
        swag_model.sample_parameters(cov=not no_cov_mat, seed=1234)
        # has to be updated manually when using collect_model directly
        swag_model.config.cov_mat_rank = swag_model.swag.cov_mat_rank

        # Test forward
        base_fwd_out = model.forward(input_ids=torch.tensor([[3, 14]]), decoder_input_ids=torch.tensor([[1, 2, 4]]))
        swag_fwd_out = swag_model.forward(input_ids=torch.tensor([[3, 14]]), decoder_input_ids=torch.tensor([[1, 2, 4]]))
        self.assertTrue(torch.allclose(base_fwd_out.logits, swag_fwd_out.logits))

        # Test generate
        example = "I have no BART and I must generate"
        batch = tokenizer(example, return_tensors="pt")
        base_generated_ids = model.generate(batch["input_ids"], generation_config=gen_config)
        base_out = tokenizer.batch_decode(base_generated_ids, skip_special_tokens=True)

        generated_ids = swag_model.generate(batch["input_ids"], generation_config=gen_config)
        out = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        self.assertEqual(base_out, out)

        # Test saving & loading
        with tempfile.TemporaryDirectory() as tempdir:
            swag_model.save_pretrained(tempdir)
            logging.debug(os.listdir(tempdir))
            with open(os.path.join(tempdir, 'config.json'), 'r', encoding='utf8') as fobj:
                logging.debug(fobj.read())
            stored_model = SwagBartForConditionalGeneration.from_pretrained(tempdir).to(device)

        stored_model.sample_parameters(cov=not no_cov_mat, seed=1234)
        stored_fwd_out = stored_model.forward(
            input_ids=torch.tensor([[3, 14]]), decoder_input_ids=torch.tensor([[1, 2, 4]]))
        self.assertTrue(torch.allclose(swag_fwd_out.logits, stored_fwd_out.logits, atol=1e-06))

        generated_ids = stored_model.generate(batch["input_ids"], generation_config=gen_config)
        out = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        self.assertEqual(base_out, out)

    def test_pretrained_bart_generative_no_cov(self):
        self.pretrained_bart_generative(no_cov_mat=True)

    def test_pretrained_bart_generative_with_cov(self):
        self.pretrained_bart_generative(no_cov_mat=False)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()
