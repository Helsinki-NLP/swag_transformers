import logging
import unittest
import tempfile

import numpy as np
import torch

from datasets import Dataset, DatasetDict
from transformers import AutoConfig, AutoModel, AutoModelForSequenceClassification, AutoModelWithLMHead, \
    AutoTokenizer, DataCollatorWithPadding, Trainer, TrainingArguments

from swag_transformers.swag_bert import SwagBertConfig, SwagBertLMHeadModel, SwagBertModel, SwagBertPreTrainedModel, \
    SwagBertForSequenceClassification
from swag_transformers.trainer_utils import SwagUpdateCallback


class TestSwagBert(unittest.TestCase):

    pretrained_model_name = 'prajjwal1/bert-tiny'

    def test_untrained(self):
        hidden_size = 240
        config = SwagBertConfig(no_cov_mat=False, hidden_size=hidden_size)
        logging.debug(config)
        swag_model = SwagBertPreTrainedModel(config)
        swag_model = SwagBertModel(config)
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
        print(swag_model.swag.base.embeddings.token_type_embeddings.weight_mean)

    def test_untrained_classifier(self):
        hidden_size = 240
        num_labels = 3
        config = SwagBertConfig(no_cov_mat=False, hidden_size=hidden_size, num_labels=num_labels)
        logging.debug(config)
        swag_model = SwagBertForSequenceClassification(config)
        swag_model.sample_parameters()
        logging.debug(swag_model)
        logging.debug(swag_model.swag.base.config)
        self.assertEqual(swag_model.device.type, 'cpu')
        out = swag_model.forward(input_ids=torch.tensor([[3, 14]]))
        logging.debug(out)
        self.assertEqual(out.logits.shape, (1, num_labels))
        print(swag_model.swag.base.bert.embeddings.token_type_embeddings.weight_mean)

    def test_untrained_lmhead(self):
        hidden_size = 128
        vocab_size = 512
        num_attention_heads = 4
        config = SwagBertConfig(
            no_cov_mat=False, num_attention_heads=num_attention_heads, hidden_size=hidden_size,
            vocab_size=vocab_size, is_decoder=True)
        swag_model = SwagBertLMHeadModel(config)
        swag_model.sample_parameters()
        logging.debug(swag_model.config)
        logging.debug(swag_model)
        prep_inputs = swag_model.prepare_inputs_for_generation(input_ids=torch.tensor([[3, 14, 45]]))
        logging.debug(prep_inputs)
        out = swag_model.forward(**prep_inputs)
        logging.debug(out)
        self.assertEqual(out.logits.shape, (1, 3, vocab_size))

    def test_pretrained_bert_tiny_base(self):
        model = AutoModel.from_pretrained(self.pretrained_model_name)
        self.assertEqual(model.device.type, 'cpu')
        hidden_size = model.config.hidden_size
        config = SwagBertConfig.from_config(model.config, no_cov_mat=False)
        logging.debug(config)
        swag_model = SwagBertModel.from_base(model)
        logging.debug(swag_model)
        self.assertEqual(swag_model.device.type, 'cpu')
        swag_model.sample_parameters()
        out = swag_model.forward(input_ids=torch.tensor([[3, 14]]))
        logging.debug(out)
        self.assertEqual(out.last_hidden_state.shape, (1, 2, hidden_size))

    def test_pretrained_bert_tiny_classifier_test(self):
        num_labels = 4
        model = AutoModelForSequenceClassification.from_pretrained(self.pretrained_model_name, num_labels=num_labels)
        print(model.config)
        swag_model = SwagBertForSequenceClassification.from_base(model)
        logging.debug(swag_model)
        self.assertEqual(swag_model.device.type, 'cpu')
        logging.debug(swag_model.device)
        logging.debug(swag_model.swag.device)
        logging.debug(swag_model.swag.base.device)
        swag_model.swag.collect_model(model)
        swag_model.sample_parameters()
        out = swag_model.forward(input_ids=torch.tensor([[3, 14]]))
        logging.debug(out)
        self.assertEqual(out.logits.shape, (1, num_labels))

    def _data_gen(self):
        yield {"text": "Hello world", "label": 0}
        yield {"text": "Just some swaggering", "label": 1}
        yield {"text": "Have a good day", "label": 0}
        yield {"text": "You swaggy boy!", "label": 1}
        yield {"text": "That's so bad", "label": 0}
        yield {"text": "This is SWAG", "label": 1}

    def test_pretrained_bert_tiny_classifier_finetune(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        num_labels = 2
        train_epochs = 5
        model = AutoModelForSequenceClassification.from_pretrained(self.pretrained_model_name, num_labels=num_labels)
        model.to(device)
        self.assertEqual(model.device.type, device)
        swag_model = SwagBertForSequenceClassification.from_base(model)
        swag_model.to(device)
        self.assertEqual(swag_model.device.type, device)
        tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model_name)
        tokens = tokenizer(["Hello world", "Just some swaggering"],
                           padding=True, truncation=False, return_tensors="pt").to(device)
        out_base = model(**tokens)
        self.assertEqual(out_base.logits.shape, (2, num_labels))
        out_swag = swag_model(**tokens)
        self.assertEqual(out_swag.logits.shape, (2, num_labels))
        self.assertTrue(torch.allclose(out_swag.logits.to('cpu'), torch.zeros(*out_swag.logits.shape)))
        swag_model.sample_parameters()
        out_swag = swag_model(**tokens)
        self.assertEqual(out_swag.logits.shape, (2, num_labels))
        self.assertTrue(torch.allclose(out_swag.logits.to('cpu'), torch.zeros(*out_swag.logits.shape)))

        def tokenize_function(example):
            return tokenizer(example["text"], truncation=False)

        dataset = Dataset.from_generator(self._data_gen)
        raw_datasets = DatasetDict({"train": dataset})
        tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        with tempfile.TemporaryDirectory() as tempdir:
            training_args = TrainingArguments(
                output_dir=tempdir,
                num_train_epochs=train_epochs,
                use_cpu=True if device == "cpu" else False
            )
            trainer = Trainer(
                model,
                training_args,
                train_dataset=tokenized_datasets["train"],
                data_collator=data_collator,
                tokenizer=tokenizer,
                callbacks=[SwagUpdateCallback(swag_model)]
            )
            trainer.train()
        self.assertEqual(swag_model.swag.n_models, train_epochs)
        swag_model.sample_parameters()
        out_swag = swag_model(**tokens)
        self.assertEqual(out_swag.logits.shape, (2, num_labels))

        # Test saving & loading
        with tempfile.TemporaryDirectory() as tempdir:
            swag_model.save_pretrained(tempdir)
            stored_model = SwagBertForSequenceClassification.from_pretrained(tempdir).to(device)
        out_stored = stored_model(**tokens)
        logging.debug(out_swag.logits)
        logging.debug(out_stored.logits)
        self.assertTrue(torch.allclose(out_swag.logits, out_stored.logits))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()
