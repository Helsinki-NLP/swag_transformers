import copy
import logging
import unittest
import tempfile

import pytest
import torch

from datasets import Dataset, DatasetDict
from transformers import AutoModel, AutoModelForSequenceClassification, AutoModelForQuestionAnswering, \
    AutoTokenizer, DataCollatorWithPadding, Trainer, TrainingArguments

from swag_transformers.swag_roberta import SwagRobertaConfig, SwagRobertaModel, SwagRobertaPreTrainedModel, \
    SwagRobertaForSequenceClassification
from swag_transformers.trainer_utils import SwagUpdateCallback



class TestSwagRoberta(unittest.TestCase):

    def test_untrained(self):
        hidden_size = 240
        config = SwagRobertaConfig(no_cov_mat=False, hidden_size=hidden_size)
        logging.debug(config)
        swag_model = SwagRobertaPreTrainedModel(config)
        swag_model = SwagRobertaModel(config)
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
        print(swag_model.swag.base.embeddings.word_embeddings.weight_mean)

    def test_untrained_classifier(self):
        hidden_size = 240
        num_labels = 3
        config = SwagRobertaConfig(no_cov_mat=False, hidden_size=hidden_size, num_labels=num_labels)
        logging.debug(config)
        swag_model = SwagRobertaForSequenceClassification(config)
        swag_model.sample_parameters()
        logging.debug(swag_model)
        logging.debug(swag_model.swag.base.config)
        self.assertEqual(swag_model.device.type, 'cpu')
        out = swag_model.forward(input_ids=torch.tensor([[3, 14]]))
        logging.debug(out)
        self.assertEqual(out.logits.shape, (1, num_labels))
        print(swag_model.swag.base.roberta.embeddings.word_embeddings.weight_mean)


class TestSwagRobertaFinetune(unittest.TestCase):

    pretrained_model_name = 'facebookai/roberta-base'
    num_labels = 2
    device = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def setUpClass(cls):
        cls.base_model = AutoModelForSequenceClassification.from_pretrained(
            cls.pretrained_model_name, num_labels=cls.num_labels)
        cls.base_model.to(cls.device)
        cls.tokenizer = AutoTokenizer.from_pretrained(cls.pretrained_model_name)

    @staticmethod
    def _data_gen():
        yield {"text": "Hello world", "label": 0}
        yield {"text": "Just some swaggering", "label": 1}
        yield {"text": "Have a good day", "label": 0}
        yield {"text": "You swaggy boy!", "label": 1}
        yield {"text": "That's so bad", "label": 0}
        yield {"text": "This is SWAG", "label": 1}

    @pytest.mark.slow
    def test_untrained(self):
        hidden_size = 240
        config = SwagRobertaConfig(no_cov_mat=False, hidden_size=hidden_size)
        logging.debug(config)
        swag_model = SwagRobertaPreTrainedModel(config)
        swag_model = SwagRobertaModel(config)
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
        print(swag_model.swag.base.embeddings.word_embeddings.weight_mean)

    def pretrained_bert_classifier_finetune(self, no_cov_mat, module_prefix_list=None):
        train_epochs = 5
        model = copy.deepcopy(self.base_model)
        tokenizer = self.tokenizer
        swag_model = SwagRobertaForSequenceClassification.from_base(
            model, no_cov_mat=no_cov_mat, module_prefix_list=module_prefix_list)
        swag_model.to(self.device)
        self.assertEqual(model.device.type, self.device)
        self.assertEqual(swag_model.device.type, self.device)
        tokens = tokenizer(["Hello world", "Just some swaggering"],
                           padding=True, truncation=False, return_tensors="pt").to(self.device)
        out_base = model(**tokens)
        self.assertEqual(out_base.logits.shape, (2, self.num_labels))
        out_swag = swag_model(**tokens)
        self.assertEqual(out_swag.logits.shape, (2, self.num_labels))
        self.assertTrue(torch.allclose(out_swag.logits.to('cpu'), torch.zeros(*out_swag.logits.shape)))
        swag_model.sample_parameters(cov=not no_cov_mat)
        out_swag = swag_model(**tokens)
        self.assertEqual(out_swag.logits.shape, (2, self.num_labels))
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
                use_cpu=True if self.device == "cpu" else False
            )
            trainer = Trainer(
                model,
                training_args,
                train_dataset=tokenized_datasets["train"],
                data_collator=data_collator,
                processing_class=tokenizer,
                callbacks=[SwagUpdateCallback(swag_model)]
            )
            trainer.train()
        self.assertEqual(swag_model.swag.n_models, train_epochs)
        return swag_model

    def finetuned_model_test(self, swag_model, no_cov_mat=True, blockwise=False, scale=1):
        tokens = self.tokenizer(["Hello world", "Just some swaggering"],
                                padding=True, truncation=False, return_tensors="pt").to(self.device)
        swag_model.sample_parameters(cov=not no_cov_mat, block=blockwise, scale=scale, seed=1234)
        out_swag = swag_model(**tokens)
        self.assertEqual(out_swag.logits.shape, (2, self.num_labels))
        # Test saving & loading
        with tempfile.TemporaryDirectory() as tempdir:
            swag_model.save_pretrained(tempdir)
            stored_model = SwagRobertaForSequenceClassification.from_pretrained(tempdir).to(self.device)
        stored_model.sample_parameters(cov=not no_cov_mat, block=blockwise, scale=scale, seed=1234)
        out_stored = stored_model(**tokens)
        logging.debug(out_swag.logits)
        logging.debug(out_stored.logits)
        self.assertTrue(torch.allclose(out_swag.logits, out_stored.logits))

    @pytest.mark.slow
    def test_pretrained_bert_classifier_finetune_no_cov(self):
        model = self.pretrained_bert_classifier_finetune(no_cov_mat=True)
        self.finetuned_model_test(model, no_cov_mat=True, blockwise=False, scale=0)  # SWA
        self.finetuned_model_test(model, no_cov_mat=True, blockwise=False, scale=1)  # SWAG-Diag

    @pytest.mark.slow
    def test_pretrained_bert_classifier_finetune_no_cov_partial(self):
        model = self.pretrained_bert_classifier_finetune(
            no_cov_mat=True, module_prefix_list=['roberta.embeddings.word_embeddings', 'classifier'])
        self.finetuned_model_test(model, no_cov_mat=True, blockwise=False, scale=0)  # SWA
        self.finetuned_model_test(model, no_cov_mat=True, blockwise=False, scale=1)  # SWAG-Diag


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()
