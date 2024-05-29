import logging
import unittest
import tempfile

import torch

from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, DataCollatorForSeq2Seq, Seq2SeqTrainer, \
    Seq2SeqTrainingArguments, MarianMTModel

from swag_transformers.swag_marian import SwagMarianConfig, SwagMarianModel, SwagMarianMTModel, \
    SwagMarianPreTrainedModel
from swag_transformers.trainer_utils import SwagUpdateCallback


def buf_and_param_names(model):
    """Return set of parameter and buffer names prefixed with BUF: or PAR:"""
    named_buffers = set('BUF:' + x[0] for x in model.named_buffers())
    named_params = set('PAR:' + x[0] for x in model.named_parameters())
    bufs_and_params = named_buffers | named_params
    return sorted(bufs_and_params)


class TestSwagMarian(unittest.TestCase):

    pretrained_model_name = 'sshleifer/tiny-marian-en-de'

    def test_untrained(self):
        hidden_size = 240
        vocab_size = 1024  # default 58101
        input_dict = {'input_ids': torch.tensor([[3, 14]]), 'decoder_input_ids': torch.tensor([[0]])}
        config = SwagMarianConfig(no_cov_mat=False, hidden_size=hidden_size, vocab_size=vocab_size,
                                  decoder_start_token_id=vocab_size-1, pad_token_id=vocab_size-1)
        logging.debug(config)

        swag_model = SwagMarianPreTrainedModel(config)
        logging.debug(swag_model)
        swag_model = SwagMarianModel(config)
        logging.debug(swag_model)
        swag_model.swag.sample()
        logging.debug(swag_model.swag.base.decoder.embed_tokens.weight)
        out = swag_model.forward(**input_dict)
        self.assertEqual(out.last_hidden_state.shape, (1, 1, hidden_size))

        swag_model = SwagMarianMTModel(config)
        logging.debug(swag_model)
        swag_model.swag.sample()
        logging.debug(swag_model.swag.base.model.decoder.embed_tokens.weight)
        out = swag_model.forward(**input_dict)
        logging.debug(out.logits.shape)
        self.assertEqual(out.encoder_last_hidden_state.shape, (1, 2, hidden_size))
        self.assertEqual(out.logits.shape, (1, 1, vocab_size))

    def test_pretrained_marian_tiny_test(self):
        model = MarianMTModel.from_pretrained(self.pretrained_model_name)
        logging.debug(model)
        tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model_name)
        hidden_size = model.config.hidden_size
        config = SwagMarianConfig.from_config(model.config, no_cov_mat=False)
        logging.debug(config)
        swag_model = SwagMarianMTModel.from_base(model)
        logging.debug(swag_model)

        # The amount of (non-duplicate) parameters should match
        self.assertEqual(len(swag_model.swag.params), len(list(model.parameters())))

        bufs_and_params_before = set(buf_and_param_names(swag_model))

        swag_model.swag.collect_model(model)
        swag_model.swag.sample()

        bufs_and_params_after = set(buf_and_param_names(swag_model))
        self.assertEqual(bufs_and_params_before, bufs_and_params_after)

        logging.debug(model.device)
        logging.debug(swag_model.device)
        logging.debug(swag_model.swag.device)
        logging.debug(swag_model.swag.base.device)

        # Tied parameters should point to the same data
        shared_pointer = swag_model.swag.base.model.shared.weight.untyped_storage().data_ptr()
        logging.debug(shared_pointer)
        logging.debug(swag_model.swag.base.model.encoder.embed_tokens.weight.untyped_storage().data_ptr())
        logging.debug(swag_model.swag.base.model.decoder.embed_tokens.weight.untyped_storage().data_ptr())
        logging.debug(swag_model.swag.base.lm_head.weight.untyped_storage().data_ptr())
        self.assertEqual(shared_pointer, swag_model.swag.base.model.encoder.embed_tokens.weight.untyped_storage().data_ptr())
        self.assertEqual(shared_pointer, swag_model.swag.base.model.decoder.embed_tokens.weight.untyped_storage().data_ptr())
        self.assertEqual(shared_pointer, swag_model.swag.base.lm_head.weight.untyped_storage().data_ptr())

        # Test forward
        base_out = model.forward(input_ids=torch.tensor([[3, 14]]), decoder_input_ids=torch.tensor([[1, 2, 4]]))
        out = swag_model.forward(input_ids=torch.tensor([[3, 14]]), decoder_input_ids=torch.tensor([[1, 2, 4]]))
        self.assertEqual(out.encoder_last_hidden_state.shape, (1, 2, hidden_size))
        self.assertTrue(torch.allclose(base_out.logits, out.logits))

        # Test generate
        sample_text = "what is so great ?"
        batch = tokenizer([sample_text], return_tensors="pt")
        generated_ids = model.generate(**batch)
        base_output = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        logging.debug(base_output)
        self.assertGreater(len(base_output), 0)
        generated_ids = swag_model.generate(**batch)
        output = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        logging.debug(output)
        self.assertGreater(len(output), 0)
        self.assertEqual(base_output, output)

    def _data_gen(self):
        yield {"source": "India and Japan prime ministers meet in Tokyo",
               "target": "Die Premierminister Indiens und Japans trafen sich in Tokio."}
        yield {"source": "High on the agenda are plans for greater nuclear co-operation.",
               "target": "Pläne für eine stärkere kerntechnische Zusammenarbeit stehen ganz oben auf der Tagesordnung."}
        yield {"source": "India is also reportedly hoping for a deal on defence collaboration between the two nations.",
               "target": ("Berichten zufolge hofft Indien darüber hinaus auf einen Vertrag zur "
                          "Verteidigungszusammenarbeit zwischen den beiden Nationen.")}
        yield {"source": "Karratha police arrest 20-year-old after high speed motorcycle chase",
               "target": "Polizei von Karratha verhaftet 20-Jährigen nach schneller Motorradjagd"}

    def test_pretrained_marian_tiny_finetune(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = MarianMTModel.from_pretrained(self.pretrained_model_name)
        model.to(device)
        self.assertEqual(model.device.type, device)
        tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model_name)
        logging.info("Init from base")
        swag_model = SwagMarianMTModel.from_base(model)
        swag_model.to(device)
        self.assertEqual(swag_model.device.type, device)
        logging.info("Done")

        max_input_length = 128
        max_target_length = 128

        def tokenize_function(example):
            inputs = example['source']
            targets = example['target']
            model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)

            # Setup the tokenizer for targets
            with tokenizer.as_target_tokenizer():
                labels = tokenizer(targets, max_length=max_target_length, truncation=True)

            model_inputs["labels"] = labels["input_ids"]
            return model_inputs

        dataset = Dataset.from_generator(self._data_gen)
        raw_datasets = DatasetDict({"train": dataset})
        tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer, model=model
        )
        train_epochs = 60
        logging.debug(model.lm_head.weight)
        logging.debug(model.model.encoder.embed_tokens.weight)
        logging.debug(model.model.decoder.embed_tokens.weight)
        with tempfile.TemporaryDirectory() as tempdir:
            training_args = Seq2SeqTrainingArguments(
                output_dir=tempdir,
                num_train_epochs=train_epochs,
                use_cpu=True if device == "cpu" else False
            )
            trainer = Seq2SeqTrainer(
                model,
                training_args,
                train_dataset=tokenized_datasets["train"],
                data_collator=data_collator,
                tokenizer=tokenizer,
                callbacks=[SwagUpdateCallback(swag_model, collect_steps=2)]
            )
            trainer.train()
        logging.info("N models: %s", swag_model.swag.n_models.item())
        # self.assertEqual(swag_model.swag.n_models, train_epochs)
        swag_model.swag.sample()
        sample_text = "India and Japan prime ministers meet in Tokyo"
        batch = tokenizer([sample_text], return_tensors="pt")
        generated_ids = model.generate(**batch, max_new_tokens=10)
        base_output = tokenizer.batch_decode(generated_ids, skip_special_tokens=False)[0]
        logging.debug(base_output)
        self.assertGreater(len(base_output), 0)
        generated_ids = swag_model.generate(**batch, max_new_tokens=10)
        logging.debug(generated_ids)
        output = tokenizer.batch_decode(generated_ids, skip_special_tokens=False)[0]
        logging.debug(output)
        self.assertGreater(len(output), 0)

        # Test saving & loading
        with tempfile.TemporaryDirectory() as tempdir:
            swag_model.save_pretrained(tempdir)
            logging.info("Loading stored model")
            stored_model = SwagMarianMTModel.from_pretrained(tempdir)

        tied_params_orig = set(x[0][0] for x in swag_model.swag.tied_params)
        tied_params_stored = set(x[0][0] for x in stored_model.swag.tied_params)
        self.assertEqual(tied_params_orig, tied_params_stored)

        orig_embed = swag_model.swag.base.model.shared.weight.to('cpu')
        for rnd in range(3):
            loaded_embed = stored_model.swag.base.model.shared.weight
            loaded_enc = stored_model.swag.base.model.encoder.embed_tokens.weight
            loaded_head = stored_model.swag.base.lm_head.weight
            logging.debug("\nORIG:%s\nNEW:%s\nENC:%s\nHEAD:%s", orig_embed, loaded_embed, loaded_enc, loaded_head)
            if rnd == 0:
                # before sampling
                self.assertTrue(torch.allclose(orig_embed, loaded_embed))
            self.assertTrue(torch.allclose(loaded_embed, loaded_enc))
            self.assertTrue(torch.allclose(loaded_embed, loaded_head))
            stored_model.swag.sample()
