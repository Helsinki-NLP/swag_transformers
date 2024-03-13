
import unittest

import torch

from swag_transformers.swag_bert import SwagBertConfig, SwagBertPreTrainedModel, SwagBertModel


class TestSwagBert(unittest.TestCase):

    def test_untrained(self):
        config = SwagBertConfig(no_cov_mat=False, hidden_size=240)
        print(config)
        model = SwagBertPreTrainedModel(config)
        print(model)
        model = SwagBertModel(config)
        print(model)
        out = model.forward(input_ids=torch.tensor([[3, 14]]))
        print(out)
        model.model.sample()
        out = model.forward(input_ids=torch.tensor([[3, 14]]))
        print(out)


if __name__ == "__main__":
    unittest.main()
