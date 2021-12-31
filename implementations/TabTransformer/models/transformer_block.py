import torch
import torch.nn as nn
import torch.nn.functional as F
import unittest


class TransformerBlock(nn.Module):
    """Implements the transformer block, used in standard transformers
    architecture. The input is recursively passed through the block.
    https://arxiv.org/pdf/2012.06678v1.pdf
    """
    def __init__(self, num_attention_heads=8, embed_dim=32):
        super(TransformerBlock, self).__init__()
        self.num_attention_heads = num_attention_heads
        self.embed_dim = embed_dim
        self.multi_head_attention = nn.MultiheadAttention(self.embed_dim, self.num_attention_heads, batch_first=True)
        self.feed_forward = nn.Linear(self.embed_dim, self.embed_dim)
        self.activation = nn.SiLU()

    def forward(self, x):
        y, _ = self.multi_head_attention(x, x, x)
        x = F.layer_norm(x + y, (self.embed_dim, ))

        y = self.feed_forward(x)
        y = self.activation(y)
        return F.layer_norm(x + y, (self.embed_dim, ))



class TestTransformerBlock(unittest.TestCase):
    NUM_ATTENTION_HEADS = 8
    EMBED_DIM = 32
    NUM_FEATUES = 9

    @classmethod
    def setUpClass(cls):
        cls.transformer_block = TransformerBlock(cls.NUM_ATTENTION_HEADS, cls.EMBED_DIM)

    def test_out_dims(self):
        """Given an input, the output should have expected shape """
        input_data = torch.rand(1, self.NUM_FEATUES, self.EMBED_DIM)
        with torch.no_grad():
            out = self.transformer_block(input_data)
        self.assertEqual((1, self.NUM_FEATUES, self.EMBED_DIM), tuple(out.shape))


if __name__ == '__main__':
    unittest.main()
    # _ = TransformerBlock()