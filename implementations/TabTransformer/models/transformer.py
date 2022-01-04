import torch
import torch.nn as nn
import torch.nn.functional as F
import sys


from models.transformer_block import TransformerBlock
import unittest
import logging

logging.basicConfig(level=logging.DEBUG)


class TabTransformer(nn.Module):
    """Class implementing TabTransformer model.
    https://arxiv.org/pdf/2012.06678v1.pdf
    """
    def __init__(self, col_embed_rows, mlp, num_transformer_block=6, embed_dim=32, num_cont_cols=4):
        """Initializing the TabTransformer Network

        Args:
            col_embed_rows (int): Number of rows in the embedding matrix. Ideally
                if categorical column i has d_i categories, then col_embed_rows = sum over i (d_i + 1)
            mlp (nn.Sequential): A Multi Layered Perceptron taking input of shape:
                num_cat_col * embed_dim + num_cont_col, and producing the output of
                dimention = num of classes in answer
            num_transformer_block (int, optional): number of transformer blocks
                to be used. Defaults to 6.
            embed_dim (int, optional): embedding dimention of each categorical 
                feature. Defaults to 32.
        """
        super(TabTransformer, self).__init__()

        self.num_transformer_block = num_transformer_block
        self.embed_dim = embed_dim

        self.transformer_block = nn.Sequential(
            *[TransformerBlock(embed_dim=self.embed_dim) for _ in range(self.num_transformer_block)]
        )

        self.embed = nn.Embedding(col_embed_rows, self.embed_dim)
        self.mlp = mlp
        self.layer_norm = nn.LayerNorm(num_cont_cols)
        # self.dropout = nn.Dropout(0.2)
        # self.mlp = nn.Sequential(
        #     *[
        #         # nn.Linear(num_cols_for_cat_feats * embed_dim + num_cols_for_continious_feats, 50),
        #         # nn.Linear(50, 25),
        #         # nn.Linear(25, 12),
        #         # nn.Linear(12, 3),
        #         # nn.Linear(3, 1) # for binary classification
        #     ]
        # )

    def forward(self, x_cat, x_cont):
        batch_size, num_cat_cols = x_cat.shape
        
        # x_cat.shape: (num_cat_cols, embed_dim)
        x_cat = self.embed(x_cat)
        assert tuple(x_cat.shape) == (batch_size, num_cat_cols, self.embed_dim), f'Error after converting to embeddings. Expected {(batch_size, num_cat_cols, embed_dim)}, got {x_cat.shape}'

        # x_cat.shape: (num_cat_cols, embed_dim)
        x_cat = self.transformer_block(x_cat)
        assert tuple(x_cat.shape) == (batch_size, num_cat_cols, self.embed_dim), f'Error after transformer block. Expected {(batch_size, num_cat_cols, embed_dim)}, got {x_cat.shape}'

        _, num_cont_cols = x_cont.shape
        assert _ == batch_size, f'Batch size of x_cat({batch_size}), x_cont({_}) does not match'
        # x_cont = F.layer_norm(x_cont, (num_cont_cols,)) # changed to nn.layer_norm to encorporate captum interpretability
        x_cont = self.layer_norm(x_cont)

        x_cat = x_cat.view(batch_size, -1)

        x = torch.cat((x_cat, x_cont), dim=1)
        assert x.shape[1] == num_cat_cols * self.embed_dim + num_cont_cols, f'Some issue while concatenating x_cat({x_cat.shape}), x_cont({x_cont.shape}). Expected {num_cat_cols * self.embed_dim + num_cont_cols}, got {x.shape[1]}'

        return self.mlp(x)


if __name__ == '__main__':
    # assuming each cat_col can have 4 values
    num_cat_col, num_cont_col = 3, 2
    num_cats_per_col = 4
    col_embed_rows = (num_cats_per_col + 1) * num_cat_col
    embed_dim = 32
    batch_size = 3

    mlp = nn.Sequential(
        nn.Linear(num_cat_col * embed_dim + num_cont_col, 50),
        nn.ReLU(),
        nn.Dropout(0.2),

        nn.Linear(50, 20),
        nn.ReLU(),
        nn.Dropout(0.2),

        nn.Linear(20, 1)
    )

    model = TabTransformer(col_embed_rows, mlp)
    x_cont = torch.rand(batch_size, num_cont_col)
    x_cat = torch.randint(0, num_cat_col, size=(batch_size, num_cat_col))
    out = model(x_cat, x_cont)
    print(out.shape)
