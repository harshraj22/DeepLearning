import torch
import torch.nn as nn
import torch.nn.functional as F

import hydra
import logging
import pathlib
from pprint import pprint

from models.transformer import TabTransformer
from data_loader.datasets import BlastcharDataset

from captum.attr import IntegratedGradients, LayerIntegratedGradients
from captum.attr import (
    IntegratedGradients,
    LayerIntegratedGradients,
    configure_interpretable_embedding_layer,
    remove_interpretable_embedding_layer,
    visualization
)
from captum.attr._utils.input_layer_wrapper import ModelInputWrapper

logging.basicConfig(level=logging.NOTSET)
# logging.getLogger('matplotlib').setLevel(logging.WARNING)
# logging.getLogger('matplotlib.pyplot').disabled = True


@hydra.main(config_path="conf", config_name="config")
def main(cfg):
    # Initialize the dataset
    blastchar_dataset = BlastcharDataset(cfg.dataset.path)
    NUM_CATEGORICAL_COLS = blastchar_dataset.num_categorical_cols
    NUM_CONTINIOUS_COLS = blastchar_dataset.num_continious_cols
    EMBED_DIM = 32

    # initialize the model with its arguments
    mlp = nn.Sequential(
            nn.Linear(NUM_CATEGORICAL_COLS * EMBED_DIM + NUM_CONTINIOUS_COLS, 50),
            nn.ReLU(),
            nn.BatchNorm1d(50),
            nn.Dropout(cfg.params.dropout),

            nn.Linear(50, 20),
            nn.ReLU(),
            nn.BatchNorm1d(20),
            nn.Dropout(cfg.params.dropout),

            nn.Linear(20, blastchar_dataset.num_classes)
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = TabTransformer(blastchar_dataset.num_categories, mlp, embed_dim=EMBED_DIM, num_cont_cols=NUM_CONTINIOUS_COLS)
    model.load_state_dict(torch.load(cfg.params.weights), strict=False)
    model = model.to(device)
    model.eval()
    
    model = ModelInputWrapper(model)

    cat, cont, _ = blastchar_dataset[0]
    cat, cont = cat.unsqueeze(0).long(), cont.unsqueeze(0).float()
    cat = torch.cat((cat, cat), dim=0)
    cont = torch.cat((cont, cont), dim=0)
    input = (cat, cont)

    outs = model(*input)
    preds = outs.argmax(-1)


    attr = LayerIntegratedGradients(model, [model.module.embed, model.module.layer_norm])

    attributions, _ = attr.attribute(inputs=(cat, cont), baselines=(torch.zeros_like(cat, dtype=torch.long), torch.zeros_like(cont, dtype=torch.float32)), target=preds.detach(), n_steps=30, return_convergence_delta=True)

    print(f'attributions: {attributions[0].shape, attributions[1].shape}')
    pprint(torch.cat((attributions[0].sum(dim=2), attributions[1]), dim=1))

main()