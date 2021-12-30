import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam

from tqdm import tqdm
from pprint import pprint
import wandb
import hydra

from models.transformer import TabTransformer
from data_loader.datasets import BlastcharDataset
from utils.utils import Phase

wandb.init(project='Tab-Transformer', entity='harshraj22', mode="disabled")


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
            nn.Dropout(0.2),

            nn.Linear(50, 20),
            nn.ReLU(),
            nn.Dropout(0.2),

            # To Do: Store the num of output classes as a param in the dataset itself
            nn.Linear(20, 2)
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = TabTransformer(blastchar_dataset.num_categories, mlp, embed_dim=EMBED_DIM).to(device)
    optimizer = Adam(model.parameters())

    # To Do: set up a LR schedular


    # Create the train and val dataset
    train_size = int(cfg.params.train_size * len(blastchar_dataset))
    val_size = len(blastchar_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(blastchar_dataset, [train_size, val_size])
    train_dataloader = DataLoader(train_dataset, batch_size=cfg.params.batch_size)
    val_dataloader = DataLoader(val_dataset, batch_size=cfg.params.batch_size)

    # write the training loop
    for epoch in range(cfg.params.num_epochs):
        for phase, dl in [(Phase.Train, train_dataloader), (Phase.Val, val_dataloader)]:
            phase_loss = 0.0
            if phase == Phase.Train:
                model.train()
            else:
                model.eval()

            for batch in tqdm(dl, desc=f'Epoch: {epoch} / {cfg.params.num_epochs}, Phase: {phase}'):
                categorical_vals, continious_vals, ground_truths = batch
                logits = model(categorical_vals.to(device), continious_vals.to(device))
                loss = F.cross_entropy(logits, ground_truths.long().to(device))

                if phase == Phase.Train:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                phase_loss += loss.item()
            tqdm.write(f'{epoch} / {cfg.params.num_epochs}: {phase}: loss {loss:.3f}')
            wandb.log({f'{phase}_loss': phase_loss})


if __name__ == '__main__':
    main()