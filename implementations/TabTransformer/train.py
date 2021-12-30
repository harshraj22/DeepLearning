import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam

from tqdm import tqdm
from pprint import pprint

from models.transformer import TabTransformer
from data_loader.datasets import BlastcharDataset



def main():
	blastchar_dataset = BlastcharDataset('/home/prabhu/spring2021/tabTransformer/data/Telco-Customer-Churn.csv')
	NUM_CATEGORICAL_COLS = blastchar_dataset.num_categorical_cols
	NUM_CONTINIOUS_COLS = blastchar_dataset.num_continious_cols
	EMBED_LEN = blastchar_dataset.num_categories
	EMBED_DIM = 32

	BATCH_SIZE = 32

	mlp = nn.Sequential(
			nn.Linear(NUM_CATEGORICAL_COLS * EMBED_DIM + NUM_CONTINIOUS_COLS, 50),
			nn.ReLU(),
			nn.Dropout(0.2),

			nn.Linear(50, 20),
			nn.ReLU(),
			nn.Dropout(0.2),

			nn.Linear(20, 2)
	)

	device = 'cuda' if torch.cuda.is_available() else 'cpu'
	model = TabTransformer(blastchar_dataset.num_categories, mlp, embed_dim=EMBED_DIM).to(device)
	optimizer = Adam(model.parameters())

	# To Do: set up a LR schedular


	# Create the train and val dataset
	train_size = int(0.8 * len(blastchar_dataset))
	val_size = len(blastchar_dataset) - train_size
	train_dataset, val_dataset = torch.utils.data.random_split(blastchar_dataset, [train_size, val_size])
	train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
	val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

	# write the training loop
	NUM_EPOCHS = 20
	for epoch in range(NUM_EPOCHS):
		for phase, dl in [('train', train_dataloader), ('val', val_dataloader)]:
			phase_loss = 0.0
			if phase == 'train':
				model.train()
			else:
				model.eval()

			for batch in tqdm(dl, desc=f'Epoch: {epoch} / {NUM_EPOCHS}, Phase: {phase}'):
				categorical_vals, continious_vals, ground_truths = batch
				logits = model(categorical_vals.to(device), continious_vals.to(device))
				loss = F.cross_entropy(logits, ground_truths.long().to(device))

				if phase == 'train':
					optimizer.zero_grad()
					loss.backward()
					optimizer.step()
				phase_loss += loss.item()
			tqdm.write(f'{epoch} / {NUM_EPOCHS}: {phase}: loss {loss:.3f}')




			# print(nn.Softmax(1)(logits), torch.argmax(logits, -1), ground_truths, loss.item())
			# exit(0)





	# cat, cont = blastchar_dataset[0]
	# out = model(torch.stack([cat, cat], dim=0), torch.stack([cont, cont]))
	# print(out.shape)

main()