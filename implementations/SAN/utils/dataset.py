from torch.utils.data import Dataset, DataLoader
from tqdm.autonotebook import tqdm
from PIL import Image
import numpy as np

"""
Write your own dataset here.
The __getitem__ method must return a dict with the following key, value pairs:
    'ques': tensor of ints, representing the index of words in the vocab
    'ans': tensor of int, representing the index of word answer
    'img': tensor representing the image
    
Get Images for the dataset:
    ! wget http://datasets.d2.mpi-inf.mpg.de/mateusz14visual-turing/nyu_depth_images.tar
Get Train question & ans:
    ! wget https://raw.githubusercontent.com/jayantk/lsp/master/data/daquar/reduced/qa.37.raw.train.txt
DAQAR dataset:
    https://github.com/jayantk/lsp/tree/master/data/daquar
"""


class VQADataset(Dataset):
  def __init__(self, ques_file, image_dir, tokenizer):
    super(Dataset, self).__init__()
    self.ques_file = ques_file
    self.img_dir = image_dir
    self.tokenizer = tokenizer

    self.data = []
    self.load_data()

  def load_data(self):
    with open(self.ques_file, 'r') as f:
      data = f.readlines()

    for index, line in tqdm(enumerate(data[::2]), desc='Iterating over questions'):
      img = line.replace('?', '').strip(' ').split()[-1] + '.png'
      dct = {
          'ques': self.tokenizer.encode(line),
          'ans': self.tokenizer.convert_tokens_to_ids([data[2*index+1]]),
          'img_file_name': img
      }

      if len(dct['ans']) == 1:
        self.data.append(dct)

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    dct = self.data[idx]
    img = Image.open(self.img_dir + dct['img_file_name'])
    dct['img'] = torch.from_numpy(np.array(img))
    return dct
