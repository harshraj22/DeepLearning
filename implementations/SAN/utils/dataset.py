from torch.utils.data import Dataset, DataLoader
from tqdm.autonotebook import tqdm
import torch
from PIL import Image
import numpy as np
from torch.nn.utils.rnn import pad_sequence

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
  def __init__(self, ques_file, image_dir, tokenizer, max_len=30):
    super(Dataset, self).__init__()
    self.ques_file = ques_file
    self.img_dir = image_dir
    self.tokenizer = tokenizer
    self.max_sentence_len = max_len

    self.data = []
    self.load_data()

  def load_data(self):
    with open(self.ques_file, 'r') as f:
      data = f.readlines()

    for index, line in tqdm(enumerate(data[::2]), desc='Iterating over questions'):
      img = line.replace('?', '').strip(' ').split()[-1] + '.png'
      ques = [x for x in self.tokenizer.encode(line)]
      ques = [torch.tensor(min(x, vocab_size-1)) for x in ques]

      ans = self.tokenizer.convert_tokens_to_ids([data[2*index+1].strip()])
      ans = [torch.tensor(min(vocab_size-1, ans[0]))]
      dct = {
          'ques': ques,
          'ques_str': line,
          'ans_str': data[2*index+1],
          'ans': ans,
          'img_file_name': img
      }

      if len(dct['ans']) == 1:
        self.data.append(dct)

  def __len__(self):
    return len(self.data) #// 10

  def __getitem__(self, idx):
    dct = self.data[idx]
    # Crop to given size, as input to vgg is fixed.
    img = Image.open(self.img_dir + dct['img_file_name']).crop((0, 0, 448, 448))
    # Normalize image pixels
    img = np.array(img, dtype=np.uint8) / 255
    # (H, W, C) -> (C, H, W)
    img = np.moveaxis(img, -1, 0)
    dct['img'] = torch.from_numpy(img)
    return dct


def pad_collate(batch):
    """Padds the sentences to given length ensuring all sentences are of same length.
    
    Args:
        batch (Dict): Dictionary containing the data. See load_data method from VQADataset class
    
    Returns:
        batch (dict): Pads the sentences and returns the dict
    """
    ques = [torch.tensor(x['ques']) for x in batch]
    ques = pad_sequence(ques, batch_first=True)
    
    for idx, x in enumerate(ques):
        batch[idx]['ques'] = x
    return batch



if __name__ == '__main__':
  dl = DataLoader(VQADataset(ques_file='/content/qa.37.raw.train.txt', image_dir='/content/nyu_depth_images/', tokenizer=tokenizer), batch_size=2, collate_fn=pad_collate)