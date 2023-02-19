import os
import glob
import random
from PIL import Image
from torch import nn
from torch.utils.data import Dataset


class DATASET(Dataset):
  def __init__(self, path, train = True, transform = None, random = True):
    self.transform = transform 
    self.random = random
    if train:           
      self.pathA = sorted(glob.glob(os.path.join(path, "trainB", '*.*')))
      self.pathB = sorted(glob.glob(os.path.join(path, "trainA", '*.*')))
    else:
      self.pathA = sorted(glob.glob(os.path.join(path, "testB", '*.*'))[:50])
      self.pathB = sorted(glob.glob(os.path.join(path, "testA", '*.*'))[:50])

  def __getitem__(self, idx):
    A = Image.open(self.pathA[idx % len(self.pathA)]).convert("RGB")
    if not self.random:
      B = Image.open(self.pathB[idx % len(self.pathB)]).convert("RGB")
    else:
      B = Image.open(self.pathB[random.randint(0, len(self.pathB) - 1)]).convert("RGB")

    if self.transform is not None:
      A = self.transform(A)
      B = self.transform(B)

    return {"A" : A, "B" : B}
    
  def __len__(self):
    return len(min(self.pathA, self.pathB))

