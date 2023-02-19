import torch
import random
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import wandb
import random
from torch import nn

def show_validation_images(val_loader, G_AB, G_BA, device, step, inv_transform, server = True): 

  imgs = next(iter(val_loader))
  A, B = imgs['A'].to(device), imgs['B'].to(device)

  G_AB.eval()
  G_BA.eval()
  f_B = G_AB(A).detach()
  f_A = G_BA(B).detach()
  A = inv_transform(A)
  B = inv_transform(B)
  f_A = inv_transform(f_A)
  f_B = inv_transform(f_B)
  
  A = make_grid(A, nrow=5, normalize=True)
  f_A = make_grid(f_A, nrow=5, normalize=True)
  B = make_grid(B, nrow=5, normalize=True)
  f_B = make_grid(f_B, nrow=5, normalize=True)

  grid_a = torch.cat((A, f_B), 1)
  grid_b = torch.cat((B, f_A), 1)

  if not server:

    plt.figure(figsize = (12, 16))
    plt.imshow(grid_a.cpu().permute(1, 2, 0))
    plt.title('Image --> Art Style')
    plt.axis('off')
    plt.show()
    
    plt.figure(figsize = (12, 16))
    plt.imshow(grid_b.cpu().permute(1, 2, 0))
    plt.title('Art Style --> Image')
    plt.axis('off')
    plt.show()

  return {'A' : A, 'f_B' : f_B, 'B' : B, 'f_A' : f_A}



class Buffer():
  def __init__(self, size):
    self.buffer_size = size
    if self.buffer_size > 0:
      self.buffer = []
      self.curr_size = 0 


  def get_and_post(self, imgs):
    if self.buffer_size == 0:
      return imgs

    return_img = []
    for img in imgs:
      image = img.unsqueeze(0)

      if self.curr_size < self.buffer_size:
        self.curr_size += 1
        self.buffer += [image]
        return_img += [image]

      else:
        p = random.uniform(0, 1)
        if p > 0.5:
          length = len(self.buffer)
          idx = random.randint(0, length - 1)
          temp = self.buffer[idx]
          self.buffer[idx] = image
          return_img += [temp]

        else:
          return_img += [image]

    return_img = torch.cat(return_img, 0)  
    return return_img

def log_images(grid, step):
  wandb.log({f'{"validation"}/{"Original Image"}':  wandb.Image(grid['A'].cpu().permute(1, 2, 0).numpy())}, step = step)
  wandb.log({f'{"validation"}/{"Generated Painting"}':  wandb.Image(grid['f_B'].cpu().permute(1, 2, 0).numpy())}, step = step)
  wandb.log({f'{"validation"}/{"Original Painting"}':  wandb.Image(grid['B'].cpu().permute(1, 2, 0).numpy())}, step = step)
  wandb.log({f'{"validation"}/{"Generated Image"}':  wandb.Image(grid['f_A'].cpu().permute(1, 2, 0).numpy())}, step = step)


def init_weights(module):

  if isinstance(module, nn.Conv2d):
    nn.init.normal_(module.weight.data, 0.0, 0.02)
    
    if hasattr(module, 'bias') and module.bias is not None:
      nn.init.constant_(module.bias.data, 0.0)
  
  elif isinstance(module, nn.InstanceNorm2d):
      
    if hasattr(module, 'weight') and module.weight is not None:
      nn.init.normal_(module.weight.data, 1.0, 0.02)
      
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.normal_(module.bias.data, 0)


class Scheduler():
  def __init__(self, max_epochs, decay_start_epoch):
    self.max_epochs = max_epochs
    self.decay_start = decay_start_epoch
  
  def LambdaLr(self, epoch):
    return 1 - (max(0, epoch - self.decay_start) / (self.max_epochs - self.decay_start)) 