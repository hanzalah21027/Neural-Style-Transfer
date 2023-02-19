# Image Setup
img_height = 256
img_width = 256
channels = 3

# Training Setup
max_epochs = 200
lr = 0.0002
b1 = 0.5
b2 = 0.999
batch_size = 4


#pip install wandb
import os
import wandb
import torch
import random
import argparse
import itertools
import numpy as np
from PIL import Image
from torch import nn, optim
from torchvision import transforms
from torch.utils.data import DataLoader
from model.dataset import DATASET
from model.networks import Generator, Discriminator
from model.utility import show_validation_images, Buffer, log_images, Scheduler, init_weights
import warnings
warnings.filterwarnings("ignore")

# rich: for a better display on terminal
from rich.table import Column, Table
from rich import box
from rich.console import Console

# define a rich console logger
console=Console(record=True)


training_logger = Table(Column("Steps", justify="center" ), 
                        Column("Leraning Rate", justify="center"),
                        Column("Discriminator Loss", justify="center"), 
                        Column("Generator Loss", justify="center"), 
                        Column("Adversarial Loss", justify="center"), 
                        Column("Identity Loss", justify="center"), 
                        Column("Cycle Loss", justify="center"), 
                        title="Training Status",pad_edge=False, box=box.ASCII)

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, default = None)
parser.add_argument('--gpu', type=int, default = 0)
parser.add_argument('--model_name', type=str, default = None)
args = parser.parse_args()
# Initial Setting

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(args.gpu)
console.log(f"GPU id: {torch.cuda.current_device()}")


# Dataset
transform = transforms.Compose([
                transforms.Resize(int(img_height*1.12), Image.BICUBIC),
                transforms.RandomCrop((img_height, img_width)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])

inv_transform = transforms.Compose([
    transforms.Normalize((-1, -1, -1), (2, 2, 2))
    ])

dataset = DATASET(path=args.path, transform = transform, train = True, random = True)
train_loader = DataLoader(dataset, batch_size=batch_size, num_workers=0, shuffle=True)

dataset = DATASET(path=args.path, transform = transform, train = False, random = True)
val_loader = DataLoader(dataset, batch_size=5, num_workers=0, shuffle=True)

max_steps = max_epochs *  len(train_loader)
console.log(f"Max Steps: {max_steps}")
decay_start_step = round(max_steps * 0.5)

# Training

G_AB = Generator()
G_AB.to(device)  
D_A = Discriminator() 
D_A.to(device)   
G_BA = Generator()
G_BA.to(device)
D_B = Discriminator() 
D_B.to(device)   
console.log('Loaded model onto GPU.')

G_AB.apply(init_weights)
D_A.apply(init_weights)
G_BA.apply(init_weights)
D_B.apply(init_weights)

criterion_identity = nn.L1Loss()
criterion_cycle = nn.L1Loss()
criterion_GAN = nn.MSELoss()

criterion_GAN.to(device)   
criterion_cycle.to(device)   
criterion_identity.to(device)   

console.log('Loaded Losses onto GPU.')

optimG = optim.Adam(itertools.chain(G_AB.parameters(), G_BA.parameters()), lr = lr, betas = (b1, b2))
optimD_A = optim.Adam(D_A.parameters(), lr = lr, betas = (b1, b2))
optimD_B = optim.Adam(D_B.parameters(), lr = lr, betas = (b1, b2))

schedulerG = optim.lr_scheduler.LambdaLR(optimG, lr_lambda=Scheduler(max_steps, decay_start_step).LambdaLr)
schedulerD_A = optim.lr_scheduler.LambdaLR(optimD_A, lr_lambda=Scheduler(max_steps, decay_start_step).LambdaLr)
schedulerD_B = optim.lr_scheduler.LambdaLR(optimD_B, lr_lambda=Scheduler(max_steps, decay_start_step).LambdaLr)

fake_A_Buffer = Buffer(50)
fake_B_Buffer = Buffer(50)

wandb.init(project = "Neural-Style-Transfer")
wandb.config.update = {
                    "img height": img_height,
                    "img width": img_width,
                    "channels": channels,

                    # Training
                    "Total epochs": max_epochs,
                    "Total steps": max_steps,
                    "Decay start step": decay_start_step,
                    "Starting Learning Rate": lr,
                    "Batch size": batch_size,
                    "beta1": b1,
                    "beta2": b2,
                    'Model name': args.model_name,

          }


console.log(f'*** Initiating Training ***\n')
trainloader_iter = enumerate(train_loader)
for step in range(1, max_steps + 1):

  try:
      _, batch = trainloader_iter.__next__()
  except:
      trainloader_iter = enumerate(train_loader)
      _, batch = trainloader_iter.__next__()

  A, B = batch['A'].to(device), batch['B'].to(device)
  real = torch.ones(A.shape[0], 1, 16, 16).to(device) # Real Vector for Discriminator
  fake = torch.zeros(A.shape[0], 1, 16, 16).to(device) # Fake Vector for Discriminator
    
  # Training Generator 
  D_A.eval()
  D_B.eval()
  G_AB.train()
  G_BA.train()

  optimG.zero_grad() 

  f_B = G_AB(A) # Fake A
  f_A = G_BA(B) # Fake B

  L_GAN = (criterion_GAN(D_B(f_B), real) + criterion_GAN(D_A(f_A), real))
  L_identity = (criterion_identity(G_AB(B), B) * 0.5 + criterion_identity(G_BA(A), A) * 0.5) * 10
  L_cycle = (criterion_cycle(G_AB(f_A), B) * 10 + criterion_cycle(G_BA(f_B), A) * 10)

  Loss_G = L_GAN + L_identity + L_cycle
  Loss_G.backward()
  optimG.step()

  # Traing Discriminator
  D_A.train()
  D_B.train()

  # Training Discriminator A
  optimD_A.zero_grad()
  f_A = fake_A_Buffer.get_and_post(f_A)
  Loss_D_A = (criterion_GAN(D_A(A), real) + criterion_GAN(D_A(f_A.detach()), fake)) * 0.5
  Loss_D_A.backward()
  optimD_A.step()

  # Traing Discriminator B
  optimD_B.zero_grad()
  f_B = fake_B_Buffer.get_and_post(f_B)
  Loss_D_B = (criterion_GAN(D_B(B), real) + criterion_GAN(D_B(f_B.detach()), fake)) * 0.5
  Loss_D_B.backward()
  optimD_B.step()

  Loss_D = (Loss_D_A + Loss_D_B) * 0.5

  learningRate = float(schedulerG.get_last_lr()[0])
  wandb.log({'Learning Rate': learningRate, 'Discriminator Loss': Loss_D.item(), 'Generator Loss': Loss_G.item(), 'Identity Loss':L_identity.item(), 'Adversarial Loss': L_GAN.item(), 'Cycle Loss': L_cycle.item()})
  
  if step % 5 == 0:
    grid = show_validation_images(val_loader, G_AB, G_BA, device, step, inv_transform)
    log_images(grid, step)
    training_logger.add_row(str(step), str(learningRate), str(Loss_D.item()), str(Loss_G.item()), str(L_GAN.item()), str(L_identity.item()), str(L_cycle.item()))
    console.print(training_logger)

  schedulerG.step()
  schedulerD_A.step()
  schedulerD_B.step()

# Saving Model to Wandb
console.log(f'*** Saving Model ***\n')
torch.save(G_AB.state_dict(), os.path.join("Trained Models", args.model_name))
wandb.save(os.path.join("Trained Models", args.model_name))

