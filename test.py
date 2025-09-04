import time
import torch
import argparse
import torch.nn as nn
from torch.utils.data import DataLoader
from utility.val_dataloader import ValData_map
import os
import numpy as np
import random
import sys

from UIE import myModel
from torchvision.utils import save_image

# --- Parse hyper-parameters  --- #
parser = argparse.ArgumentParser(description='Hyper-parameters for network')
parser.add_argument('-val_batch_size', help='Set the validation/test batch size', default=1, type=int)
parser.add_argument('-exp_name', help='directory for saving the networks of the experiment', type=str)
parser.add_argument('-seed', help='set random seed', default=19, type=int)
parser.add_argument('-val_data_dir', help='test image path', default='/opt/data/private/data/LSUI/test/',
                    type=str)
parser.add_argument('-val_filename',
                    help='### The following files should be placed inside the directory "./data/test/"',
                    default='input.txt', type=str)
parser.add_argument('-category', help='output image path', default='test', type=str)
parser.add_argument('-pretrain_dir', help='pretrain model path',
                    default='/opt/data/private/Pro02/weight/...pth', type=str)

args = parser.parse_args()

val_batch_size = args.val_batch_size
exp_name = args.exp_name
val_data_dir = args.val_data_dir
val_filename = args.val_filename
category = args.category
pretrain_dir = args.pretrain_dir
# set seed
seed = args.seed
if seed is not None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    print('Seed:\t{}'.format(seed))

# --- Gpu device --- #
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# --- Validation data loader --- #
val_data_loader = DataLoader(ValData_map(val_data_dir, val_filename), batch_size=val_batch_size, shuffle=False)

# --- Define the network --- #
net = myModel()
net = net.to(device)
print('----------------------')

# --- Load the network weight --- #
net.load_state_dict(torch.load(pretrain_dir))
num_parameters = sum(p.numel() for p in net.parameters())
num_parameters = num_parameters / (1024 ** 2)
print(f"Number of parameters in the model: {num_parameters}M")
print('--- Testing starts! ---')

net.eval()

if os.path.exists('./results/{}/'.format(category)) == False:
    os.makedirs('./results/{}/'.format(category))
output_images_path = './results/{}/'.format(category)

for batch_id, train_data in enumerate(val_data_loader):
    with torch.no_grad():
        input_image, gt, imgid = train_data
        input_image = input_image.cuda()
        gt = gt.cuda()
        im_out = net(input_image)

        save_image(im_out, './results/{}/{}.png'.format(category, imgid[0][:-4]), normalize=True)

print('--- Testing Done! ---')
