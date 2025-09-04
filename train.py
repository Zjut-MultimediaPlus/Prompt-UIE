# Code used to train backbone
import time
import torch
import argparse
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from UIE import myModel
from utility.train_dataloader import TrainData_map
from utility.val_dataloader import ValData_map
from torchvision.models import vgg16
import sys

from perceptual import LossNetwork
import os
import numpy as np
import random
import pytorch_ssim

parser = argparse.ArgumentParser(description='Hyper-parameters for network')
parser.add_argument('-learning_rate', help='Set the learning rate', default=1e-4, type=float)
parser.add_argument('-crop_size', help='Set the crop_size', default=[256, 256], nargs='+', type=int)
parser.add_argument('-train_batch_size', help='Set the training batch size', default=2, type=int)
parser.add_argument('-epoch_start', help='Starting epoch number of the training', default=0, type=int)
parser.add_argument('-lambda_loss', help='Set the lambda in loss function', default=0.04, type=float)
parser.add_argument('-val_batch_size', help='Set the validation/test batch size', default=1, type=int)
parser.add_argument('-exp_name', help='directory for saving the networks of the experiment', default='weight', type=str)
parser.add_argument('-seed', help='set random seed', default=19, type=int)
parser.add_argument('-num_epochs', help='number of epochs', default=151, type=int)
parser.add_argument('-category', help='output image path', default='110xL1', type=str)
parser.add_argument('-weight_out', help='output weight path', default='bsl', type=str)
parser.add_argument('-train_data_dir', help='train image path', default='/opt/data/private/data/LUSI/train/',
                    type=str)
parser.add_argument('-val_data_dir', help='test image path', default='/opt/data/private/data/LUSI/test/',
                    type=str)
parser.add_argument('-labeled_name', help='The following file should be placed inside the directory ... ',
                    default='input.txt', type=str)
parser.add_argument('-val_filename1',
                    help='### The following files should be placed inside the directory "..."',
                    default='input.txt', type=str)
parser.add_argument('--milestones', nargs='+', type=int, default=[5000000, 9000000])
parser.add_argument('--lr_gamma_condition', type=float, default=0.1)
parser.add_argument('--figure_path', type=str, default='./results/plt/1')

args = parser.parse_args()

learning_rate = args.learning_rate
crop_size = args.crop_size
train_batch_size = args.train_batch_size
epoch_start = args.epoch_start
lambda_loss = args.lambda_loss
val_batch_size = args.val_batch_size
exp_name = args.exp_name
num_epochs = args.num_epochs
category = args.category
weight_out = args.weight_out
train_data_dir = args.train_data_dir
val_data_dir = args.val_data_dir
labeled_name = args.labeled_name
val_filename1 = args.val_filename1

seed = args.seed
if seed is not None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    print('Seed:\t{}'.format(seed))

print('--- Hyper-parameters for training ---')
print(
    'learning_rate: {}\ncrop_size: {}\ntrain_batch_size: {}\nval_batch_size: {}\nweight_out:{}'.format(
        learning_rate,
        crop_size,
        train_batch_size,
        val_batch_size,
        weight_out))

# --- Gpu device --- #
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# --- Define the network --- #
net = myModel()
net = net.to(device)

# --- Build optimizer --- #
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
train_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones,
                                                       gamma=args.lr_gamma)

vgg_model = vgg16(pretrained=True).features[:16]
vgg_model = vgg_model.to(device)
for param in vgg_model.parameters():
    param.requires_grad = False
loss_network = LossNetwork(vgg_model)
loss_network.eval()
loss_ssim = pytorch_ssim.SSIM().to(device)

if os.path.exists('./{}/'.format(exp_name)) == False:
    os.mkdir('./{}/'.format(exp_name))
if os.path.exists('./weight/{}/'.format(weight_out)) == False:
    os.makedirs('./weight/{}/'.format(weight_out))

lbl_train_data_loader = DataLoader(TrainData_map(crop_size, train_data_dir, labeled_name),
                                   batch_size=train_batch_size,
                                   shuffle=True, num_workers=8)
val_data_loader = DataLoader(ValData_map(val_data_dir, val_filename1), batch_size=val_batch_size, shuffle=False,
                             num_workers=8)

max_PSNR = 0


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    learning_rate = args.learning_rate * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rate

def checkpoint(epoch, val_loss):
    model_out_path = "./weight/" + weight_out + "/epoch_{}_ValLoss_{:.6f}.pth".format(epoch, val_loss)
    torch.save(net.state_dict(), model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))

best_val_loss = float("inf")
def eval(val_data_loader, test_model):
    if os.path.exists('./results/{}/'.format(category)) == False:
        os.makedirs('./results/{}/'.format(category))
    test_model.eval()

    val_loss_list = []
    with torch.no_grad():
        for batch_id, val_data in enumerate(val_data_loader):
            input_image, gt, imgid = val_data
            input_image = input_image.to(device)
            gt = gt.to(device)

            im_out = net(input_image)

            # === 计算验证集损失 ===
            smooth_loss = F.smooth_l1_loss(im_out, gt)
            perceptual_loss = loss_network(im_out, gt)
            ssim_loss = - loss_ssim(im_out, gt)
            loss = smooth_loss + lambda_loss * perceptual_loss + ssim_loss
            val_loss_list.append(loss.item())

            # 保存验证结果图像
            save_image(im_out, './results/{}/{}.png'.format(category, imgid[0][:-4]), normalize=True)

    return np.mean(val_loss_list)

Loss_list = []
net.train()
for epoch in range(epoch_start, num_epochs):

    loss_list = []
    start_time = time.time()
    adjust_learning_rate(optimizer, epoch)
    for batch_id, train_data in enumerate(lbl_train_data_loader):

        input_image, gt, imgid = train_data
        input_image = input_image.to(device)
        gt = gt.to(device)

        # start main network training
        optimizer.zero_grad()

        pred_image = net(input_image)

        smooth_loss = F.smooth_l1_loss(pred_image, gt)
        perceptual_loss = loss_network(pred_image, gt)
        ssim_loss = - loss_ssim(pred_image, gt)

        loss = smooth_loss + 0.4 * perceptual_loss + 0.02 * ssim_loss

        loss_list.append(loss.item())
        loss.backward()
        optimizer.step()

        if not (batch_id % 10):
            sys.stdout.write(
                "\r[Epoch %d/%d] , [batch %d],[smooth_loss: %f],[perceptual_loss: %f],[ssim_loss : %f],[total_loss :%f]"
                % (
                    epoch,
                    num_epochs,
                    batch_id,
                    smooth_loss,
                    perceptual_loss * 0.4,
                    ssim_loss * 0.02,
                    loss
                )
            )

    # --- Save the network parameters(per epoch) --- #
    net.eval()
    Loss_list.append(np.mean(loss_list))

    one_epoch_time = time.time() - start_time
    if epoch % 10 == 0 or epoch > 30:
        val_loss = eval(val_data_loader, net)
        if val_loss < best_val_loss:
            checkpoint(epoch, val_loss)
            best_val_loss = val_loss
        print()