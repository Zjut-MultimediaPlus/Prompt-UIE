import torch
import torch.utils.data as data
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Normalize
import numpy as np

class ValData_map(data.Dataset):
    def __init__(self, val_data_dir, val_filename):
        super().__init__()
        val_list = val_data_dir + val_filename  # ../data/LUSI2/test/input.txt
        input_names = []
        type_id = []
        with open(val_list) as f:
            for line in f:
                filename, img_type = line.strip().split(' ')
                input_names.append(filename)
                type_id.append(int(img_type))

        self.input_names = input_names
        self.gt_names = input_names
        self.val_data_dir = val_data_dir

    def get_images(self, index):
        input_name = self.input_names[index]
        gt_name = self.gt_names[index]

        input_img = Image.open(self.val_data_dir + 'input/' + input_name).convert('RGB') #../data/LUSI2/test/ input/ 0.jpg
        gt_img = Image.open(self.val_data_dir + 'gt/' + gt_name).convert('RGB')

        input_img = input_img.resize((256, 256), Image.ANTIALIAS)
        gt_img = gt_img.resize((256, 256), Image.ANTIALIAS)

        # --- Transform to tensor --- #
        transform_input = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        transform_gt = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        input_im = transform_input(input_img)
        gt = transform_gt(gt_img)

        return input_im, gt, input_name

    def __getitem__(self, index):
        res = self.get_images(index)
        return res

    def __len__(self):
        return len(self.input_names)
