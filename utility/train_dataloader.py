import torch.utils.data as data
from PIL import Image
from random import randrange
from torchvision.transforms import Compose, ToTensor, Normalize
import re
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

class TrainData_map(data.Dataset):
    def __init__(self, crop_size, train_data_dir, train_filename):  # train_data_dir=../data/LUSI2/train/
        super().__init__()
        train_list = train_data_dir + train_filename
        input_names = []
        type_id = []
        with open(train_list) as f:
            for line in f:
                filename, img_type = line.strip().split(' ')
                input_names.append(filename)
                type_id.append(int(img_type))


        self.input_names = input_names
        self.gt_names = input_names
        self.type_id = type_id
        self.crop_size = crop_size
        self.train_data_dir = train_data_dir

    def get_images(self, index):
        crop_width, crop_height = self.crop_size
        input_name = self.input_names[index]
        gt_name = self.gt_names[index]

        img_id = re.split('/', input_name)[-1][:-4]  # 0

        input_img = Image.open(self.train_data_dir + 'input/' + input_name)  # data/LUSI2/train/input/0.jpg
        #########################################
        try:
            gt_img = Image.open(self.train_data_dir + 'gt/' + gt_name)
        except:
            gt_img = Image.open(self.train_data_dir + 'gt/' + gt_name).convert('RGB')

        width, height = input_img.size

        if width < crop_width and height < crop_height:
            input_img = input_img.resize((crop_width, crop_height), Image.ANTIALIAS)
            gt_img = gt_img.resize((crop_width, crop_height), Image.ANTIALIAS)
        elif width < crop_width:
            input_img = input_img.resize((crop_width, height), Image.ANTIALIAS)
            gt_img = gt_img.resize((crop_width, height), Image.ANTIALIAS)
        elif height < crop_height:
            input_img = input_img.resize((width, crop_height), Image.ANTIALIAS)
            gt_img = gt_img.resize((width, crop_height), Image.ANTIALIAS)

        width, height = input_img.size
        # 生成一个随机数来确定裁剪框在图像中的位置(左上角顶点位置)
        x, y = randrange(0, width - crop_width + 1), randrange(0, height - crop_height + 1)
        input_crop_img = input_img.crop((x, y, x + crop_width, y + crop_height))  # 生成裁剪图像
        gt_crop_img = gt_img.crop((x, y, x + crop_width, y + crop_height))
        # --- Transform to tensor --- #
        transform_input = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        transform_gt = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        input_im = transform_input(input_crop_img)
        gt = transform_gt(gt_crop_img)

        return input_im, gt, img_id

    def __getitem__(self, index):
        res = self.get_images(index)
        return res

    def __len__(self):
        return len(self.input_names)