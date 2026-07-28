import copy
import cv2
import torch.utils.data
from torch.utils.data import Dataset
import os
from torchvision.transforms import transforms
from Landmark2.Config import cfg
import Landmark2.utils as utils
import numpy as np


class StyleDataSet(Dataset):
    def __init__(self, root, is_train, transform=None):
        self.Image_size = cfg.MODEL.IMG_SIZE
        self.is_train = is_train
        self.root = os.path.join(root, 'images')
        self.number_landmarks = 68

        self.Fraction = cfg.W300.FRACTION

        self.Transform = transform

        self.root_landmark = os.path.join(root, 'landmarks')
        self.root_parsing = os.path.join(root, 'parsing')  # 新增 parsing 資料夾路徑
        self.root_latent   = os.path.join(root, 'output_batch')
        if is_train:
            self.annotation_file = os.path.join(root, 'train_list.txt')
        else:
            self.annotation_file = os.path.join(root, 'test_list.txt')

        self.database = self.get_image()

    def get_image(self):
        Data_base = []
        with open(self.annotation_file) as f:
            info_list = f.read().splitlines()
        for temp_info in info_list:
            temp_name = os.path.join(self.root, temp_info)
            parsing_name = os.path.join(self.root_parsing, temp_info.split('.')[0] + ".png")  # 解析圖檔名

            Points = np.load(os.path.join(self.root_landmark, temp_info + ".npy"))
            max_index = np.max(Points, axis=0)
            min_index = np.min(Points, axis=0)
            temp_box = np.array([min_index[0], min_index[1],
                                 max_index[0] - min_index[0],
                                 max_index[1] - min_index[1]])

            Data_base.append({'Img': temp_name,
                              'Parsing': parsing_name,
                              'bbox': temp_box,
                              'points': Points,
                              'latent_name': temp_info.split('.')[0] + ".pt"})
        return Data_base

    def __len__(self):
        return len(self.database)

    def __getitem__(self, idx):
        db_slic = copy.deepcopy(self.database[idx])
        Img_path = db_slic['Img']
        Parsing_path = db_slic['Parsing']
        BBox = db_slic['bbox']
        Points = db_slic['points']
        latent_name = db_slic['latent_name']
        Annotated_Points = Points.copy()

        # 讀取原始 image
        Img = cv2.imread(Img_path)
        Img_shape = Img.shape
        Img = cv2.cvtColor(Img, cv2.COLOR_RGB2BGR)
        if len(Img_shape) < 3:
            Img = cv2.cvtColor(Img, cv2.COLOR_GRAY2RGB)
        else:
            if Img_shape[2] == 4:
                Img = cv2.cvtColor(Img, cv2.COLOR_RGBA2RGB)
            elif Img_shape[2] == 1:
                Img = cv2.cvtColor(Img, cv2.COLOR_GRAY2RGB)
                
        # 讀取 parsing 圖 (解析圖)，以灰階模式讀取
        Parsing = cv2.imread(Parsing_path, cv2.IMREAD_GRAYSCALE)
        if Parsing is None:
            raise FileNotFoundError(f"Parsing file not found: {Parsing_path}")

        # 取得仿射轉換矩陣，使用與 image 相同的 bbox
        trans = utils.get_transforms(BBox, self.Fraction, 0.0, self.Image_size, shift_factor=[0.0, 0.0])

        # 使用同一轉換矩陣對 image 和 parsing 做 warp
        input_img = cv2.warpAffine(Img, trans, (int(self.Image_size), int(self.Image_size)), flags=cv2.INTER_LINEAR)
        input_parsing = cv2.warpAffine(Parsing, trans, (int(self.Image_size), int(self.Image_size)), flags=cv2.INTER_NEAREST)

        # 對 landmarks 做仿射轉換
        for i in range(self.number_landmarks):
            Points[i, 0:2] = utils.affine_transform(Points[i, 0:2], trans)
            
        latent_path = os.path.join(self.root_latent, latent_name)
        if not os.path.exists(latent_path):
            raise FileNotFoundError(f"Latent file not found: {latent_path}")
        latent_dict = torch.load(latent_path, map_location='cpu')
        # 假設 .pt 裡面是 {'latent': Tensor of shape (512,)}
        latent = latent_dict['wplus']

        # 如有 transform，對 image 進行轉換；parsing 轉成 tensor 並保持原標籤
        if self.Transform is not None:
            input_img = self.Transform(input_img)
            input_parsing = torch.tensor(input_parsing, dtype=torch.long)

        meta = {
            'image': input_img,
            'parsing': input_parsing,
            'Annotated_Points': Annotated_Points,
            'Img_path': Img_path,
            'points': Points / (self.Image_size),
            'latent':  latent,  
            'BBox': BBox,
            'trans': trans,
            'Scale': self.Fraction,
            'angle': 0.0,
            'Translation': [0.0, 0.0],
        }
        return meta


def get_cartoon_dataloader(batch_size, opt):
    normalize = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]
    )

    train_dataset = StyleDataSet(root=opt.tgt_data, is_train=True, transform=normalize)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True
    )
    cartoon_dataset = StyleDataSet(root=opt.tgt_data, is_train=False, transform=normalize)
    cartoon_loader = torch.utils.data.DataLoader(
        cartoon_dataset,
        batch_size=1,
        shuffle=False,
        drop_last=True,
    )
    return train_loader, cartoon_loader
