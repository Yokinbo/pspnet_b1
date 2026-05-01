import os

import cv2
import numpy as np
import torch
import rasterio
from PIL import Image
from torch.utils.data.dataset import Dataset

from utils.utils import cvtColor, preprocess_input


class PSPnetDataset(Dataset):
    def __init__(self, annotation_lines, input_shape, num_classes, train, dataset_path, image_ext=".tif", selected_bands=None):
        super(PSPnetDataset, self).__init__()
        self.annotation_lines   = annotation_lines
        self.length             = len(annotation_lines)
        self.input_shape        = input_shape
        self.num_classes        = num_classes
        self.train              = train
        self.dataset_path       = dataset_path
        # 第一步修改：
        # 1. 增加 image_ext，允许数据集从 .tif 而不是 .jpg 读取影像。
        # 2. 增加 selected_bands，用于控制训练时实际取哪些波段。
        #    例如 6 波段影像里只训练 4 波段时，可传 [1, 2, 3, 4]。
        self.image_ext          = image_ext
        self.selected_bands     = selected_bands

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        annotation_line = self.annotation_lines[index]
        name            = annotation_line.split()[0]

        #-------------------------------#
        #   从文件中读取图像
        #-------------------------------#
        # 第一步修改：
        # 原来这里固定按 .jpg + PIL 读取，现在改成：
        # - 若是 .tif，则用 rasterio 读取多波段影像
        # - 若不是 .tif，则保留原始 jpg/png 的读取方式，兼容旧流程
        image_path  = os.path.join(os.path.join(self.dataset_path, "VOC2007/JPEGImages"), name + self.image_ext)
        if self.image_ext.lower() in [".tif", ".tiff"]:
            jpg     = self.read_tif(image_path)
        else:
            jpg     = Image.open(image_path)
        png         = Image.open(os.path.join(os.path.join(self.dataset_path, "VOC2007/SegmentationClass"), name + ".png"))
        #-------------------------------#
        #   数据增强
        #-------------------------------#
        jpg, png    = self.get_random_data(jpg, png, self.input_shape, random = self.train)

        # 第一步修改：
        # 原来默认把 PIL RGB 图转成 HWC，再转 CHW。
        # 现在 tif 读取后可能本身就是 HWC 多波段数组，因此这里统一兼容：
        # - 如果是 PIL.Image，先转 numpy
        # - 如果已经是 numpy，则直接处理
        if isinstance(jpg, Image.Image):
            jpg = np.array(jpg, np.float64)
        else:
            jpg = np.array(jpg, np.float64)

        if len(np.shape(jpg)) == 2:
            jpg = np.expand_dims(jpg, -1)

        jpg         = np.transpose(preprocess_input(jpg), [2, 0, 1])
        png         = np.array(png)
        png[png >= self.num_classes] = self.num_classes
        #-------------------------------------------------------#
        #   转化成one_hot的形式
        #   在这里需要+1是因为voc数据集有些标签具有白边部分
        #   我们需要将白边部分进行忽略，+1的目的是方便忽略。
        #-------------------------------------------------------#
        seg_labels  = np.eye(self.num_classes + 1)[png.reshape([-1])]
        seg_labels  = seg_labels.reshape((int(self.input_shape[0]), int(self.input_shape[1]), self.num_classes+1))

        return jpg, png, seg_labels

    def read_tif(self, image_path):
        # 第一步修改：
        # 用 rasterio 读取多波段 tif，并且在这里支持按 selected_bands 选波段。
        # 注意：selected_bands 使用 1-based 编号，更符合遥感波段编号习惯。
        with rasterio.open(image_path) as src:
            if self.selected_bands is None:
                band_indexes = list(range(1, src.count + 1))
            else:
                band_indexes = self.selected_bands

            image = src.read(indexes=band_indexes)  # (C, H, W)
            image = np.transpose(image, (1, 2, 0))  # -> (H, W, C)
        return image

    def rand(self, a=0, b=1):
        return np.random.rand() * (b - a) + a

    def get_random_data(self, image, label, input_shape, jitter=.3, hue=.1, sat=0.7, val=0.3, random=True):
        # 第一步修改：
        # 多波段 tif 不能再强制转 RGB，因此只有 PIL 图像才走原来的 cvtColor。
        if isinstance(image, Image.Image):
            image = cvtColor(image)
        label   = Image.fromarray(np.array(label))
        #------------------------------#
        #   获得图像的高宽与目标高宽
        #------------------------------#
        if isinstance(image, Image.Image):
            iw, ih = image.size
        else:
            ih, iw = image.shape[:2]
        h, w    = input_shape

        if not random:
            if isinstance(image, Image.Image):
                iw, ih = image.size
            else:
                ih, iw = image.shape[:2]
            scale   = min(w/iw, h/ih)
            nw      = int(iw*scale)
            nh      = int(ih*scale)

            if isinstance(image, Image.Image):
                image = image.resize((nw,nh), Image.BICUBIC)
                new_image = Image.new('RGB', [w, h], (128,128,128))
                new_image.paste(image, ((w-nw)//2, (h-nh)//2))
            else:
                image = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
                channels = image.shape[2]
                new_image = np.full((h, w, channels), 128, dtype=image.dtype)
                new_image[(h-nh)//2:(h-nh)//2 + nh, (w-nw)//2:(w-nw)//2 + nw, :] = image

            label       = label.resize((nw,nh), Image.NEAREST)
            new_label   = Image.new('L', [w, h], (0))
            new_label.paste(label, ((w-nw)//2, (h-nh)//2))
            return new_image, new_label

        #------------------------------------------#
        #   对图像进行缩放并且进行长和宽的扭曲
        #------------------------------------------#
        new_ar = iw/ih * self.rand(1-jitter,1+jitter) / self.rand(1-jitter,1+jitter)
        scale = self.rand(0.25, 2)
        if new_ar < 1:
            nh = int(scale*h)
            nw = int(nh*new_ar)
        else:
            nw = int(scale*w)
            nh = int(nw/new_ar)
        if isinstance(image, Image.Image):
            image = image.resize((nw,nh), Image.BICUBIC)
        else:
            image = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
        label = label.resize((nw,nh), Image.NEAREST)
        
        #------------------------------------------#
        #   翻转图像
        #------------------------------------------#
        flip = self.rand()<.5
        if flip: 
            if isinstance(image, Image.Image):
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
            else:
                image = np.ascontiguousarray(image[:, ::-1, :])
            label = label.transpose(Image.FLIP_LEFT_RIGHT)
        
        #------------------------------------------#
        #   将图像多余的部分加上灰条
        #------------------------------------------#
        dx = int(self.rand(0, w-nw))
        dy = int(self.rand(0, h-nh))
        if isinstance(image, Image.Image):
            new_image = Image.new('RGB', (w,h), (128,128,128))
            new_image.paste(image, (dx, dy))
            image = new_image
        else:
            channels = image.shape[2]
            new_image = np.full((h, w, channels), 128, dtype=image.dtype)
            # 第一步补充修复：
            # PIL 的 paste 在 dx/dy 为负、或者 nw/nh 大于目标尺寸时，会自动裁掉越界部分。
            # numpy 直接赋值不会自动裁剪，所以这里手动做一次“求交集”。
            x1 = max(dx, 0)
            y1 = max(dy, 0)
            x2 = min(dx + nw, w)
            y2 = min(dy + nh, h)

            src_x1 = max(-dx, 0)
            src_y1 = max(-dy, 0)
            src_x2 = src_x1 + (x2 - x1)
            src_y2 = src_y1 + (y2 - y1)

            if x2 > x1 and y2 > y1:
                new_image[y1:y2, x1:x2, :] = image[src_y1:src_y2, src_x1:src_x2, :]
            image = new_image
        new_label = Image.new('L', (w,h), (0))
        new_label.paste(label, (dx, dy))
        label = new_label

        image_data      = np.array(image)

        #---------------------------------#
        #   对图像进行色域变换
        #   计算色域变换的参数
        #---------------------------------#
        # 第一步修改：
        # 对多波段影像，先不做 HSV 色彩增强；这部分后面我们会专门再优化。
        # 这里只保留几何增强，确保先把多光谱读取链路打通。
        if image_data.ndim == 3 and image_data.shape[2] == 3:
            r               = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1
            hue, sat, val   = cv2.split(cv2.cvtColor(image_data.astype(np.uint8), cv2.COLOR_RGB2HSV))
            dtype           = hue.dtype
            x       = np.arange(0, 256, dtype=r.dtype)
            lut_hue = ((x * r[0]) % 180).astype(dtype)
            lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
            lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

            image_data = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
            image_data = cv2.cvtColor(image_data, cv2.COLOR_HSV2RGB)
        
        return image_data, label

# DataLoader中collate_fn使用
def pspnet_dataset_collate(batch):
    images      = []
    pngs        = []
    seg_labels  = []
    for img, png, labels in batch:
        images.append(img)
        pngs.append(png)
        seg_labels.append(labels)
    images      = torch.from_numpy(np.array(images)).type(torch.FloatTensor)
    pngs        = torch.from_numpy(np.array(pngs)).long()
    seg_labels  = torch.from_numpy(np.array(seg_labels)).type(torch.FloatTensor)
    return images, pngs, seg_labels
