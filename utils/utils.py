import random

import numpy as np
import torch
from PIL import Image

try:
    # 第七步修改：
    # 把多光谱标准化配置集中放到 multispectral_config.py 中统一管理。
    # 这样训练 / 验证 / patch 推理 / 大图推理都会共用同一套预处理逻辑。
    from multispectral_config import normalization_config
except Exception:
    # 若当前不是多光谱任务，或者用户暂时没有这个配置文件，
    # 就退回到“只做基础缩放”的保底行为，避免影响旧流程。
    normalization_config = {
        "reflectance_scale": 10000.0,
        "enable_clip": False,
        "clip_min": None,
        "clip_max": None,
        "enable_mean_std": False,
        "mean": None,
        "std": None,
    }

#---------------------------------------------------------#
#   将图像转换成RGB图像，防止灰度图在预测时报错。
#   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
#---------------------------------------------------------#
def cvtColor(image):
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image 
    else:
        image = image.convert('RGB')
        return image 

#---------------------------------------------------#
#   对输入图像进行resize
#---------------------------------------------------#
def resize_image(image, size):
    iw, ih  = image.size
    w, h    = size

    scale   = min(w/iw, h/ih)
    nw      = int(iw*scale)
    nh      = int(ih*scale)

    image   = image.resize((nw,nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128,128,128))
    new_image.paste(image, ((w-nw)//2, (h-nh)//2))

    return new_image, nw, nh
    
#---------------------------------------------------#
#   获得学习率
#---------------------------------------------------#
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

#---------------------------------------------------#
#   设置种子
#---------------------------------------------------#
def seed_everything(seed=11):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

#---------------------------------------------------#
#   设置Dataloader的种子
#---------------------------------------------------#
def worker_init_fn(worker_id, rank, seed):
    worker_seed = rank + seed
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)

def _reshape_band_vector(values, channels, name):
    # 第七步新增：
    # 将 [C] 形式的波段参数整理成可广播的 [1,1,C]，
    # 方便对 HWC 影像逐波段做 clip / mean/std。
    if values is None:
        return None

    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim != 1 or arr.shape[0] != channels:
        raise ValueError(
            f"{name} 的长度应与输入通道数一致，当前通道数={channels}，"
            f"但拿到的是 shape={arr.shape}。"
        )
    return arr.reshape((1, 1, channels))

def preprocess_input(image):
    # 第七步修改：
    # 原来这里只做：
    # 1. 多光谱 tif -> /10000
    # 2. 普通 RGB  -> /255
    #
    # 现在扩展成统一的可配置链路：
    # A. 若是 Sentinel-2 这类 uint16 多光谱输入：
    #    - 先 /10000 做物理缩放
    #    - 再按配置决定是否做 clip
    #    - 再按配置决定是否做 mean/std
    # B. 若是普通 jpg/png：
    #    - 仍然保持原来的 /255 逻辑，避免误伤旧项目流程
    #
    # 这样做的目的，是让 3/4/6 波段实验都能共用同一套预处理代码，
    # 但每种模式又能使用各自独立的标准化参数。
    image = image.astype(np.float32, copy=False)

    if image.size == 0:
        return image

    max_value = float(np.max(image))

    if max_value <= 255:
        # 普通 8bit 图像继续保持老逻辑，只做 /255。
        image /= 255.0
        return image

    # 从这里开始，默认认为当前输入是 Sentinel-2 这类多光谱 uint16 数据。
    # 第一步：基础物理缩放。你当前 GEE 导出的 SR 数据通常就是这个尺度。
    reflectance_scale = float(normalization_config.get("reflectance_scale", 10000.0))
    if reflectance_scale <= 0:
        raise ValueError(f"reflectance_scale 必须大于 0，当前值={reflectance_scale}")
    image /= reflectance_scale

    channels = image.shape[2] if image.ndim == 3 else 1

    # 第二步：可选的逐波段裁剪。
    # 论文实验里常用于压制异常值；默认先关闭，等你统计好参数再开启。
    if normalization_config.get("enable_clip", False):
        clip_min = _reshape_band_vector(normalization_config.get("clip_min"), channels, "clip_min")
        clip_max = _reshape_band_vector(normalization_config.get("clip_max"), channels, "clip_max")
        if np.any(clip_max <= clip_min):
            raise ValueError("clip_max 中的每个值都必须大于 clip_min 对应位置的值。")
        image = np.clip(image, clip_min, clip_max)

    # 第三步：可选的逐波段 mean/std 标准化。
    # 这是后续做 3/4/6 波段公平对比实验时最推荐启用的一步。
    if normalization_config.get("enable_mean_std", False):
        mean = _reshape_band_vector(normalization_config.get("mean"), channels, "mean")
        std = _reshape_band_vector(normalization_config.get("std"), channels, "std")
        if np.any(std <= 0):
            raise ValueError("std 中的每个值都必须大于 0。")
        image = (image - mean) / std

    return image

def show_config(**kwargs):
    print('Configurations:')
    print('-' * 70)
    print('|%25s | %40s|' % ('keys', 'values'))
    print('-' * 70)
    for key, value in kwargs.items():
        print('|%25s | %40s|' % (str(key), str(value)))
    print('-' * 70)

def download_weights(backbone, model_dir="./model_data"):
    import os
    from torch.hub import load_state_dict_from_url
    
    download_urls = {
        'mobilenet' : 'https://github.com/bubbliiiing/pspnet-pytorch/releases/download/v1.0/mobilenet_v2.pth.tar',
        'resnet50'  : 'https://github.com/bubbliiiing/pspnet-pytorch/releases/download/v1.0/resnet50s-a75c83cf.pth'
    }
    url = download_urls[backbone]
    
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    load_state_dict_from_url(url, model_dir)
