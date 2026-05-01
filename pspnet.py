import colorsys
import copy
import time

import cv2
import numpy as np
import rasterio
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

from multispectral_config import image_ext, in_channels, selected_bands, trained_model_path, vis_bands
from nets.pspnet import PSPNet as pspnet
from utils.utils import cvtColor, preprocess_input, resize_image, show_config


#-----------------------------------------------------------------------------------#
#   使用自己训练好的模型预测需要修改3个参数
#   model_path、backbone和num_classes都需要修改！
#   如果出现shape不匹配，一定要注意训练时的model_path、backbone和num_classes的修改
#-----------------------------------------------------------------------------------#
class PSPNet(object):
    _defaults = {
        #-------------------------------------------------------------------#
        #   model_path指向logs文件夹下的权值文件
        #   训练好后logs文件夹下存在多个权值文件，选择验证集损失较低的即可。
        #   验证集损失较低不代表miou较高，仅代表该权值在验证集上泛化性能较好。
        #-------------------------------------------------------------------#
        # 第十步修改：
        # 推理阶段默认使用的“训练好权重路径”统一从 multispectral_config.py 读取。
        # 这不影响 train.py 里的预训练权重/断点训练配置。
        "model_path"        : trained_model_path,
        #----------------------------------------#
        #   所需要区分的类的个数+1
        #----------------------------------------#
        "num_classes"       : 2,
        #----------------------------------------#
        #   所使用的的主干网络：mobilenet、resnet50
        #----------------------------------------#
        "backbone"          : "mobilenet",
        #----------------------------------------#
        #   第六步修改：推理阶段也支持多光谱输入
        #   image_ext       推理时默认影像后缀
        #   selected_bands  推理时实际使用的波段（1-based）
        #   in_channels     自动由 selected_bands 长度得到
        #   vis_bands       可视化叠加时使用的真彩色波段顺序
        #----------------------------------------#
        # 第九步修改：
        # 这里也统一从 multispectral_config.py 读取多光谱配置，
        # 避免 train / predict / get_miou / pspnet.py 四处不一致。
        "image_ext"         : image_ext,
        "selected_bands"    : selected_bands,
        "vis_bands"         : vis_bands,
        "in_channels"       : in_channels,
        #----------------------------------------#
        #   输入图片的大小
        #----------------------------------------#
        "input_shape"       : [512, 512],
        #----------------------------------------#
        #   下采样的倍数，一般可选的为8和16
        #   与训练时设置的一样即可
        #----------------------------------------#
        "downsample_factor" : 16,
        #-------------------------------------------------#
        #   mix_type参数用于控制检测结果的可视化方式
        #
        #   mix_type = 0的时候代表原图与生成的图进行混合
        #   mix_type = 1的时候代表仅保留生成的图
        #   mix_type = 2的时候代表仅扣去背景，仅保留原图中的目标
        #-------------------------------------------------#
        "mix_type"          : 0,
        #--------------------------------#
        #   是否使用Cuda
        #   没有GPU可以设置成False
        #--------------------------------#
        "cuda"              : True,
    }

    #---------------------------------------------------#
    #   初始化PSPNET
    #---------------------------------------------------#
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)
        # 第六步修改：
        # 若用户只改了 selected_bands，没有手动改 in_channels，
        # 则这里自动同步，避免推理模型和输入通道数不一致。
        if self.selected_bands is not None:
            self.in_channels = len(self.selected_bands)
        #---------------------------------------------------#
        #   画框设置不同的颜色
        #---------------------------------------------------#
        if self.num_classes <= 21:
            self.colors = [ (0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128), (0, 128, 128), 
                            (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0), (192, 128, 0), (64, 0, 128), (192, 0, 128), 
                            (64, 128, 128), (192, 128, 128), (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128), 
                            (128, 64, 12)]
        else:
            hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
            self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
            self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))
        #---------------------------------------------------#
        #   获得模型
        #---------------------------------------------------#
        self.generate()
        
        show_config(**self._defaults)

    #---------------------------------------------------#
    #   获得所有的分类
    #---------------------------------------------------#
    def generate(self, onnx=False):
        #-------------------------------#
        #   载入模型与权值
        #-------------------------------#
        # 第六步修改：
        # 推理时构建模型也传入 in_channels，和训练阶段保持一致。
        self.net    = pspnet(num_classes=self.num_classes, downsample_factor=self.downsample_factor, pretrained=False, backbone=self.backbone, aux_branch=False, in_channels=self.in_channels)
        
        device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net.load_state_dict(torch.load(self.model_path, map_location=device), strict=False)
        self.net    = self.net.eval()
        print('{} model, and classes loaded.'.format(self.model_path))
        if not onnx:
            if self.cuda:
                self.net = nn.DataParallel(self.net)
                self.net = self.net.cuda()

    def read_inference_image(self, image):
        # 第六步修改：
        # 推理输入支持三种形式：
        # 1. 文件路径（.tif / 普通图片）
        # 2. PIL.Image
        # 3. numpy 数组
        if isinstance(image, str):
            if image.lower().endswith((".tif", ".tiff")):
                with rasterio.open(image) as src:
                    if self.selected_bands is None:
                        band_indexes = list(range(1, src.count + 1))
                    else:
                        band_indexes = self.selected_bands
                    arr = src.read(indexes=band_indexes)  # (C, H, W)
                    arr = np.transpose(arr, (1, 2, 0))   # -> (H, W, C)
                return arr
            return Image.open(image)
        return image

    def resize_multiband_image(self, image, size):
        # 第六步修改：
        # 为多波段 numpy 影像补一个 resize + padding 版本。
        ih, iw  = image.shape[:2]
        w, h    = size

        scale   = min(w / iw, h / ih)
        nw      = int(iw * scale)
        nh      = int(ih * scale)

        image   = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
        channels = image.shape[2]
        new_image = np.zeros((h, w, channels), dtype=image.dtype)
        new_image[(h - nh) // 2:(h - nh) // 2 + nh, (w - nw) // 2:(w - nw) // 2 + nw, :] = image
        return new_image, nw, nh

    def make_vis_image(self, image):
        # 第六步修改：
        # 多波段推理时，mix_type=0/2 仍需要一张 RGB 预览图来叠加显示结果。
        if isinstance(image, Image.Image):
            return copy.deepcopy(image)

        image = np.asarray(image)
        idx = [b - 1 for b in self.vis_bands]
        rgb = image[:, :, idx].astype(np.float32)
        out = np.zeros_like(rgb, dtype=np.uint8)
        for i in range(rgb.shape[2]):
            band = rgb[:, :, i]
            low = np.percentile(band, 2)
            high = np.percentile(band, 98)
            if high <= low:
                scaled = np.zeros_like(band, dtype=np.uint8)
            else:
                scaled = ((band - low) / (high - low) * 255.0).clip(0, 255).astype(np.uint8)
            out[:, :, i] = scaled
        return Image.fromarray(out)

    def prepare_input(self, image):
        # 第六步修改：
        # 统一准备推理输入：
        # - RGB/PIL 走原来的 resize_image
        # - 多波段 numpy 走 resize_multiband_image
        # 返回：
        #   old_img      用于可视化叠加的原图预览
        #   image_data   网络输入 BCHW
        #   nw, nh       resize 后有效区域大小
        #   orininal_h/w 原始尺寸
        image = self.read_inference_image(image)

        if isinstance(image, Image.Image):
            image       = cvtColor(image)
            old_img     = copy.deepcopy(image)
            orininal_h  = np.array(image).shape[0]
            orininal_w  = np.array(image).shape[1]
            image_data, nw, nh = resize_image(image, (self.input_shape[1], self.input_shape[0]))
            image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, np.float32)), (2, 0, 1)), 0)
        else:
            old_img     = self.make_vis_image(image)
            orininal_h, orininal_w = image.shape[0], image.shape[1]
            image_data, nw, nh = self.resize_multiband_image(np.array(image, np.float32), (self.input_shape[1], self.input_shape[0]))
            image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, np.float32)), (2, 0, 1)), 0)

        return old_img, image_data, nw, nh, orininal_h, orininal_w

    #---------------------------------------------------#
    #   检测图片
    #---------------------------------------------------#
    def detect_image(self, image, count=False, name_classes=None):
        # 第六步修改：
        # 推理输入现在统一由 prepare_input 准备，兼容 tif 多波段和普通 RGB。
        old_img, image_data, nw, nh, orininal_h, orininal_w = self.prepare_input(image)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
                
            #---------------------------------------------------#
            #   图片传入网络进行预测
            #---------------------------------------------------#
            pr = self.net(images)[0]
            #---------------------------------------------------#
            #   取出每一个像素点的种类
            #---------------------------------------------------#
            pr = F.softmax(pr.permute(1,2,0),dim = -1).cpu().numpy()
            #--------------------------------------#
            #   将灰条部分截取掉
            #--------------------------------------#
            pr = pr[int((self.input_shape[0] - nh) // 2) : int((self.input_shape[0] - nh) // 2 + nh), \
                    int((self.input_shape[1] - nw) // 2) : int((self.input_shape[1] - nw) // 2 + nw)]
            #---------------------------------------------------#
            #   进行图片的resize
            #---------------------------------------------------#
            pr = cv2.resize(pr, (orininal_w, orininal_h), interpolation = cv2.INTER_LINEAR)
            #---------------------------------------------------#
            #   取出每一个像素点的种类
            #---------------------------------------------------#
            pr = pr.argmax(axis=-1)
        
        #---------------------------------------------------------#
        #   计数
        #---------------------------------------------------------#
        if count:
            classes_nums        = np.zeros([self.num_classes])
            total_points_num    = orininal_h * orininal_w
            print('-' * 63)
            print("|%25s | %15s | %15s|"%("Key", "Value", "Ratio"))
            print('-' * 63)
            for i in range(self.num_classes):
                num     = np.sum(pr == i)
                ratio   = num / total_points_num * 100
                if num > 0:
                    print("|%25s | %15s | %14.2f%%|"%(str(name_classes[i]), str(num), ratio))
                    print('-' * 63)
                classes_nums[i] = num
            print("classes_nums:", classes_nums)
    
        if self.mix_type == 0:
            # seg_img = np.zeros((np.shape(pr)[0], np.shape(pr)[1], 3))
            # for c in range(self.num_classes):
            #     seg_img[:, :, 0] += ((pr[:, :] == c ) * self.colors[c][0]).astype('uint8')
            #     seg_img[:, :, 1] += ((pr[:, :] == c ) * self.colors[c][1]).astype('uint8')
            #     seg_img[:, :, 2] += ((pr[:, :] == c ) * self.colors[c][2]).astype('uint8')
            seg_img = np.reshape(np.array(self.colors, np.uint8)[np.reshape(pr, [-1])], [orininal_h, orininal_w, -1])
            #------------------------------------------------#
            #   将新图片转换成Image的形式
            #------------------------------------------------#
            image   = Image.fromarray(np.uint8(seg_img))
            #------------------------------------------------#
            #   将新图与原图及进行混合
            #------------------------------------------------#
            image   = Image.blend(old_img, image, 0.7)

        elif self.mix_type == 1:
            # seg_img = np.zeros((np.shape(pr)[0], np.shape(pr)[1], 3))
            # for c in range(self.num_classes):
            #     seg_img[:, :, 0] += ((pr[:, :] == c ) * self.colors[c][0]).astype('uint8')
            #     seg_img[:, :, 1] += ((pr[:, :] == c ) * self.colors[c][1]).astype('uint8')
            #     seg_img[:, :, 2] += ((pr[:, :] == c ) * self.colors[c][2]).astype('uint8')
            seg_img = np.reshape(np.array(self.colors, np.uint8)[np.reshape(pr, [-1])], [orininal_h, orininal_w, -1])
            #------------------------------------------------#
            #   将新图片转换成Image的形式
            #------------------------------------------------#
            image   = Image.fromarray(np.uint8(seg_img))

        elif self.mix_type == 2:
            seg_img = (np.expand_dims(pr != 0, -1) * np.array(old_img, np.float32)).astype('uint8')
            #------------------------------------------------#
            #   将新图片转换成Image的形式
            #------------------------------------------------#
            image = Image.fromarray(np.uint8(seg_img))
        
        return image

    def get_FPS(self, image, test_interval):
        # 第六步修改：
        # FPS 测试也与正式推理共用同一套多光谱输入准备逻辑。
        _, image_data, nw, nh, _, _ = self.prepare_input(image)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
                
            #---------------------------------------------------#
            #   图片传入网络进行预测
            #---------------------------------------------------#
            pr = self.net(images)[0]
            #---------------------------------------------------#
            #   取出每一个像素点的种类
            #---------------------------------------------------#
            pr = F.softmax(pr.permute(1,2,0),dim = -1).cpu().numpy().argmax(axis=-1)
            #--------------------------------------#
            #   将灰条部分截取掉
            #--------------------------------------#
            pr = pr[int((self.input_shape[0] - nh) // 2) : int((self.input_shape[0] - nh) // 2 + nh), \
                    int((self.input_shape[1] - nw) // 2) : int((self.input_shape[1] - nw) // 2 + nw)]

        t1 = time.time()
        for _ in range(test_interval):
            with torch.no_grad():
                #---------------------------------------------------#
                #   图片传入网络进行预测
                #---------------------------------------------------#
                pr = self.net(images)[0]
                #---------------------------------------------------#
                #   取出每一个像素点的种类
                #---------------------------------------------------#
                pr = F.softmax(pr.permute(1,2,0),dim = -1).cpu().numpy().argmax(axis=-1)
                #--------------------------------------#
                #   将灰条部分截取掉
                #--------------------------------------#
                pr = pr[int((self.input_shape[0] - nh) // 2) : int((self.input_shape[0] - nh) // 2 + nh), \
                        int((self.input_shape[1] - nw) // 2) : int((self.input_shape[1] - nw) // 2 + nw)]
        t2 = time.time()
        tact_time = (t2 - t1) / test_interval
        return tact_time

    def convert_to_onnx(self, simplify, model_path):
        import onnx
        self.generate(onnx=True)

        # 第六步修改：
        # 导出 onnx 时的假输入也改成当前 in_channels。
        im                  = torch.zeros(1, self.in_channels, *self.input_shape).to('cpu')
        input_layer_names   = ["images"]
        output_layer_names  = ["output"]
        
        # Export the model
        print(f'Starting export with onnx {onnx.__version__}.')
        torch.onnx.export(self.net,
                        im,
                        f               = model_path,
                        verbose         = False,
                        opset_version   = 12,
                        training        = torch.onnx.TrainingMode.EVAL,
                        do_constant_folding = True,
                        input_names     = input_layer_names,
                        output_names    = output_layer_names,
                        dynamic_axes    = None)

        # Checks
        model_onnx = onnx.load(model_path)  # load onnx model
        onnx.checker.check_model(model_onnx)  # check onnx model

        # Simplify onnx
        if simplify:
            import onnxsim
            print(f'Simplifying with onnx-simplifier {onnxsim.__version__}.')
            model_onnx, check = onnxsim.simplify(
                model_onnx,
                dynamic_input_shape=False,
                input_shapes=None)
            assert check, 'assert check failed'
            onnx.save(model_onnx, model_path)

        print('Onnx model save as {}'.format(model_path))
    
    def get_miou_png(self, image):
        # 第六步修改：
        # mIoU 预测也复用正式推理的多光谱输入准备逻辑。
        _, image_data, nw, nh, orininal_h, orininal_w = self.prepare_input(image)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
                
            #---------------------------------------------------#
            #   图片传入网络进行预测
            #---------------------------------------------------#
            pr = self.net(images)[0]
            #---------------------------------------------------#
            #   取出每一个像素点的种类
            #---------------------------------------------------#
            pr = F.softmax(pr.permute(1,2,0),dim = -1).cpu().numpy()
            #--------------------------------------#
            #   将灰条部分截取掉
            #--------------------------------------#
            pr = pr[int((self.input_shape[0] - nh) // 2) : int((self.input_shape[0] - nh) // 2 + nh), \
                    int((self.input_shape[1] - nw) // 2) : int((self.input_shape[1] - nw) // 2 + nw)]
            #---------------------------------------------------#
            #   进行图片的resize
            #---------------------------------------------------#
            pr = cv2.resize(pr, (orininal_w, orininal_h), interpolation = cv2.INTER_LINEAR)
            #---------------------------------------------------#
            #   取出每一个像素点的种类
            #---------------------------------------------------#
            pr = pr.argmax(axis=-1)
    
        image = Image.fromarray(np.uint8(pr))
        return image
