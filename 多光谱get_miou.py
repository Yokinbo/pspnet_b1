import os

from tqdm import tqdm

from multispectral_config import image_ext, in_channels, selected_bands, trained_model_path
from pspnet import PSPNet
from utils.utils_metrics import compute_mIoU, show_results

"""
多光谱 mIoU 评估脚本。

适配你当前任务：
1. 数据仍使用 VOC 目录结构
2. 影像支持 tif
3. 支持 4 波段 / 6 波段切换
4. 预测时直接复用当前已经打通的多光谱推理链路
"""

if __name__ == "__main__":
    # ---------------------------------------------------------------------------#
    #   miou_mode 用于指定该文件运行时计算的内容
    #   miou_mode = 0 代表整个 miou 计算流程，包括获得预测结果、计算 miou
    #   miou_mode = 1 代表仅仅获得预测结果
    #   miou_mode = 2 代表仅仅计算 miou
    # ---------------------------------------------------------------------------#
    miou_mode = 0

    # ------------------------------#
    #   分类个数（包含背景）
    # ------------------------------#
    num_classes = 2

    # --------------------------------------------#
    #   类别名称
    # --------------------------------------------#
    name_classes = ["_background_", "PV"]

    # -------------------------------------------------------#
    #   指向 VOC 数据集所在的文件夹
    # -------------------------------------------------------#
    vocdevkit_path = "VOCdevkit"

    # 第八步修改：
    # 多光谱配置统一从 multispectral_config.py 读取，
    # 以后切换 3/4/6 波段时不需要在这里重复改。

    # -------------------------------------------------------#
    #   使用哪个划分文件进行评估
    #   一般评估 val.txt
    # -------------------------------------------------------#
    image_set = "val.txt"

    image_ids = open(
        os.path.join(vocdevkit_path, "VOC2007/ImageSets/Segmentation", image_set),
        "r",
        encoding="utf-8",
    ).read().splitlines()

    gt_dir = os.path.join(vocdevkit_path, "VOC2007/SegmentationClass")
    miou_out_path = "miou_out"
    pred_dir = os.path.join(miou_out_path, "detection-results")

    if miou_mode == 0 or miou_mode == 1:
        if not os.path.exists(pred_dir):
            os.makedirs(pred_dir)

        print("Load model.")
        # 第七步修改：
        # 这里直接把多光谱配置传给推理包装类，
        # 保证评估时和训练时使用同样的波段与输入通道数。
        pspnet = PSPNet(
            model_path=trained_model_path,
            num_classes=num_classes,
            image_ext=image_ext,
            selected_bands=selected_bands,
            in_channels=in_channels,
        )
        print("Load model done.")

        print("Get predict result.")
        for image_id in tqdm(image_ids):
            # 第七步修改：
            # 对 tif 多光谱影像，直接将路径传给 get_miou_png，
            # 底层会自动按 selected_bands 读取。
            image_path = os.path.join(
                vocdevkit_path, "VOC2007/JPEGImages", image_id + image_ext
            )
            image = pspnet.get_miou_png(image_path)
            image.save(os.path.join(pred_dir, image_id + ".png"))
        print("Get predict result done.")

    if miou_mode == 0 or miou_mode == 2:
        print("Get miou.")
        hist, IoUs, PA_Recall, Precision = compute_mIoU(
            gt_dir, pred_dir, image_ids, num_classes, name_classes
        )
        print("Get miou done.")
        show_results(miou_out_path, hist, IoUs, PA_Recall, Precision, name_classes)
