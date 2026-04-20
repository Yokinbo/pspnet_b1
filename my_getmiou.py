import os

# ================== 无GUI环境下禁用图形后端，避免Qt/xcb报错 ==================
import matplotlib
matplotlib.use("Agg")   # 必须在 import pyplot 之前
import matplotlib.pyplot as plt
# ========================================================================

import numpy as np
from PIL import Image
from tqdm import tqdm

from pspnet import PSPNet
from utils.utils_metrics import compute_mIoU, show_results


def save_fallback_outputs(miou_out_path, hist, IoUs, PA_Recall, Precision, name_classes):
    """
    当 show_results 因为 cv2/Qt/xcb 在 WSL 或无GUI环境崩溃时，
    使用纯 numpy + matplotlib(Agg) 输出评估结果。
    """
    os.makedirs(miou_out_path, exist_ok=True)

    hist = np.array(hist, dtype=np.float64)
    IoUs = np.array(IoUs, dtype=np.float64)
    PA_Recall = np.array(PA_Recall, dtype=np.float64)
    Precision = np.array(Precision, dtype=np.float64)

    acc = (np.diag(hist).sum() / (hist.sum() + 1e-10)) * 100.0

    # 1. 保存文本汇总
    summary_path = os.path.join(miou_out_path, "miou_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"num_classes: {len(name_classes)}\n\n")
        for i, cls in enumerate(name_classes):
            f.write(
                f"{cls:15s} "
                f"IoU={IoUs[i] * 100:.2f}  "
                f"PA(Recall)={PA_Recall[i] * 100:.2f}  "
                f"Precision={Precision[i] * 100:.2f}\n"
            )
        f.write("\n")
        f.write(f"mIoU={IoUs.mean() * 100:.2f}\n")
        f.write(f"mPA={PA_Recall.mean() * 100:.2f}\n")
        f.write(f"Accuracy={acc:.2f}\n")

    # 2. 保存混淆矩阵
    cm_path = os.path.join(miou_out_path, "confusion_matrix.csv")
    with open(cm_path, "w", encoding="utf-8") as f:
        header = ",".join([""] + list(name_classes))
        f.write(header + "\n")
        for i, cls in enumerate(name_classes):
            row = ",".join([cls] + [str(int(x)) for x in hist[i]])
            f.write(row + "\n")

    # 3. 保存柱状图
    def barh_save(values, title, xlabel, filename):
        values = np.array(values, dtype=np.float64)
        y = np.arange(len(name_classes))

        plt.figure(figsize=(8, 5))
        plt.barh(y, values)
        plt.yticks(y, name_classes)
        plt.xlim(0.0, 1.0)
        plt.xlabel(xlabel)
        plt.title(title)

        for i, v in enumerate(values):
            plt.text(min(v + 0.01, 0.98), i, f"{v:.2f}", va="center")

        plt.tight_layout()
        plt.savefig(os.path.join(miou_out_path, filename), dpi=200)
        plt.close()

    barh_save(IoUs, f"mIoU = {IoUs.mean() * 100:.2f}%", "Intersection over Union", "mIoU.png")
    barh_save(PA_Recall, f"mPA = {PA_Recall.mean() * 100:.2f}%", "Pixel Accuracy (Recall)", "mPA.png")
    barh_save(Precision, "Precision", "Precision", "Precision.png")
    barh_save(PA_Recall, "Recall", "Recall", "Recall.png")

    print(f"[Fallback saved] {summary_path}")
    print(f"[Fallback saved] {cm_path}")
    print(f"[Fallback saved] mIoU.png / mPA.png / Precision.png / Recall.png")


if __name__ == "__main__":
    # ---------------------------------------------------------------------#
    # miou_mode = 0: 预测 + 计算 mIoU + 输出结果
    # miou_mode = 1: 仅生成预测结果
    # miou_mode = 2: 仅计算 mIoU（需要已有 prediction）
    # ---------------------------------------------------------------------#
    miou_mode = 0

    # 背景 + 光伏
    num_classes = 2
    name_classes = ["_background_", "LMC"]

    VOCdevkit_path = "VOCdevkit"
    val_txt_path = os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Segmentation/val.txt")
    image_ids = open(val_txt_path, "r", encoding="utf-8").read().splitlines()

    gt_dir = os.path.join(VOCdevkit_path, "VOC2007/SegmentationClass")
    miou_out_path = "miou_out"
    pred_dir = os.path.join(miou_out_path, "detection-results")

    os.makedirs(miou_out_path, exist_ok=True)

    # ====================== 1) 生成预测结果 ======================
    if miou_mode in (0, 1):
        os.makedirs(pred_dir, exist_ok=True)

        print("Load model.")
        pspnet = PSPNet()
        print("Load model done.")

        print("Get predict result.")
        for image_id in tqdm(image_ids):
            image_path = os.path.join(VOCdevkit_path, "VOC2007/JPEGImages", image_id + ".jpg")
            image = Image.open(image_path)
            pred = pspnet.get_miou_png(image)
            pred.save(os.path.join(pred_dir, image_id + ".png"))
        print("Get predict result done.")

    # ====================== 2) 计算 mIoU ======================
    if miou_mode in (0, 2):
        print("Get miou.")
        hist, IoUs, PA_Recall, Precision = compute_mIoU(
            gt_dir, pred_dir, image_ids, num_classes, name_classes
        )
        print("Get miou done.")

        # ====================== 3) 输出结果 ======================
        try:
            show_results(miou_out_path, hist, IoUs, PA_Recall, Precision, name_classes)
            print("show_results done.")
        except Exception as e:
            print("[WARN] show_results failed, use fallback output mode.")
            print("Error:", repr(e))
            save_fallback_outputs(miou_out_path, hist, IoUs, PA_Recall, Precision, name_classes)

        print("All done. Outputs in:", os.path.abspath(miou_out_path))