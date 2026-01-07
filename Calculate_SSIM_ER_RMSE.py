import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim


def get_edge_ratio(pred, label):#Edge ratio=Edge_area.sum()/Total.sum()
    """
    计算边缘值占比
    pred, label: 形状为 (H, W) 的 Tensor, 取值 [0, 1]
    """
    # 转为 (1, 1, H, W) 以便进行卷积
    pred_t = pred.unsqueeze(0).unsqueeze(0)
    label_t = label.unsqueeze(0).unsqueeze(0)

    # 定义 3x3 全 1 卷积核
    kernel = torch.ones((1, 1, 3, 3), device=pred.device)

    # 对 Label 滤波，使用 padding=1 保持尺寸一致
    filtered_label = F.conv2d(label_t, kernel, padding=1)

    # 根据要求定义 Mask: 滤波结果 > 1 的 pixel (注意：若只要覆盖原边附近，通常>0即可，此处严格遵照指令用>1)
    mask = (filtered_label > 0).float()

    # 计算边缘合计值 (不二值化的 Prediction * Mask)
    edge_sum = torch.sum(pred_t * mask)
    # 计算 Prediction 所有 pixel 之和
    total_sum = torch.sum(pred_t)

    # 返回比例，防止除以 0
    return (edge_sum / (total_sum + 1e-8)).item()

def evaluate_SSIM_ER_RMSE(label_path, pred_path, Binarization_threshold=None):
    """
    主评估函数
    """
    label_files = sorted(os.listdir(label_path))
    pred_files = sorted(os.listdir(pred_path))

    # 结果容器
    ssim_raw_list = []
    ssim_bin_list = []
    ratio_raw_list = []
    ratio_bin_list = []
    RMSE_raw_list = []
    RMSE_bin_list = []

    for f in label_files:
        if f not in pred_files:
            continue

        # 1. 读取并归一化为 [0, 1] 的 float32 (OpenCV 读取默认是 uint8)
        # 使用灰度模式读取
        img_label = cv2.imread(os.path.join(label_path, f), 0).astype(np.float32)
        if img_label.max()>1:
            img_label=img_label/255
        img_pred = cv2.imread(os.path.join(pred_path, f), 0).astype(np.float32)
        if img_pred.max() > 1:
            img_pred = img_pred / 255

        # 转换为 Tensor 供卷积计算使用
        t_label = torch.from_numpy(img_label).float()
        t_pred = torch.from_numpy(img_pred).float()

        # --- 不做二值化的平均 SSIM和边缘占比 ---
        # data_range=1.0 对应 [0, 1] 范围
        s_raw = ssim(img_label, img_pred, data_range=1.0)
        ssim_raw_list.append(s_raw)
        r_raw = get_edge_ratio(t_pred, t_label)
        ratio_raw_list.append(r_raw)
        RMSE_raw=torch.sqrt(((t_label-t_pred)**2).mean())
        RMSE_raw_list.append(RMSE_raw)

        # 2. 二值化后的图像平均 SSIM和边缘占比
        if Binarization_threshold!=None:
            img_pred_bin = (img_pred > Binarization_threshold).astype(np.float32)
            t_pred_bin = torch.from_numpy(img_pred_bin).float()
            s_bin = ssim(img_label, img_pred_bin, data_range=1.0)
            ssim_bin_list.append(s_bin)
            r_bin = get_edge_ratio(t_pred_bin, t_label)
            ratio_bin_list.append(r_bin)
            RMSE_bin = torch.sqrt(((t_label - t_pred_bin) ** 2).mean())
            RMSE_bin_list.append(RMSE_bin)

    # 输出平均值
    print(f"Total {len(ssim_raw_list)} images：")
    print(f"Avg SSIM (Raw):       {np.mean(ssim_raw_list):.3f}")
    print(f"Avg Edge ratio (Raw): {np.mean(ratio_raw_list):.3f}")
    print(f"Avg RMSE (Raw):       {np.mean(RMSE_raw_list):.3f}")

    if Binarization_threshold != None:
        print(f"Avg SSIM (Binarization):       {np.mean(ssim_bin_list):.3f}")
        print(f"Avg Edge ratio (Binarization): {np.mean(ratio_bin_list):.3f}")
        print(f"Avg RMSE:                      {np.mean(RMSE_bin_list):.3f}")
    print("-" * 30)
####################################################################


if __name__=='__main__':

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = 'cuda'

    evaluate_SSIM_ER_RMSE(label_path='', pred_path='', Binarization_threshold=None)

    ###############################################################################################
