import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image
import torch
import torchvision.models as models
import torchvision.transforms as T

# ===================== 全局配置 =====================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = (480, 480)
PATCH_SIZE = 16
K_NEAREST = 9
# 全局模型缓存
patch_model = {
    "feature_memory": [],
    "backbone": None,
    "transform": None,
    "trained": False
}
test_img_path = None

# ===================== 图片读取（兼容中文路径） =====================
def cv_imread(path):
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)

# ===================== 初始化骨干网络 ResNet18 =====================
def init_backbone():
    backbone = models.resnet18(pretrained=True)
    backbone.eval()
    backbone.to(DEVICE)
    # 提取中间层特征
    feature_extractor = torch.nn.Sequential(*list(backbone.children())[:-3])
    # 图像归一化变换
    transform = T.Compose([
        T.Resize(IMG_SIZE),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return feature_extractor, transform

# ===================== 提取单张图所有Patch特征 =====================
def extract_patch_features(img_pil, extractor, transform):
    img_tensor = transform(img_pil).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        feat_map = extractor(img_tensor)  # [1, C, H, W]
    B, C, H, W = feat_map.shape
    # 展开所有patch特征
    patches = feat_map.permute(0,2,3,1).reshape(-1, C)
    return patches.cpu().numpy()

# ===================== 训练：载入全部干净图片，构建特征记忆库 =====================
def train_patchcore(train_folder):
    global patch_model
    # 初始化网络
    extractor, transform = init_backbone()
    patch_model["backbone"] = extractor
    patch_model["transform"] = transform

    # 遍历训练图片
    suffix = (".jpg", ".png", ".jpeg")
    file_list = [f for f in os.listdir(train_folder) if f.lower().endswith(suffix)]
    if len(file_list) < 10:
        return 0, "训练图片至少10张，建议20张以上不同视角干净图"
    
    all_features = []
    for name in file_list:
        full_path = os.path.join(train_folder, name)
        img_bgr = cv_imread(full_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        feat = extract_patch_features(img_pil, extractor, transform)
        all_features.append(feat)
    
    # 合并全部patch特征存入记忆库
    memory = np.vstack(all_features)
    patch_model["feature_memory"] = memory
    patch_model["trained"] = True
    return len(file_list), "PatchCore深度学习模型训练完成"

# ===================== 推理：计算异常热力图 =====================
def predict_anomaly_map(img_pil, extractor, transform, memory):
    feat_test = extract_patch_features(img_pil, extractor, transform)
    H_feat = IMG_SIZE[0] // PATCH_SIZE
    W_feat = IMG_SIZE[1] // PATCH_SIZE

    # 逐patch计算与记忆库最小距离
    anomaly_scores = []
    for f in feat_test:
        dists = np.sqrt(np.sum((memory - f)**2, axis=1))
        min_dist = np.min(dists)
        anomaly_scores.append(min_dist)
    
    # 重塑为热力图尺寸
    score_map = np.array(anomaly_scores).reshape(H_feat, W_feat)
    # 归一化0~255
    score_map = (score_map - score_map.min()) / (score_map.max() - score_map.min() + 1e-6) * 255
    return score_map.astype(np.uint8)

# ===================== 核心修改：红框精准收缩至热力红色中心，无多余留白 =====================
def get_anomaly_boxes(original_bgr, anomaly_map, threshold=60):
    h_ori, w_ori = original_bgr.shape[:2]
    # 1. 热力图强制对齐原图分辨率，像素坐标一一对应
    heat_resize = cv2.resize(anomaly_map, (w_ori, h_ori), interpolation=cv2.INTER_LINEAR)

    # 2. 只保留全局亮度80%以上纯红色核心区域
    global_max_val = np.max(heat_resize)
    red_only_mask = heat_resize >= (global_max_val * 0.8)
    red_region = np.where(red_only_mask, heat_resize, 0)

    # 3. 二值化红色区域
    _, binary = cv2.threshold(red_region, threshold, 255, cv2.THRESH_BINARY)

    # 4. 极小形态核，不扩大红色区域范围
    kernel = np.ones((2, 2), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # 5. 轮廓提取，仅保留面积最大单一块（唯一小人）
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    draw_img = original_bgr.copy()
    count = 0
    total_pixel = h_ori * w_ori

    valid_contours = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        area_ratio = area / total_pixel * 100
        if 0.001 < area_ratio < 0.8:
            valid_contours.append((area, cnt))
    
    # 只取最大轮廓，框大小严格贴合红色区域最小外接矩形
    if len(valid_contours) > 0:
        valid_contours.sort(key=lambda x: x[0], reverse=True)
        best_cnt = valid_contours[0][1]
        x, y, bw, bh = cv2.boundingRect(best_cnt)
        # ========== 关键改动：移除固定大pad，仅微小2像素留白，框精准贴合红色中心 ==========
        tiny_pad = 2
        x1 = max(0, x - tiny_pad)
        y1 = max(0, y - tiny_pad)
        x2 = min(w_ori, x + bw + tiny_pad)
        y2 = min(h_ori, y + bh + tiny_pad)
        cv2.rectangle(draw_img, (x1, y1), (x2, y2), (0, 0, 255), 3)
        count = 1
    return draw_img, count

# ===================== 可视化展示 =====================
def show_result(ori_rgb, heatmap, draw_rgb, num):
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.figure(figsize=(15,5))

    plt.subplot(1,3,1)
    plt.title("输入全新视角截图")
    plt.imshow(ori_rgb)
    plt.axis("off")

    plt.subplot(1,3,2)
    plt.title("深度学习异常热力图")
    plt.imshow(heatmap, cmap="jet")
    plt.axis("off")

    plt.subplot(1,3,3)
    plt.title(f"PatchCore检测结果，共{num}处躲藏者")
    plt.imshow(draw_rgb)
    plt.axis("off")
    plt.tight_layout()
    plt.show()

# ===================== GUI交互逻辑 =====================
def select_train_folder():
    folder = filedialog.askdirectory(title="选择【全部干净无小人截图】文件夹训练PatchCore")
    if not folder:
        return
    cnt, msg = train_patchcore(folder)
    if cnt == 0:
        messagebox.showerror("训练失败", msg)
    else:
        lab_train_info.config(text=f"训练完成，共加载{cnt}张干净场景图")

def select_test_image():
    global test_img_path
    test_img_path = filedialog.askopenfilename(
        title="选择带躲藏者的全新视角截图",
        filetypes=[("图片", "*.jpg;*.png;*.jpeg")]
    )
    if test_img_path:
        lab_test_info.config(text=f"待检测图：{os.path.basename(test_img_path)}")

def run_detect():
    global test_img_path
    if not patch_model["trained"]:
        messagebox.showwarning("提示", "请先选择干净图片文件夹完成深度学习训练！")
        return
    if test_img_path is None:
        messagebox.showwarning("提示", "请选择一张待检测游戏截图")
        return
    # 读取测试图
    bgr = cv_imread(test_img_path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    # 推理生成热力图
    heatmap = predict_anomaly_map(
        pil_img,
        patch_model["backbone"],
        patch_model["transform"],
        patch_model["feature_memory"]
    )
    # 绘制红框
    draw_bgr, hide_count = get_anomaly_boxes(bgr, heatmap, threshold=60)
    draw_rgb = cv2.cvtColor(draw_bgr, cv2.COLOR_BGR2RGB)
    # 绘图展示
    show_result(rgb, heatmap, draw_rgb, hide_count)
    # 弹窗提示
    if hide_count == 0:
        tip = "未检测到躲藏者"
    else:
        tip = f"深度学习PatchCore检测完成，发现{hide_count}名躲藏者"
    messagebox.showinfo("检测结果", tip)

# ===================== GUI主窗口 =====================
if __name__ == "__main__":
    root = tk.Tk()
    root.title("PatchCore 深度学习躲藏者检测系统")
    root.geometry("560x400")
    root.resizable(False, False)

    tk.Label(root, text="PatchCore 无监督深度学习异常检测", font=("黑体",16,"bold")).pack(pady=15)
    tk.Label(root, text="仅用干净地图训练，泛化识别任意新视角躲藏者", fg="#444").pack()

    # 训练模块
    lab_train_info = tk.Label(root, text="未训练深度学习模型", fg="#555")
    lab_train_info.pack(pady=12)
    tk.Button(root, text="1、选择干净截图文件夹训练", command=select_train_folder, width=26, height=2).pack()

    # 测试图片模块
    lab_test_info = tk.Label(root, text="未选择待检测截图", fg="#555")
    lab_test_info.pack(pady=12)
    tk.Button(root, text="2、选择带躲藏者截图", command=select_test_image, width=26, height=2).pack()

    # 检测按钮
    tk.Button(root, text="🔥 PatchCore深度学习检测", command=run_detect, bg="#c41e3a", fg="white", font=("黑体",12), width=26, height=2).pack(pady=20)

    root.mainloop()