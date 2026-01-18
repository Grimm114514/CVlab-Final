import os
import torch
import torch.nn as nn
from torchvision import transforms as T
from PIL import Image
import numpy as np
import cv2
from tqdm import tqdm
from glob import glob

os.makedirs("../Masks/Images_For_Colmap_1", exist_ok=True)
import network
import utils

# 配置参数
CKPT_PATH = "./checkpoints/best_deeplabv3plus_mobilenet_cityscapes_os16.pth"
INPUT_DIR = "../Images/Images_4"
OUTPUT_MASK_DIR = "../Masks/Mask4"
OUTPUT_COLMAP_IMG_DIR = "../Masks/Images_For_Colmap_4"
GPU_ID = '0'

# 动态物体类别 (Person, Rider, Car, Truck, Bus, Train, Motorcycle, Bicycle)
DYNAMIC_CLASSES = [11, 12, 13, 14, 15, 16, 17, 18]

def main():
    # 初始化设备
    os.environ['CUDA_VISIBLE_DEVICES'] = GPU_ID
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"运行设备: {device}")

    # 加载模型
    model_name = 'deeplabv3plus_mobilenet' 
    if 'resnet101' in os.path.basename(CKPT_PATH):
        model_name = 'deeplabv3plus_resnet101'
    
    print(f"加载模型: {model_name}...")
    model = network.modeling.__dict__[model_name](num_classes=19, output_stride=16)
    
    if os.path.isfile(CKPT_PATH):
        checkpoint = torch.load(CKPT_PATH, map_location=torch.device('cpu'), weights_only=False)
        if "model_state" in checkpoint:
            model.load_state_dict(checkpoint["model_state"])
        else:
            model.load_state_dict(checkpoint)
        print("权重加载成功！")
    else:
        print(f"错误: 找不到权重文件 {CKPT_PATH}")
        return

    model = nn.DataParallel(model)
    model.to(device)
    model.eval()

    # 图像预处理
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 准备文件列表
    image_files = glob(os.path.join(INPUT_DIR, '*.jpg')) + \
                  glob(os.path.join(INPUT_DIR, '*.png')) + \
                  glob(os.path.join(INPUT_DIR, '*.jpeg'))
    
    print(f"共发现 {len(image_files)} 张图片，开始处理...")
    os.makedirs(OUTPUT_MASK_DIR, exist_ok=True)
    os.makedirs(OUTPUT_COLMAP_IMG_DIR, exist_ok=True)

    # 批量推理
    with torch.no_grad():
        for img_path in tqdm(image_files):
            filename = os.path.basename(img_path)
            basename = os.path.splitext(filename)[0]
            
            # 读取图片
            raw_pil = Image.open(img_path).convert('RGB')
            raw_cv = cv2.imread(img_path)
            w, h = raw_pil.size
            
            # 语义分割
            input_tensor = transform(raw_pil).unsqueeze(0).to(device)
            output = model(input_tensor)
            
            pred_map = output.max(1)[1].cpu().numpy()[0]
            pred_map_resized = cv2.resize(pred_map.astype('uint8'), (w, h), interpolation=cv2.INTER_NEAREST)
            
            # 保存 Mask
            mask_save_path = os.path.join(OUTPUT_MASK_DIR, basename + ".png")
            cv2.imwrite(mask_save_path, pred_map_resized)
            
            # 去除动态物体
            mask_dynamic = np.isin(pred_map_resized, DYNAMIC_CLASSES)
            masked_img = raw_cv.copy()
            masked_img[mask_dynamic] = [0, 0, 0] 
            
            # 保存涂改后的图片
            colmap_img_path = os.path.join(OUTPUT_COLMAP_IMG_DIR, filename)
            cv2.imwrite(colmap_img_path, masked_img)

    print("\n处理完成！")

if __name__ == '__main__':
    main()