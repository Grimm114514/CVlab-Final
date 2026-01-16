import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms as T
from PIL import Image
import numpy as np
import cv2
from tqdm import tqdm
from glob import glob

# 引入项目模块
import network
import utils

# 路径配置
CKPT_PATH = "./checkpoints/best_deeplabv3plus_mobilenet_cityscapes_os16.pth"
INPUT_DIR = "../Images/Images_2"
OUTPUT_DIR = "../Masks/Mask2"


GPU_ID = '0'

def main():
    # 设备配置
    os.environ['CUDA_VISIBLE_DEVICES'] = GPU_ID
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")

    # 根据权重文件自动选择模型架构
    model_name = 'deeplabv3plus_mobilenet' 
    if 'resnet101' in os.path.basename(CKPT_PATH):
        model_name = 'deeplabv3plus_resnet101'
    
    print(f"模型: {model_name}")
    # 创建模型 (19类是Cityscapes数据集的类别数)
    model = network.modeling.__dict__[model_name](num_classes=19, output_stride=16)
    
    # 加载预训练权重
    if os.path.isfile(CKPT_PATH):
        checkpoint = torch.load(CKPT_PATH, map_location=torch.device('cpu'),weights_only=False)
        if "model_state" in checkpoint:
            model.load_state_dict(checkpoint["model_state"])
        else:
            model.load_state_dict(checkpoint)
        print("权重加载完成")
    else:
        print(f"错误: 找不到权重文件 {CKPT_PATH}")
        return

    model = nn.DataParallel(model)
    model.to(device)
    model.eval()

    # 图像预处理: 转tensor并使用ImageNet标准化参数
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.229, 0.225]),
    ])

    # 查找所有图片文件
    image_files = glob(os.path.join(INPUT_DIR, '*.jpg')) + \
                  glob(os.path.join(INPUT_DIR, '*.png')) + \
                  glob(os.path.join(INPUT_DIR, '*.jpeg'))
    
    print(f"共 {len(image_files)} 张图片")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 批量推理并保存分割结果
    with torch.no_grad():
        for img_path in tqdm(image_files):
            filename = os.path.basename(img_path)
            basename = os.path.splitext(filename)[0]
            
            # 读取并预处理
            raw_img = Image.open(img_path).convert('RGB')
            w, h = raw_img.size
            
            # 模型推理
            input_tensor = transform(raw_img).unsqueeze(0).to(device)
            output = model(input_tensor)
            pred_map = output.max(1)[1].cpu().numpy()[0]  # 取argmax得到类别ID
            
            # 恢复到原始尺寸 (使用最近邻插值保持类别ID)
            pred_map_resized = cv2.resize(pred_map.astype('uint8'), (w, h), interpolation=cv2.INTER_NEAREST)
            
            # 保存分割掩码 (像素值=类别ID)
            save_path = os.path.join(OUTPUT_DIR, basename + ".png")
            cv2.imwrite(save_path, pred_map_resized)

    print(f"\n完成! 输出目录: {OUTPUT_DIR}")

if __name__ == '__main__':
    main()