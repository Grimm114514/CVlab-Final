import os
import numpy as np
import cv2
from tqdm import tqdm
import shutil

# 配置参数
CONFIG = {
    "colmap_text_dir": "./models/col4",  # COLMAP 模型目录
    "mask_dir": "./Masks/Mask4",  # DeepLab Mask 目录
    "output_dir": "./models/final_semantic_model4",  # 输出目录
    
    # 语义类别颜色映射
    "color_map": {
        0:  [128, 128, 128],  # Road
        1:  [128, 128, 128],  # Sidewalk
        2:  [128, 0, 0],      # Building
        3:  [128, 64, 128],   # Wall
        4:  [128, 128, 0],    # Fence
        8:  [0, 128, 0],      # Vegetation
        9:  [0, 255, 0],      # Terrain
    },
    
    "default_color": [0, 0, 0]  # 未识别点默认颜色
}

class ImageObj:
    def __init__(self, id, name, xys, point3D_ids):
        self.id = id
        self.name = name
        self.xys = xys
        self.point3D_ids = point3D_ids

class Point3D:
    def __init__(self, id, xyz, rgb, error, image_ids, point2D_idxs):
        self.id = id
        self.xyz = xyz
        self.rgb = rgb
        self.error = error
        self.image_ids = image_ids
        self.point2D_idxs = point2D_idxs

def read_images_text(path):
    """读取 images.txt"""
    images = {}
    if not os.path.exists(path):
        print(f"错误: 文件不存在 {path}")
        return images
        
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line: break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                image_id = int(elems[0])
                image_name = elems[-1] 
                
                line = fid.readline()
                elems = line.split()
                points2D = np.array(elems).reshape(-1, 3)
                xys = points2D[:, :2].astype(float)
                point3D_ids = points2D[:, 2].astype(int)
                
                images[image_id] = ImageObj(image_id, image_name, xys, point3D_ids)
    return images

def read_points3D_text(path):
    """读取 points3D.txt"""
    points3D = {}
    if not os.path.exists(path):
        print(f"错误: 文件不存在 {path}")
        return points3D

    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line: break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                point3D_id = int(elems[0])
                xyz = np.array(elems[1:4], dtype=float)
                rgb = np.array(elems[4:7], dtype=int)
                error = float(elems[7])
                track = np.array(elems[8:], dtype=int).reshape(-1, 2)
                image_ids = track[:, 0]
                point2D_idxs = track[:, 1]
                points3D[point3D_id] = Point3D(point3D_id, xyz, rgb, error, image_ids, point2D_idxs)
    return points3D

def write_points3D_text(points3D, path):
    """写入 points3D.txt"""
    print(f"保存点云: {path}")
    with open(path, "w") as fid:
        fid.write("# 3D point list with one line of data per point:\n")
        fid.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        for point in points3D.values():
            track_str = ""
            for i in range(len(point.image_ids)):
                track_str += f" {point.image_ids[i]} {point.point2D_idxs[i]}"
            fid.write(f"{point.id} {point.xyz[0]} {point.xyz[1]} {point.xyz[2]} "
                      f"{point.rgb[0]} {point.rgb[1]} {point.rgb[2]} {point.error}{track_str}\n")

def main():
    print("开始语义点云融合")
    
    # 读取 COLMAP 模型
    print(f"读取模型: {CONFIG['colmap_text_dir']}")
    images_map = read_images_text(os.path.join(CONFIG['colmap_text_dir'], "images.txt"))
    points3D_map = read_points3D_text(os.path.join(CONFIG['colmap_text_dir'], "points3D.txt"))
    
    if len(images_map) == 0 or len(points3D_map) == 0:
        print("错误: 模型数据加载失败")
        return

    print(f"图像数: {len(images_map)}")
    print(f"点云数: {len(points3D_map)}")

    # 预加载 Masks
    print("加载 Masks...")
    masks_cache = {}
    loaded_count = 0
    
    for img_id, img_obj in tqdm(images_map.items()):
        base_name = os.path.splitext(img_obj.name)[0]
        mask_filename = base_name + ".png"
        mask_path = os.path.join(CONFIG['mask_dir'], mask_filename)
        
        if os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
            masks_cache[img_id] = mask
            loaded_count += 1
    
    print(f"-> 成功加载 {loaded_count} 张 Mask")

    # 语义投票上色
    print("开始为 3D 点进行语义投票...")
    
    modified_count = 0
    
    for point_id, point in tqdm(points3D_map.items()):
        votes = []
        
        for i in range(len(point.image_ids)):
            img_id = point.image_ids[i]
            
            if img_id in masks_cache:
                img_obj = images_map[img_id]
                mask = masks_cache[img_id]
                
                point2d_idx = point.point2D_idxs[i]
                u, v = img_obj.xys[point2d_idx]
                u, v = int(u), int(v)
                
                h, w = mask.shape
                if 0 <= v < h and 0 <= u < w:
                    label_id = mask[v, u]
                    votes.append(label_id)
        
        # 投票决定颜色
        final_color = CONFIG["default_color"]
        
        if votes:
            valid_votes = [v for v in votes if v != 255]
            
            if valid_votes:
                most_common_id = max(set(valid_votes), key=valid_votes.count)
                
                if most_common_id in CONFIG["color_map"]:
                    final_color = CONFIG["color_map"][most_common_id]
                else:
                    final_color = CONFIG["default_color"]
        
        point.rgb = np.array(final_color)
        modified_count += 1

    # 保存结果
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    out_path = os.path.join(CONFIG["output_dir"], "points3D.txt")
    write_points3D_text(points3D_map, out_path)
    
    # 复制其他文件(src_img, CONFIG["output_dir"])
    
    print("复制相机参数文件...")
    src_cam = os.path.join(CONFIG['colmap_text_dir'], "cameras.txt")
    src_img = os.path.join(CONFIG['colmap_text_dir'], "images.txt")
    if os.path.exists(src_cam): shutil.copy(src_cam, CONFIG["output_dir"])
    if os.path.exists(src_img): shutil.copy
    print(f"结果已保存在: {CONFIG['output_dir']}")
if __name__ == "__main__":
    main()