import os
import numpy as np
import cv2
from tqdm import tqdm

CONFIG = {
    # COLMAP 导出的 txt 文件夹路径
    "colmap_text_dir": "./models/origin_2",
    
    # DeepLab 生成的 Mask 文件夹路径
    "mask_dir": "./Masks/Mask2",
    
    # 输出的新模型保存路径
    "output_dir": "./models/sparse_semantic_2",
    
    # 颜色映射 (R, G, B)
    "color_map": {
        2:  [255, 0, 0],    # 建筑 -> 红色
        8:  [0, 255, 0],    # 植被 -> 绿色
        0:  [128, 128, 128],# 道路 -> 灰色
        1:  [128, 128, 128],# 人行道 -> 灰色
    },
    
    # 默认颜色 (黑色)
    "default_color": [0, 0, 0]
}

class Image:
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
    images = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line: break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                image_id = int(elems[0])
                image_name = elems[-1] # 文件名
                
                # 读取下一行 (Points2D)
                line = fid.readline()
                elems = line.split()
                points2D = np.array(elems).reshape(-1, 3)
                xys = points2D[:, :2].astype(float)
                point3D_ids = points2D[:, 2].astype(int)
                
                images[image_id] = Image(image_id, image_name, xys, point3D_ids)
    return images

def read_points3D_text(path):
    points3D = {}
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
    print("开始语义融合...")
    
    # 读取 COLMAP 数据
    print(f"正在读取 COLMAP 模型: {CONFIG['colmap_text_dir']}")
    images_map = read_images_text(os.path.join(CONFIG['colmap_text_dir'], "images.txt"))
    points3D_map = read_points3D_text(os.path.join(CONFIG['colmap_text_dir'], "points3D.txt"))
    print(f"   -> 加载了 {len(images_map)} 张位姿图像")
    print(f"   -> 加载了 {len(points3D_map)} 个稀疏 3D 点")

    # 预加载所有 Masks
    print("正在预加载 Mask 图片...")
    masks_cache = {}
    
    for img_id, img_obj in tqdm(images_map.items()):
        base_name = os.path.splitext(img_obj.name)[0]
        mask_filename = base_name + ".png"
        mask_path = os.path.join(CONFIG['mask_dir'], mask_filename)
        
        if os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
            masks_cache[img_id] = mask
        else:
            pass

    print(f"   -> 成功缓存了 {len(masks_cache)} 张 Mask")
    if len(masks_cache) == 0:
        print("错误: 一张 Mask 也没读到，请检查文件名和路径配置！")
        return

    # 遍历 3D 点进行语义投票
    print("正在为 3D 点上色 (Voting)...")
    
    modified_count = 0
    for point_id, point in tqdm(points3D_map.items()):
        votes = []
        
        for i in range(len(point.image_ids)):
            img_id = point.image_ids[i]
            point2d_idx = point.point2D_idxs[i]
            
            if img_id in masks_cache:
                img_obj = images_map[img_id]
                mask = masks_cache[img_id]
                
                x, y = img_obj.xys[point2d_idx]
                x, y = int(x), int(y)
                
                h, w = mask.shape
                if 0 <= y < h and 0 <= x < w:
                    label_id = mask[y, x]
                    votes.append(label_id)
        
        # 决定颜色
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
    print(f"正在保存结果到: {out_path}")
    write_points3D_text(points3D_map, out_path)
    
    # 复制 images.txt 和 cameras.txt
    import shutil
    shutil.copy(os.path.join(CONFIG['colmap_text_dir'], "cameras.txt"), CONFIG["output_dir"])
    shutil.copy(os.path.join(CONFIG['colmap_text_dir'], "images.txt"), CONFIG["output_dir"])
    
    print("全部完成！")

if __name__ == "__main__":
    main()