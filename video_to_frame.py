import cv2
import os
from tqdm import tqdm

# 配置参数
CONFIG = {
    "video_path": "./video/1.mp4",  # 视频路径
    "output_dir": "./Images/Images_3",  # 输出目录
    "frame_interval": 60,  # 每隔多少帧保存一张
    "resize_factor": 1.0  # 图片缩放比例 (1.0=原始大小)
}

def main():
    video_path = CONFIG["video_path"]
    output_dir = CONFIG["output_dir"]
    interval = CONFIG["frame_interval"]
    
    if not os.path.exists(video_path):
        print(f"❌ 错误: 找不到视频文件 -> {video_path}")
        return

    os.makedirs(output_dir, exist_ok=True)
    print(f"📂 图片将保存在: {output_dir}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ 无法打开视频，请检查格式或路径。")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps if fps > 0 else 0
    
    print(f"📹 视频信息: 时长 {duration:.1f}秒 | 总帧数 {total_frames} | FPS {fps:.1f}")
    print(f"⚙️ 抽帧间隔: 每 {interval} 帧 (约 {interval/fps:.2f} 秒) 保存一张")
    
    count = 0
    saved_count = 0
    pbar = tqdm(total=total_frames, unit="frame")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if count % interval == 0:
            if CONFIG["resize_factor"] != 1.0:
                h, w = frame.shape[:2]
                new_size = (int(w * CONFIG["resize_factor"]), int(h * CONFIG["resize_factor"]))
                frame = cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)
            
            filename = f"frame_{saved_count:05d}.jpg"
            save_path = os.path.join(output_dir, filename)
            cv2.imwrite(save_path, frame)
            saved_count += 1
        
        count += 1
        pbar.update(1)
        
    cap.release()
    pbar.close()
    
    print(f"\n✅ 处理完成！")
    print(f"📊 共提取了 {saved_count} 张图片。")
    print(f"👉 下一步: 请将 {output_dir} 作为输入文件夹，运行 DeepLab 生成 Mask。")

if __name__ == "__main__":
    main()