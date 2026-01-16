# -*- coding: utf-8 -*-
import os

def rename_images():
	# 获取当前脚本所在目录
	dir_path = os.path.dirname(os.path.abspath(__file__))
	# 支持的图片扩展名
	exts = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.gif']
	# 获取所有图片文件
	files = [f for f in os.listdir(dir_path) if os.path.splitext(f)[1].lower() in exts]
	files.sort()  # 按文件名排序
	for idx, filename in enumerate(files, 1):
		ext = os.path.splitext(filename)[1]
		new_name = f"{idx}{ext}"
		src = os.path.join(dir_path, filename)
		dst = os.path.join(dir_path, new_name)
		if src != dst:
			os.rename(src, dst)
	print(f"已重命名{len(files)}个图片文件。")

if __name__ == "__main__":
	rename_images()
