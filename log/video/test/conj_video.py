import os
import subprocess

video_dir = "./"
video_files = sorted([f for f in os.listdir(video_dir) if f.endswith('.mp4')])[:100]  # 取前100个

# 生成复杂的filter_complex参数
filter_str = ""
inputs = ""
for i, video in enumerate(video_files):
    inputs += f" -i {os.path.join(video_dir, video)}"
    filter_str += f"[{i}:v]scale=320:240[v{i}];"  # 统一缩放到320x240

# 假设10x10网格布局 (100个视频)
layout = ""
for row in range(10):
    for col in range(10):
        index = row * 10 + col
        if index < len(video_files):
            layout += f"[v{index}]"
layout += f"xstack=inputs={len(video_files)}:layout="

# 添加布局位置信息
positions = []
for row in range(10):
    for col in range(10):
        positions.append(f"{col*320}_{row*240}")
layout += "|".join(positions)

filter_str += layout

subprocess.run(
    f"ffmpeg{inputs} -filter_complex \"{filter_str}\" -c:v libx264 -preset fast output.mp4",
    shell=True
)