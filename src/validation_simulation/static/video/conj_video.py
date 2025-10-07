import os
import subprocess
import re

# Extract the numerical part and sort by numerical value
def extract_number(filename):
    match = re.search(r'_(\d+)\.mp4', filename)
    return int(match.group(1)) if match else float('inf')

video_dir = "./"
# Sort by numerical value
video_files = sorted(
    [f for f in os.listdir(video_dir) if f.endswith('.mp4')],
    key=extract_number
)[:100]
print(video_files)
input()

# Generate a complex filter_complex parameter
filter_str = ""
inputs = ""
for i, video in enumerate(video_files):
    inputs += f" -i {os.path.join(video_dir, video)}"
    filter_str += f"[{i}:v]scale=400:320[v{i}];"  # Uniformly scale to 400x320

# Assume a 10x10 grid layout (100 videos)
layout = ""
for row in range(10):
    for col in range(10):
        index = row * 10 + col
        if index < len(video_files):
            layout += f"[v{index}]"
layout += f"xstack=inputs={len(video_files)}:layout="

# Add layout position information
positions = []
for row in range(10):
    for col in range(10):
        positions.append(f"{col*400}_{row*320}")
layout += "|".join(positions)

filter_str += layout

subprocess.run(
    f"ffmpeg{inputs} -filter_complex \"{filter_str}\" -c:v libx264 -preset fast output.mp4",
    shell=True
)