import cv2
import os

# 视频文件路径
video_path = r"D:\try\try\szmCode\paconv_\huaxi_init.mp4"
output_dir = r"D:\try\try\szmCode\paconv_\frames"  # 输出图片保存目录

# 创建保存帧的目录
os.makedirs(output_dir, exist_ok=True)

# 打开视频文件
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Cannot open video file.")
    exit()

frame_count = 0
target_size = (1280, 1024)  # 目标分辨率

# 遍历视频的每一帧
while True:
    ret, frame = cap.read()  # 读取一帧
    if not ret:
        break  # 如果没有更多帧，退出循环

    # 插值调整分辨率
    resized_frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_LINEAR)

    # 保存调整后的帧为图片文件
    frame_filename = os.path.join(output_dir, f"frame_{frame_count:04d}.png")
    cv2.imwrite(frame_filename, resized_frame)
    print(f"Saved: {frame_filename}")
    frame_count += 1

cap.release()
print(f"Video processing complete. {frame_count} frames saved.")