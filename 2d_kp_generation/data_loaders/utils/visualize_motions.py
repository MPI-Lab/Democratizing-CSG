import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

# 设置路径
pose_folder = '/home/yangxu/tencent-cospeech/_data/_datasets/initial-220h/filtered_dataset/pose'
video_folder = '/home/yangxu/tencent-cospeech/_data/_datasets/initial-220h/filtered_dataset/mp4'
output_folder = '/home/yangxu/tencent-cospeech/_data/_datasets/initial-220h/filtered_dataset/motion_mp4'
os.makedirs(output_folder, exist_ok=True)

# COCO Whole Body 骨骼连接定义
skeleton = [
    [15, 13], [13, 11], [16, 14], [14, 12], [11, 12], [5, 11], [6, 12],
    [5, 6], [5, 7], [6, 8], [7, 9], [8, 10], [1, 2], [0, 1], [0, 2],
    [1, 3], [2, 4], [3, 5], [4, 6], [15, 17], [15, 18], [15, 19],
    [16, 20], [16, 21], [16, 22], [91, 92], [92, 93], [93, 94],
    [94, 95], [91, 96], [96, 97], [97, 98], [98, 99], [91, 100],
    [100, 101], [101, 102], [102, 103], [91, 104], [104, 105],
    [105, 106], [106, 107], [91, 108], [108, 109], [109, 110],
    [110, 111], [112, 113], [113, 114], [114, 115], [115, 116],
    [112, 117], [117, 118], [118, 119], [119, 120], [112, 121],
    [121, 122], [122, 123], [123, 124], [112, 125], [125, 126],
    [126, 127], [127, 128], [112, 129], [129, 130], [130, 131],
    [131, 132]
]

# 定义绘制函数
def draw_points(image, points, color=(0, 255, 0), confidence_threshold=0.5):
    for pt in points:
        x, y, conf = pt
        if conf > confidence_threshold:
            cv2.circle(image, (int(x), int(y)), 3, color, -1)
    return image

def draw_skeleton(image, points, skeleton, color=(0, 255, 0), confidence_threshold=0.5):
    for joint in skeleton:
        pt1 = points[joint[0]]
        pt2 = points[joint[1]]
        if pt1[2] > confidence_threshold and pt2[2] > confidence_threshold:
            cv2.line(image, (int(pt1[0]), int(pt1[1])), (int(pt2[0]), int(pt2[1])), color, 2)
    return image

# 加载姿态数据和视频
pose_files = [f for f in os.listdir(pose_folder) if f.endswith('.npy')]

i = 0
for pose_file in pose_files:
    pose_path = os.path.join(pose_folder, pose_file)
    video_filename = pose_file.replace('.npy', '.mp4')
    video_path = os.path.join(video_folder, video_filename)
    
    if not os.path.exists(video_path):
        continue
    
    # 加载姿态数据
    poses = np.load(pose_path)
    
    # 加载视频
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_path = os.path.join(output_folder, video_filename)
    out = cv2.VideoWriter(out_path, fourcc, cap.get(cv2.CAP_PROP_FPS),
                          (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                           int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
    
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx < len(poses):
            pose = poses[frame_idx].squeeze()
            frame = draw_points(frame, pose)
            frame = draw_skeleton(frame, pose, skeleton)
        
        out.write(frame)
        frame_idx += 1
    
    cap.release()
    out.release()
    i = i+1
    if i >= 10:
        break

print("视频处理完成。")