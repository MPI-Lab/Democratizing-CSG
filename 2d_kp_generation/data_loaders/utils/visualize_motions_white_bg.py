import os
import numpy as np
import cv2

# 设置路径
pose_folder = '/home/yangxu/tencent-cospeech/2d_kp_generation_fixed/dataset/miniset_40min/motions_smooth_cut'
output_folder = '/home/yangxu/tencent-cospeech/2d_kp_generation_fixed/dataset/miniset_40min/motions_smooth_cut_vis_20fps'
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

# 获取所有 .npy 文件
pose_files = [f for f in os.listdir(pose_folder) if f.endswith('.npy')]

for pose_file in pose_files:
    pose_path = os.path.join(pose_folder, pose_file)
    
    # 加载姿态数据
    poses = np.load(pose_path)
    
    # 创建白色背景
    frame_height = 720
    frame_width = 1280
    white_background = np.ones((frame_height, frame_width, 3), dtype=np.uint8) * 255
    
    # 设置视频输出路径和参数
    video_output_path = os.path.join(output_folder, f"{pose_file.replace('.npy', '')}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 20  # 可以根据需求调整帧率
    video = cv2.VideoWriter(video_output_path, fourcc, fps, (frame_width, frame_height))
    
    # 逐帧绘制并写入视频
    for frame_idx, pose in enumerate(poses):
        frame = white_background.copy()
        pose = pose.squeeze()
        frame = draw_points(frame, pose)
        frame = draw_skeleton(frame, pose, skeleton)
        
        video.write(frame)  # 写入视频帧
    
    # 释放视频对象
    video.release()

print("所有姿态视频生成完毕。")
