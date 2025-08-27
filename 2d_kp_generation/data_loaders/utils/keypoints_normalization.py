import os
import numpy as np

# 输入和输出目录
input_dir = '/home/yangxu/tencent-cospeech/2d_kp_generation_fixed/dataset/miniset_40min/motions'
output_dir = '/home/yangxu/tencent-cospeech/2d_kp_generation_fixed/dataset/miniset_40min/motions_normalized'

# 确保输出目录存在
os.makedirs(output_dir, exist_ok=True)

# 获取目录中的所有文件
files = os.listdir(input_dir)

for file_name in files:
    if file_name.endswith('.npy'):
        file_path = os.path.join(input_dir, file_name)
        
        # 加载.npy文件
        data = np.load(file_path)

        # 获取每一帧的鼻子坐标
        nose_coords = data[:, 0, :2]  # 只取xy坐标，不包括置信度

        # 找到 x 和 y 坐标的最大值和最小值
        max_coords = np.max(nose_coords, axis=0)
        min_coords = np.min(nose_coords, axis=0)

        # 计算最大值和最小值的中点
        nose_center = (max_coords + min_coords) / 2

        # 减去中心坐标，获得相对坐标
        relative_coords = data[:, :, :2] - nose_center  # 只对xy坐标进行操作

        # 计算每一帧的两个眼睛的中点和鼻子的距离，得到标准距离
        left_eye_coords = data[:, 1, :2]
        right_eye_coords = data[:, 2, :2]
        eyes_midpoint = (left_eye_coords + right_eye_coords) / 2

        head_distances = np.linalg.norm(eyes_midpoint - nose_coords, axis=1)
        estimated_heights = head_distances * 25
        
        # 计算整个视频的平均身高
        mean_height = np.mean(estimated_heights)

        # 对每一帧的每一个坐标进行归一化
        normalized_coords = relative_coords / mean_height

        # 把每个关键点的y坐标-0.5
        normalized_coords[:, :, 1] -= 0.5

        # 创建最终结果数组，包含归一化的坐标和原始置信度
        final_result = np.zeros_like(data)
        final_result[:, :, :2] = normalized_coords
        final_result[:, :, 2] = data[:, :, 2]  # 保留原始的置信度

        # 保存预处理后的数据到新的.npy文件
        output_file_path = os.path.join(output_dir, file_name)
        np.save(output_file_path, final_result)

        print(f"预处理完成，结果保存到 {output_file_path}")

print("所有文件处理完成。")
