# 读取/home/yangxu/tencent-cospeech/_data/2d_kp_generation_fixed/dataset/miniset_40min/motions_smooth中每一个npy文件，npy文件名字的规律是_QrZvwNE510_3788_3921，也就是vid：_QrZvwNE510，起始帧：3788，结束帧：3921，内容是frames,joints,feature。将motions_smooth中所有的frames去掉前5帧和后5帧，并保存在/home/yangxu/tencent-cospeech/_data/2d_kp_generation_fixed/dataset/miniset_40min/motions_smooth_cut中，同时名字的帧数部分有相应的修改

# 同时，/home/yangxu/tencent-cospeech/_data/2d_kp_generation_fixed/dataset/miniset_40min/metadata.json中的内容是{
# "2ShZKR5Uo2I_2404_2710": {
# "name": "2ShZKR5Uo2I.json",
# "fps": 24
# },
# "2ShZKR5Uo2I_2710_2775": {
# "name": "2ShZKR5Uo2I.json",
# "fps": 24
# },
# "2ShZKR5Uo2I_2946_3091": {
# "name": "2ShZKR5Uo2I.json",
# "fps": 24
# },
# "2ShZKR5Uo2I_3317_3506": {
# "name": "2ShZKR5Uo2I.json",
# "fps": 24
# },
# 从中找到npy相应的fps，把帧数变化变成时间变化（精确到ms），将/home/yangxu/tencent-cospeech/_data/2d_kp_generation_fixed/dataset/miniset_40min/audios下的wav也做相应的修改，保存在audios_cut，同时新建一个meta文件，其中的名字的帧数部分也修改成变化后的
# 用ffmpeg来进行音频的cut
import os
import numpy as np
import json
import subprocess

# 路径设置
motions_smooth_dir = '/home/yangxu/tencent-cospeech/_data/2d_kp_generation_fixed/dataset/miniset_40min/motions_smooth'
motions_smooth_cut_dir = '/home/yangxu/tencent-cospeech/_data/2d_kp_generation_fixed/dataset/miniset_40min/motions_smooth_cut'
audios_dir = '/home/yangxu/tencent-cospeech/_data/2d_kp_generation_fixed/dataset/miniset_40min/audios'
audios_cut_dir = '/home/yangxu/tencent-cospeech/_data/2d_kp_generation_fixed/dataset/miniset_40min/audios_cut'
metadata_path = '/home/yangxu/tencent-cospeech/_data/2d_kp_generation_fixed/dataset/miniset_40min/metadata.json'
meta_output_path = '/home/yangxu/tencent-cospeech/_data/2d_kp_generation_fixed/dataset/miniset_40min/metadata_cut.json'

# 创建目标目录
os.makedirs(motions_smooth_cut_dir, exist_ok=True)
os.makedirs(audios_cut_dir, exist_ok=True)

# 加载metadata.json
with open(metadata_path, 'r') as f:
    metadata = json.load(f)

# 新的meta文件字典
new_meta = {}

# 处理每个npy文件
for npy_file in os.listdir(motions_smooth_dir):
    if npy_file.endswith('.npy'):
        # 解析文件名获取视频ID和帧数信息
        base_name = npy_file.replace('.npy', '')
        parts = base_name.split('_')
        vid = '_'.join(parts[:-2])  # 将除最后两个元素外的部分拼接成vid
        start_frame = parts[-2]     # 倒数第二个元素作为start_frame
        end_frame = parts[-1]       # 最后一个元素作为end_frame

        start_frame, end_frame = int(start_frame), int(end_frame)

        # 加载npy文件
        npy_path = os.path.join(motions_smooth_dir, npy_file)
        data = np.load(npy_path)

        # 去除前10帧和后10帧
        if data.shape[0] > 20:  # 确保至少有20帧
            data_cut = data[10:-10]

            # 修改帧数
            new_start_frame = start_frame + 10
            new_end_frame = end_frame - 10

            # 保存新的npy文件
            new_npy_name = f"{vid}_{new_start_frame}_{new_end_frame}.npy"
            new_npy_path = os.path.join(motions_smooth_cut_dir, new_npy_name)
            np.save(new_npy_path, data_cut)



            # 裁剪音频文件使用ffmpeg
            audio_file = f"{base_name}.wav"
            audio_path = os.path.join(audios_dir, audio_file)
            new_audio_path = os.path.join(audios_cut_dir, f"{vid}_{new_start_frame}_{new_end_frame}.wav")

            # 获取fps
            fps = metadata[base_name]['fps']
            # turn to ms
            start_time_s = 10 / fps
            end_time_s = (data.shape[0] - 10) / fps


            ffmpeg_command = [
                'ffmpeg',
                '-y',
                '-i', audio_path,
                '-ss', f"{start_time_s:.3f}",
                '-to', f"{end_time_s:.3f}",
                '-c', 'copy',
                new_audio_path
            ]

            subprocess.run(ffmpeg_command, check=True)

            # 更新新的meta信息
            new_meta[f"{vid}_{new_start_frame}_{new_end_frame}"] = {
                "name": f"{vid}.json",
                "fps": fps,
            }

# 保存新的meta文件
with open(meta_output_path, 'w') as f:
    json.dump(new_meta, f, indent=4)

print("处理完成。")
