import os
import librosa
import soundfile as sf
import shutil
import numpy as np

# 定义源文件夹路径
audio_dir = '/home/yangxu/tencent-cospeech/2d_kp_generation_fixed/dataset/miniset_40min/audios_cut'
npy_dir = '/home/yangxu/tencent-cospeech/2d_kp_generation_fixed/dataset/miniset_40min/motions_smooth_cut'
video_dir = '/home/yangxu/tencent-cospeech/2d_kp_generation_fixed/dataset/miniset_40min/motions_smooth_cut_vis_20fps'
json_dir = '/home/yangxu/tencent-cospeech/2d_kp_generation_fixed/dataset/miniset_40min/metadata_cut'

# 定义目标文件夹路径
audio_target_dir = audio_dir + '_longer4s'
npy_target_dir = npy_dir + '_longer4s'
video_target_dir = video_dir + '_longer4s'
json_target_dir = json_dir + '_longer4s'

# 创建目标文件夹
os.makedirs(audio_target_dir, exist_ok=True)
os.makedirs(npy_target_dir, exist_ok=True)
os.makedirs(video_target_dir, exist_ok=True)
os.makedirs(json_target_dir, exist_ok=True)

# 获取所有WAV文件名
audio_files = [f for f in os.listdir(audio_dir) if f.endswith('.wav')]

for audio_file in audio_files:
    # 获取音频路径
    audio_path = os.path.join(audio_dir, audio_file)
    
    # 使用librosa读取音频文件并获取时长
    y, sr = librosa.load(audio_path, sr=None)
    duration = librosa.get_duration(y=y, sr=sr)
    
    # 如果时长小于4秒或大于15秒，跳过
    if duration < 4 or duration > 15:
        continue
    
    # 重采样音频到16kHz
    y_resampled = librosa.resample(y, orig_sr=sr, target_sr=16000)
    
    # 生成新的音频文件路径
    audio_target_path = os.path.join(audio_target_dir, audio_file)
    
    # 保存重采样的音频
    sf.write(audio_target_path, y_resampled, 16000)
    
    # 复制对应的npy文件
    npy_file = audio_file.replace('.wav', '.npy')
    npy_path = os.path.join(npy_dir, npy_file)
    if os.path.exists(npy_path):
        shutil.copy(npy_path, os.path.join(npy_target_dir, npy_file))
    
    # 复制对应的mp4文件
    video_file = audio_file.replace('.wav', '.mp4')
    video_path = os.path.join(video_dir, video_file)
    if os.path.exists(video_path):
        shutil.copy(video_path, os.path.join(video_target_dir, video_file))
    
    # 复制对应的json文件
    json_file = audio_file.replace('.wav', '.json')
    json_path = os.path.join(json_dir, json_file)
    if os.path.exists(json_path):
        shutil.copy(json_path, os.path.join(json_target_dir, json_file))
