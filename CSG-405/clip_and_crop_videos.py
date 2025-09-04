import json
import os
import subprocess
from multiprocessing import Pool
import random
import math
import time
import re

def format_filename(vid, part, start_time, end_time):
    """格式化文件名：{Filename}_{Split}_{Start_Time}_{End_Time}"""
    # 清理时间字符串中的冒号和点
    clean_start = re.sub(r'[:.]', '_', start_time)
    clean_end = re.sub(r'[:.]', '_', end_time)
    
    # 构建文件名
    return f"{vid}_{part}_{clean_start}_{clean_end}"

def process_item(item, output_dir):
    """处理单个视频项的裁剪和截取"""
    vid = item['vid']
    part = item['part']
    start_time = item['start_time']
    end_time = item['end_time']
    crop_box = item['crop_box']  # 假设格式为 [x1, y1, x2, y2]
    
    # 创建输出目录
    output_dir_mp4 = os.path.join(output_dir, 'cropped_mp4')
    output_dir_wav = os.path.join(output_dir, 'cropped_wav')
    os.makedirs(output_dir_mp4, exist_ok=True)
    os.makedirs(output_dir_wav, exist_ok=True)
    
    # 格式化文件名
    base_filename = format_filename(vid, part, start_time, end_time)
    
    # 检查原始视频文件
    mp4_file_path = os.path.join('/home/yangxu/public_tencent_cospeech/Democratizing-CSG/CSG-405/mp4', f'{vid}.mp4')
    if not os.path.exists(mp4_file_path):
        print(f"原始视频 {vid}.mp4 不存在")
        return False
    
    # 检查原始音频文件
    wav_file_path = os.path.join('/home/yangxu/public_tencent_cospeech/Democratizing-CSG/CSG-405/wav', f'{vid}.wav')
    if not os.path.exists(wav_file_path):
        print(f"原始音频 {vid}.wav 不存在")
        return False
    
    # 输出文件路径
    cropped_mp4_path = os.path.join(output_dir_mp4, f'{base_filename}.mp4')
    cropped_wav_path = os.path.join(output_dir_wav, f'{base_filename}.wav')
    
    # 计算裁剪参数
    crop_x1 = crop_box[0]
    crop_y1 = crop_box[1]
    crop_width = crop_box[2] - crop_box[0]
    crop_height = crop_box[3] - crop_box[1]
    
    # 计算缩放和填充参数
    target_size = 512
    aspect_ratio = crop_width / crop_height
    
    if aspect_ratio > 1:  # 宽大于高
        scaled_width = target_size
        scaled_height = int(target_size / aspect_ratio)
        pad_top = (target_size - scaled_height) // 2
        pad_bottom = target_size - scaled_height - pad_top
        pad_left = 0
        pad_right = 0
    else:  # 高大于宽
        scaled_height = target_size
        scaled_width = int(target_size * aspect_ratio)
        pad_left = (target_size - scaled_width) // 2
        pad_right = target_size - scaled_width - pad_left
        pad_top = 0
        pad_bottom = 0
    
    try:
        # 处理视频
        print(f"处理视频 {vid} part {part}: {start_time} - {end_time}")
        
        # 构建视频处理命令
        video_cmd = [
            'ffmpeg',
            '-y',
            '-ss', start_time,
            '-to', end_time,
            '-i', mp4_file_path,
            '-vf', f'crop={crop_width}:{crop_height}:{crop_x1}:{crop_y1},'
                   f'pad={target_size}:{target_size}:{pad_left}:{pad_top}:black,'
                   f'scale={target_size}:{target_size}:flags=lanczos',
            '-c:a', 'copy',
            cropped_mp4_path
        ]
        
        # 执行视频处理
        video_result = subprocess.run(video_cmd, capture_output=True, text=True, timeout=3600)
        if video_result.returncode != 0:
            print(f"视频处理失败: {video_result.stderr}")
            return False
        
        # 处理音频
        print(f"处理音频 {vid} part {part}: {start_time} - {end_time}")
        
        # 构建音频处理命令
        audio_cmd = [
            'ffmpeg',
            '-y',
            '-i', wav_file_path,
            '-ss', start_time,
            '-to', end_time,
            '-c', 'copy',
            cropped_wav_path
        ]
        
        # 执行音频处理
        audio_result = subprocess.run(audio_cmd, capture_output=True, text=True, timeout=600)
        if audio_result.returncode != 0:
            print(f"音频处理失败: {audio_result.stderr}")
            # 删除可能已创建的视频文件
            if os.path.exists(cropped_mp4_path):
                os.remove(cropped_mp4_path)
            return False
        
        # 检查输出文件
        if not os.path.exists(cropped_mp4_path) or os.path.getsize(cropped_mp4_path) == 0:
            print(f"视频输出文件创建失败: {cropped_mp4_path}")
            return False
        
        if not os.path.exists(cropped_wav_path) or os.path.getsize(cropped_wav_path) == 0:
            print(f"音频输出文件创建失败: {cropped_wav_path}")
            # 删除视频文件
            if os.path.exists(cropped_mp4_path):
                os.remove(cropped_mp4_path)
            return False
        
        print(f"成功创建裁剪视频: {cropped_mp4_path}")
        print(f"成功创建裁剪音频: {cropped_wav_path}")
        return True
        
    except subprocess.TimeoutExpired:
        print(f"处理超时: {vid} part {part}")
        return False
    except Exception as e:
        print(f"处理异常: {str(e)}")
        return False

def process_items(video_items, output_dir):
    """处理一批视频项"""
    success_count = 0
    for item in video_items:
        if process_item(item, output_dir):
            success_count += 1
    return success_count

def main():
    # 配置路径
    json_path = '/home/yangxu/public_tencent_cospeech/Democratizing-CSG/CSG-405/all_with_crop.json'
    output_dir = '/home/yangxu/public_tencent_cospeech/Democratizing-CSG/CSG-405'
    
    # 读取JSON文件
    with open(json_path, 'r') as f:
        video_items = json.load(f)
    
    print(f"总共需要处理的视频数量: {len(video_items)}")
    
    # 创建进程池
    num_processes = min(8, os.cpu_count())  # 根据CPU核心数调整
    print(f"使用 {num_processes} 个进程并行处理")
    
    # 打乱顺序以平衡负载
    random.shuffle(video_items)
    
    # 计算每个进程的任务量
    chunk_size = math.ceil(len(video_items) / num_processes)
    chunks = [video_items[i:i+chunk_size] for i in range(0, len(video_items), chunk_size)]
    
    # 使用进程池并行处理
    with Pool(processes=num_processes) as pool:
        results = []
        for i, chunk in enumerate(chunks):
            print(f"启动进程处理 {len(chunk)} 个视频片段")
            results.append(pool.apply_async(process_items, (chunk, output_dir)))
        
        # 等待所有进程完成
        pool.close()
        pool.join()
        
        # 收集结果
        total_success = 0
        for res in results:
            total_success += res.get()
        
        print(f"处理完成! 成功处理 {total_success}/{len(video_items)} 个视频片段")

if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"总耗时: {end_time - start_time:.2f} 秒")