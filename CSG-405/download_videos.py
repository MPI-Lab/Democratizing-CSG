import json
import subprocess
import os
from multiprocessing import Process, current_process
import random
import time
import filelock

def process_videos(video_list, output_dir, json_path):
    """处理每个子进程的下载和切分工作"""
    # 创建目录
    output_dir_mp4 = os.path.join(output_dir, 'mp4')
    output_dir_wav = os.path.join(output_dir, 'wav')
    os.makedirs(output_dir_mp4, exist_ok=True)
    os.makedirs(output_dir_wav, exist_ok=True)
    
    downloaded_ids = []
    
    for video_id in video_list:
        # 检查是否已处理过（任意片段存在即跳过）
        if os.path.exists(f'{output_dir_mp4}/{video_id}_0.mp4'):
            print(f"视频 {video_id} 已存在，跳过处理")
            continue
            
        video_url = f'https://www.youtube.com/watch?v={video_id}'
        raw_mp4_path = f'{output_dir_mp4}/{video_id}.mp4'
        
        print(f"下载 {video_id}.mp4")
        result = subprocess.run([
            'yt-dlp',
            '-f', 'bestvideo[width=1280][height=720][ext=mp4]+bestaudio[ext=m4a][language^=en]',
            '--merge-output-format', 'mp4',
            '--postprocessor-args', 'ffmpeg: -c:v h264_nvenc -c:a copy -filter:v fps=25',
            '-o', raw_mp4_path,
            video_url], capture_output=True, text=True)

        if result.returncode != 0:
            print(f"下载失败: {result.stderr}")
            # 清理可能存在的部分下载文件
            if os.path.exists(raw_mp4_path):
                os.remove(raw_mp4_path)
            continue

        def get_duration(file_path):
            result = subprocess.run(
                ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', file_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT
            )
            return float(result.stdout)

        try:
            video_duration = get_duration(raw_mp4_path)
        except Exception as e:
            print(f"获取时长失败: {str(e)}")
            os.remove(raw_mp4_path)
            continue

        segment_duration = 3600  # 1小时
        num_segments = int(video_duration // segment_duration)

        # 处理视频片段
        segments_created = 0
        for i in range(num_segments + 1):
            # 跳过最后不足120秒的片段
            if i == num_segments and (video_duration % segment_duration) < 120:
                break

            start_time = i * segment_duration
            duration = segment_duration if (i < num_segments) else (video_duration % segment_duration)
            end_time = start_time + duration
            
            video_segment = f'{output_dir_mp4}/{video_id}_{i}.mp4'
            audio_segment = f'{output_dir_wav}/{video_id}_{i}.wav'
            
            # 提取视频片段
            subprocess.run([
                'ffmpeg', '-ss', str(start_time), '-to', str(end_time), 
                '-i', raw_mp4_path, '-c', 'copy', video_segment
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            # 提取音频片段
            subprocess.run([
                'ffmpeg', '-i', video_segment, 
                '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1',
                audio_segment
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            segments_created += 1
            print(f"创建片段: {video_id}_{i} ({duration:.1f}s)")
        
        # 删除原始文件
        os.remove(raw_mp4_path)
        print(f"完成处理 {video_id}, 创建 {segments_created} 个片段")
        downloaded_ids.append(video_id)
    
    # 更新JSON文件
    if downloaded_ids:
        # 使用文件锁确保安全写入
        lock_path = json_path + ".lock"
        lock = filelock.FileLock(lock_path)
        
        try:
            with lock:
                # 读取原始JSON
                with open(json_path, 'r') as f:
                    data = json.load(f)
                
                # 创建视频ID到索引的映射
                vid_to_index = {item['vid']: idx for idx, item in enumerate(data)}
                
                # 更新下载状态
                for vid in downloaded_ids:
                    if vid in vid_to_index:
                        data[vid_to_index[vid]]['if_download'] = True
                
                # 写回文件
                with open(json_path, 'w') as f:
                    json.dump(data, f, indent=2)
                
                print(f"进程 {current_process().name} 更新了 {len(downloaded_ids)} 个视频的下载状态")
        except filelock.Timeout:
            print(f"进程 {current_process().name} 获取文件锁超时")
        except Exception as e:
            print(f"更新JSON文件失败: {str(e)}")
    
    print(f"进程 {current_process().name} 完成")

def split_and_run(video_ids, num_splits, output_dir, json_path):
    """将列表拆分并在新进程中运行"""
    random.shuffle(video_ids)  # 打乱顺序均衡负载
    chunk_size = len(video_ids) // num_splits
    processes = []
    
    for i in range(num_splits):
        chunk_start = i * chunk_size
        chunk_end = (i + 1) * chunk_size if i < num_splits - 1 else len(video_ids)
        video_chunk = video_ids[chunk_start:chunk_end]
        
        p = Process(target=process_videos, args=(video_chunk, output_dir, json_path))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

# 主程序
if __name__ == "__main__":
    # 配置路径
    json_path = '/home/yangxu/public_tencent_cospeech/Democratizing-CSG/CSG-405/all.json'
    output_dir = '.'  # 当前目录
    
    # 从JSON读取视频ID
    with open(json_path, 'r') as f:
        data = json.load(f)
        video_ids = [item['vid'] for item in data]
    
    print(f"总共需要处理的视频数量: {len(video_ids)}")
    
    # 分进程处理
    num_splits = 32
    split_and_run(video_ids, num_splits, output_dir, json_path)

    print("所有进程完成！")