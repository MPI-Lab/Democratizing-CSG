import json
import os

# 路径配置
main_json_path = "/home/yangxu/public_tencent_cospeech/Democratizing-CSG/CSG-405/all.json"
meta_dir = "/home/yangxu/public_tencent_cospeech/tencent-cospeech/_data/dataset/filtered_400h_2_13/meta"
output_json_path = "/home/yangxu/public_tencent_cospeech/Democratizing-CSG/CSG-405/all_with_crop.json"

# 加载主 JSON 文件
with open(main_json_path, 'r') as f:
    data = json.load(f)
i = 0
# 处理每个视频片段
for item in data:
    print(f"正在处理第{i}个文件")
    i += 1
    vid = item['vid']
    part = item['part']
    start_time = item['start_time']
    end_time = item['end_time']
    
    # 构建元数据文件名
    meta_filename = f"{vid}_{part}.json"
    meta_path = os.path.join(meta_dir, meta_filename)
    
    # 检查元数据文件是否存在
    if not os.path.exists(meta_path):
        print(f"警告: 元数据文件 {meta_filename} 不存在")
        item['crop_box'] = []  # 添加空 crop_box
        continue
    
    # 加载元数据文件
    with open(meta_path, 'r') as f:
        meta_data = json.load(f)
    
    # 构建时间范围键
    time_key = f"{start_time} - {end_time}"
    
    # 在 scene_detection 中查找匹配的时间段
    scene_detection = meta_data.get('scene_detection', {})
    if time_key in scene_detection:
        # 提取 crop_box
        crop_box = scene_detection[time_key].get('crop_box', [])
        item['crop_box'] = crop_box
    else:
        # 尝试查找包含当前时间段的范围
        found = False
        for key, scene_info in scene_detection.items():
            scene_start, scene_end = key.split(" - ")
            if scene_start <= start_time and scene_end >= end_time:
                crop_box = scene_info.get('crop_box', [])
                item['crop_box'] = crop_box
                found = True
                break
        
        if not found:
            print(f"警告: 在 {meta_filename} 中未找到匹配的时间段 {time_key}")
            item['crop_box'] = []  # 添加空 crop_box

# 保存更新后的 JSON 文件
with open(output_json_path, 'w') as f:
    json.dump(data, f, indent=2)

print(f"处理完成! 结果已保存到 {output_json_path}")
print(f"处理了 {len(data)} 个视频片段")