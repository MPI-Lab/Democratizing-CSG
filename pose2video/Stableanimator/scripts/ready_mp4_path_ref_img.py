import os
import argparse

def extract_first_frame(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    for file in os.listdir(input_folder):
        if file.endswith('.mp4'):
            input_file = os.path.join(input_folder, file)
            output_file = os.path.join(output_folder, file.replace('.mp4', '_raw.png'))
            cmd = f'ffmpeg -i {input_file} -vf "select=eq(n\,0)" -q:v 2 {output_file}'
            os.system(cmd)
            cmd_resize = f"ffmpeg -i {output_file} -vf scale=512:512 {output_file.replace('_raw.png', '.png')}"
            os.system(cmd_resize)
            os.system(f'rm {output_file}')

def copy_and_extract_frames(txt_file, original_video_folder, new_video_folder, ref_image_folder):
    os.makedirs(new_video_folder, exist_ok=True)
    os.makedirs(ref_image_folder, exist_ok=True)

    with open(txt_file, 'r') as file:
        for line in file:
            vid1, vid2 = line.strip().split()
            for vid in [vid1, vid2]:
                original_video_path = os.path.join(original_video_folder, f'{vid}.mp4')
                new_video_path = os.path.join(new_video_folder, f'{vid}.mp4')
                if os.path.exists(original_video_path):
                    os.system(f'cp {original_video_path} {new_video_path}')
    
    extract_first_frame(new_video_folder, ref_image_folder)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Copy videos and extract first frames as reference images.")
    parser.add_argument('--dataset_path', type=str, default='', help='Path to the dataset.')
    parser.add_argument('--output_path', type=str, default='', help='Path to save outputs.')

    args = parser.parse_args()
    txt_file = os.path.join(args.dataset_path, 'test.txt')
    original_video_folder = os.path.join(args.dataset_path, 'mp4_cropped')
    new_video_folder = os.path.join(args.output_path, 'mp4_cropped')
    ref_image_folder = os.path.join(args.output_path, 'ref_images_512')
    
    copy_and_extract_frames(txt_file, original_video_folder, new_video_folder, ref_image_folder)
