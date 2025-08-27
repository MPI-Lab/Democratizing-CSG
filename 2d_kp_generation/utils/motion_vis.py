import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os


import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np


def visualize_local_motion_with_audio(motion, audio_path, start_time, title, filename, save_path, skeleton_links, skeleton_links_colors):
    """
    Visualizes local motion data as an animated skeleton, saves it as a video,
    and combines the video with a given audio file using ffmpeg.
    """
    # root_coords = motion[:, 0]
    # coords = motion[:, 1:]
    coords = motion

    fig, ax = plt.subplots(figsize=(12.8, 9.6))
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(f'{title} {filename}')

    # 用红色的点显示主体坐标
    scat = ax.scatter(coords[0][:, 0], coords[0][:, 1], color='red', s=3)
    # # 用黄色的点显示 root_coords
    # root_scat = ax.scatter(root_coords[0, 0], root_coords[0, 1], color='yellow', s=50, label='Root')

    # Pre-create Line2D objects for skeleton
    lines = [ax.plot([], [], color=skeleton_links_colors[idx])[0] for idx in range(len(skeleton_links))]

    # 更新函数，用于动画中的每一帧
    def update_local(frame):
        # 更新主体部分
        scat.set_offsets(coords[frame])
        
        # # 更新 root_coords 显示 (使用黄色点)
        # root_scat.set_offsets(root_coords[frame])  # 更新`root_coords`位置

        # Update lines for the skeleton
        for idx, (pt1_idx, pt2_idx) in enumerate(skeleton_links):
            pt1, pt2 = coords[frame][pt1_idx], coords[frame][pt2_idx]
            if (pt1[0] == 0 and pt1[1] == 0) or (pt2[0] == 0 and pt2[1] == 0):
                lines[idx].set_data([], [])  # Skip invalid joints
            else:
                lines[idx].set_data([pt1[0], pt2[0]], [pt1[1], pt2[1]])  # Update line positions

        # return scat, root_scat  # 返回主体和 root 出现的 Scatter 对象
        return scat, *lines  # 返回主体和 root 出现的 Scatter 对象

    # 创建动画
    ani = animation.FuncAnimation(fig, update_local, frames=range(len(coords)), interval=40, blit=True)

    # 保存到 mp4 文件
    ani.save(save_path, writer='ffmpeg', dpi=100, extra_args=['-loglevel', 'info'])
    plt.close(fig)

    if audio_path is None:
        print(f"Saved video without audio at {save_path}")
        return

    # Combine the generated video with audio using ffmpeg
    output_file_with_audio_path = save_path.replace('.mp4', '_sound.mp4')

    ffmpeg_command = (
        f"ffmpeg -y -ss {start_time} -i {audio_path} -i {save_path} "
        f"-c:v libx264 -c:a aac -strict experimental {output_file_with_audio_path}"
    )
    os.system(ffmpeg_command)

    print(f"Saved video with audio at {output_file_with_audio_path}")




def visualize_motion_with_audio(
    motion, audio_path, start_time, title, filename, save_path, skeleton_links, skeleton_links_colors):
    """
    Visualizes a motion sequence as an animated skeleton, saves it as a video, 
    and combines the video with a given audio file using ffmpeg.
    """

    # Create a figure for the animation
    fig, ax = plt.subplots(figsize=(12.8, 9.6))
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(f'{title} {filename}')

    # Initialize scatter plot for joints
    scat = ax.scatter(motion[0][:, 0], motion[0][:, 1], color='red', s=3)

    # Pre-create Line2D objects for skeleton
    lines = [ax.plot([], [], color=skeleton_links_colors[idx])[0] for idx in range(len(skeleton_links))]

    def update(frame):
        """Update scatter plot and skeleton for each frame."""
        # Update joint positions
        scat.set_offsets(motion[frame])

        # Update lines for the skeleton
        for idx, (pt1_idx, pt2_idx) in enumerate(skeleton_links):
            pt1, pt2 = motion[frame][pt1_idx], motion[frame][pt2_idx]
            if (pt1[0] == 0 and pt1[1] == 0) or (pt2[0] == 0 and pt2[1] == 0):
                lines[idx].set_data([], [])  # Skip invalid joints
            else:
                lines[idx].set_data([pt1[0], pt2[0]], [pt1[1], pt2[1]])  # Update line positions

        return scat, *lines  # Return updated objects
    
    # Create an animation using the update function
    ani = animation.FuncAnimation(fig, update, frames=range(len(motion)), interval=40, blit=True)

    # Save the animation as an MP4 video
    ani.save(save_path, writer='ffmpeg', dpi=100, extra_args=['-loglevel', 'info'])

    plt.close(fig)

    if audio_path is None:
        print(f"Saved video without audio at {save_path}")
        return

    # Combine the generated video with audio using ffmpeg
    output_file_with_audio_path = save_path.replace('.mp4', '_sound.mp4')
    
    ffmpeg_command = (
        f"ffmpeg -y -ss {start_time} -i {audio_path} -i {save_path} "
        f"-c:v libx264 -c:a aac -strict experimental {output_file_with_audio_path}"
    )
    os.system(ffmpeg_command)

    print(f"Saved video with audio at {output_file_with_audio_path}")

def visualize_kp(
    initial_kp, title, filename, save_path, skeleton_links, skeleton_links_colors):
    """
    Visualizes a single motion frame (joint keypoints) as a static skeleton and saves it as a PNG file.
    """
    
    
    # Take the first (and only) motion frame which contains 133 keypoints (with x, y coordinates)
    motion_frame = initial_kp[0]
    
    # Create a figure to visualize
    fig, ax = plt.subplots(figsize=(12.8, 9.6))
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(f'{title} {filename}')
    
    # Scatter plot for the 133 keypoints
    ax.scatter(motion_frame[:, 0], motion_frame[:, 1], color='red', s=3)

    # Draw skeleton
    for idx, (pt1_idx, pt2_idx) in enumerate(skeleton_links):
        pt1, pt2 = motion_frame[pt1_idx], motion_frame[pt2_idx]
        
        # Check for invalid keypoints (assumed to be [0, 0])
        if (pt1[0] == 0 and pt1[1] == 0) or (pt2[0] == 0 and pt2[1] == 0):
            continue  # Skip invalid joints
        
        # Draw line connecting valid joints
        ax.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], color=skeleton_links_colors[idx])

    # Save the plot as a jpg image
    plt.savefig(save_path, dpi=100)
    
    # Close the plot to free memory
    plt.close(fig)
    
    print(f"Saved keypoints visualization as a PNG at {save_path}")
