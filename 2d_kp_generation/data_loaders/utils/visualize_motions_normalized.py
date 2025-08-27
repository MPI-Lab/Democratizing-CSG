import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os

folder = '/home/yangxu/tencent-cospeech/2d_kp_generation_fixed/dataset/miniset_40min/motions_normalized'
output_folder = '/home/yangxu/tencent-cospeech/2d_kp_generation_fixed/dataset/miniset_40min/motions_normalized_vis'
os.makedirs(output_folder, exist_ok=True)
files = sorted(os.listdir(folder))
for file_name in files:
    if file_name.endswith('.npy'):
        file_path = os.path.join(folder, file_name)
        data = np.load(file_path)
        coords = data[:, :, :2]
        coords[:, :, 1] = -coords[:, :, 1]
        confidence = data[:, :, 2]

        fig, ax = plt.subplots()
        scat = ax.scatter([], [])

        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title('motion')

        def update(frame):
            mask = confidence[frame] > 0.6
            scat.set_offsets(coords[frame][mask])
            return scat,

        ani = animation.FuncAnimation(fig, update, frames=range(len(coords)), interval=50, blit=True)
        output_file_path = os.path.join(output_folder, file_name.replace('.npy', '.mp4')) 
        ani.save(output_file_path, writer='ffmpeg')
        print(f"save to {output_file_path}")