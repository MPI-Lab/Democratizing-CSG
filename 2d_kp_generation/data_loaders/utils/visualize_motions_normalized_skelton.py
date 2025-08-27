import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import argparse

def main():
    folder = args.folder
    files = sorted(os.listdir(folder))

    # 骨架连接点和对应的颜色
    skeleton_links = [
        [15, 13], [13, 11], [16, 14], [14, 12], [11, 12], [5, 11], [6, 12],
        [5, 6], [5, 7], [6, 8], [7, 9], [8, 10], [1, 2], [0, 1], [0, 2],
        [1, 3], [2, 4], [3, 5], [4, 6], [15, 17], [15, 18], [15, 19],
        [16, 20], [16, 21], [16, 22], [91, 92], [92, 93], [93, 94],
        [94, 95], [91, 96], [96, 97], [97, 98], [98, 99], [91, 100],
        [100, 101], [101, 102], [102, 103], [91, 104], [104, 105], [105, 106],
        [106, 107], [91, 108], [108, 109], [109, 110], [110, 111],
        [112, 113], [113, 114], [114, 115], [115, 116], [112, 117],
        [117, 118], [118, 119], [119, 120], [112, 121], [121, 122],
        [122, 123], [123, 124], [112, 125], [125, 126], [126, 127],
        [127, 128], [112, 129], [129, 130], [130, 131], [131, 132]
    ]

    skeleton_link_colors = np.array([
        [0, 255, 0], [0, 255, 0], [255, 128, 0], [255, 128, 0], [51, 153, 255], 
        [51, 153, 255], [51, 153, 255], [51, 153, 255], [0, 255, 0], [255, 128, 0], 
        [0, 255, 0], [255, 128, 0], [51, 153, 255], [51, 153, 255], [51, 153, 255], 
        [51, 153, 255], [51, 153, 255], [51, 153, 255], [51, 153, 255], [0, 255, 0], 
        [0, 255, 0], [0, 255, 0], [255, 128, 0], [255, 128, 0], [255, 128, 0], 
        [255, 128, 0], [255, 128, 0], [255, 128, 0], [255, 128, 0], [255, 153, 255], 
        [255, 153, 255], [255, 153, 255], [255, 153, 255], [102, 178, 255], 
        [102, 178, 255], [102, 178, 255], [102, 178, 255], [255, 51, 51], 
        [255, 51, 51], [255, 51, 51], [255, 51, 51], [0, 255, 0], [0, 255, 0], 
        [0, 255, 0], [0, 255, 0], [255, 128, 0], [255, 128, 0], [255, 128, 0], 
        [255, 128, 0], [255, 153, 255], [255, 153, 255], [255, 153, 255], 
        [255, 153, 255], [102, 178, 255], [102, 178, 255], [102, 178, 255], 
        [102, 178, 255], [255, 51, 51], [255, 51, 51], [255, 51, 51], 
        [255, 51, 51], [0, 255, 0], [0, 255, 0], [0, 255, 0], [0, 255, 0]
    ]) / 255.0  # 将颜色值归一化为 0-1 之间

    def draw_colored_skeleton(ax, points, skeleton_links, colors):
        for idx, joint in enumerate(skeleton_links):
            pt1 = points[joint[0]]
            pt2 = points[joint[1]]
            ax.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], color=colors[idx])

    for file_name in files:
        if file_name.endswith('.npy'):
            file_path = os.path.join(folder, file_name)
            datas = np.load(file_path, allow_pickle=True)
            datas = datas.item()["motions"]
            
            for i in range(len(datas)):
                data = datas[i]
                data = data.transpose(2, 0, 1)
                coords = data[:, :, :2]
                coords[:, :, 1] = -coords[:, :, 1]

                fig, ax = plt.subplots(figsize=(12.8, 9.6))
                ax.set_xlim(-1, 1)
                ax.set_ylim(-1, 1)
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_title('Motion Visualization')

                # 绘制骨架和散点，设置点的大小为10
                scat = ax.scatter(coords[0][:, 0], coords[0][:, 1], color='red', s=3)

                def update(frame):
                    scat.set_offsets(coords[frame])
                    ax.lines.clear()  # 清除之前的线条
                    draw_colored_skeleton(ax, coords[frame], skeleton_links, skeleton_link_colors)
                    return scat,

                ani = animation.FuncAnimation(fig, update, frames=range(len(coords)), interval=50, blit=True)
                output_file_path = os.path.join(folder, file_name.replace('.npy', f'_{i}.mp4'))
                ani.save(output_file_path, writer='ffmpeg',dpi=100)
                print(f"Saved to {output_file_path}")
                plt.close(fig)
if __name__ == "__main__":
    # folder = '/home/yangxu/tencent-cospeech/2d_kp_generation_fixed/save/miniset_40min_train_uncondition_w_z_score/sample_seed_0_20240814_145921_checkpoint_160000'
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, required=True)
    args = parser.parse_args()
    main()
    # python /home/yangxu/tencent-cospeech/2d_kp_generation_fixed/sample/visualize_motions_normalized_skelton.py --folder '/home/yangxu/tencent-cospeech/2d_kp_generation_fixed/save/miniset_40min_train_uncondition_w_z_score/sample_seed_0_20240814_145921_checkpoint_160000'
    # python /home/yangxu/tencent-cospeech/2d_kp_generation_fixed/sample/visualize_motions_normalized_skelton.py --folder '/home/yangxu/tencent-cospeech/2d_kp_generation_fixed/save/miniset_40min_train_uncondition_wo_z_score/sample_seed_0_20240814_145850_checkpoint_200000'