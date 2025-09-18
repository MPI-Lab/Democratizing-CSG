import numpy as np

def vis_data(data, save_path, xmin=-1, ymin=-1, xmax=1, ymax=1, if_ge=False):
    data = data.copy()
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    import os
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

    def draw_colored_skeleton(ax, points, skeleton_links, colors, mask):
        for idx, joint in enumerate(skeleton_links):
            pt1 = points[joint[0]]
            pt2 = points[joint[1]]
            if mask[joint[0]] and mask[joint[1]]:
                ax.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], color=colors[idx])
    output_path = save_path
    # output_path = os.path.join(save_path, 'vis.mp4')
    # os.makedirs(save_path, exist_ok=True)

    coords = data[:, :, :2]

    # print(coords.shape)
    # ## confidence的shape置为(coords.shape[0], coords.shape[1],1),数值全为1
    # confidence = np.ones((coords.shape[0], coords.shape[1]))
    confidence = data[:, :, 2]
    # print(confidence.shape)
    fig, ax = plt.subplots(figsize=(5.76,5.76))
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Motion Visualization')

    # 绘制骨架和散点，设置点的大小为10
    scat = ax.scatter(coords[0][:, 0], coords[0][:, 1], color='red', s=3)

    def update(frame):
        mask = confidence[frame] > 0.6
        scat.set_offsets(coords[frame][mask])
        # 清除之前的线条
        for line in ax.lines:
            line.remove()  # 移除每一条线
        # ax.lines.clear()
        draw_colored_skeleton(ax, coords[frame], skeleton_links, skeleton_link_colors, mask)
        return scat,

    ani = animation.FuncAnimation(fig, update, frames=range(len(coords)), interval=40, blit=True)

    ani.save(output_path, writer='ffmpeg',dpi=100)
    print(f"Saved to {output_path}")
    plt.close(fig)

def convert_keypoints(keypoints):    
    keypoints_info = keypoints.copy()
    # compute neck joint
    neck = np.mean(keypoints_info[:, [5, 6]], axis=1)
    # neck score when visualizing pred
    neck[:, 2:4] = np.logical_and(
        keypoints_info[:, 5, 2:4] > 0.3,
        keypoints_info[:, 6, 2:4] > 0.3).astype(int)
    new_keypoints_info = np.insert(
        keypoints_info, 17, neck, axis=1)
    mmpose_idx = [
        17, 6, 8, 10, 7, 9, 12, 14, 16, 13, 15, 2, 1, 4, 3
    ]
    openpose_idx = [
        1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17
    ]
    new_keypoints_info[:, openpose_idx] = \
        new_keypoints_info[:, mmpose_idx]
    keypoints_info = new_keypoints_info

    candidate, score = keypoints_info[
        ..., :2], keypoints_info[..., 2]
    
    nums, _, locs = candidate.shape
    body = candidate[:, :18].copy()
    body = body.reshape(nums * 18, locs)
    subset = score[:, :18].copy()
    for i in range(len(subset)):
        for j in range(len(subset[i])):
            if subset[i][j] > 0.6:
                subset[i][j] = int(18 * i + j)
            else:
                subset[i][j] = -1

    faces = candidate[:, 24:92]

    hands = candidate[:, 92:113]
    hands = np.vstack([hands, candidate[:, 113:]])

    faces_score = score[:, 24:92]
    hands_score = np.vstack([score[:, 92:113], score[:, 113:]])

    bodies = dict(candidate=body, subset=subset, score=score[:, :18])
    pose = dict(bodies=bodies, hands=hands, hands_score=hands_score, faces=faces, faces_score=faces_score)

    return pose


def preprocess_pose(keypoints, mask=None):
    '''
    Given a keypoints_info, which shape is (frames, 133, 2), 
    First, add 1 to its last two dimensions,
    Second, add 1 to its last dimension and its shape becomes (frames, 133,3).
    '''
    keypoints_info = keypoints.copy()
    # frames, keypoints_num, _ = keypoints_info.shape
    # kps = keypoints_info.reshape(frames * keypoints_num, 2)
    # center_x = np.mean(kps[:, 0])
    # center_y = np.mean(kps[:, 1])

    # # 计算偏移量
    # offset_x = 0.0 - center_x
    # offset_y = 0.0 - center_y

    # # 将偏移量应用于所有关节点坐标
    # kps[:, 0] += offset_x
    # kps[:, 1] += offset_y

    # max_xy = np.max(np.abs(kps), axis=0)
    # rescale = 1.0 / np.max(max_xy)

    # # keypoints_info[..., :2] *= rescale
    
    keypoints_info = (keypoints_info + 1)/2
    if keypoints_info.shape[1] == 123:
        # 在第一维上的13:23位置都插入0.5，0.5
        inser_kp = np.zeros((keypoints_info.shape[0], 10, 2)) + 0.5
        keypoints_info = np.concatenate((keypoints_info[:, :13], inser_kp, keypoints_info[:, 13:]), axis=1)
    # keypoints_info = affine_transform_2d_keypoints(keypoints_info)
    if mask is None:
        keypoints_info = np.squeeze(keypoints_info)
        keypoints_info = np.concatenate(
            (keypoints_info, np.ones((keypoints_info.shape[0], keypoints_info.shape[1], 1))), axis=-1)
        # low body keypoints confidence set to 0.0
        keypoints_info[:, 13:23, 2] = 0.0
        # if keypoints_info[...,[]0,1]==[0,0], set keypoints_info[...,2]=0.0
        mask = (keypoints_info[...,0]==0.5) & (keypoints_info[...,1]==0.5)
        keypoints_info[...,2] = np.where(mask, 0.0, keypoints_info[...,2])
    else:
        assert mask.shape == keypoints_info.shape
        keypoints_info = np.concatenate((keypoints_info, mask[...,[0]]), axis=-1)
        # keypoints_info[...,0]= keypoints_info[...,0]/(1280/720)
    ############debug################
    # import sys
    # sys.path.append('/home/shenbo/projects/co-seech-ges-projects/MimicMotion')
    # from utils.visualize_normalize import vis_data
    # vis_data(keypoints_info, 'output_debug/vis_ge_data', xmin=0, ymin=0, xmax=1, ymax=1)

    # if_ge = False
    # keypoints_info_vis = keypoints_info.copy()
    # if if_ge:
    #     pass
    # else:
    #     keypoints_info_vis[:, :, 1] = 1-keypoints_info_vis[:, :, 1]
    # vis_data(keypoints_info_vis, '/home/shenbo/shenbo/projects/Pose2video/StableAnimator/dataset/test_long_unpaired_YqnhRPq2Hq0_0_00:05:51.920_00:06:04.717_gHlQOxsYoKw_0_00:32:58.031_00:33:06.992/pose_raw.mp4', xmin=0, ymin=0, xmax=1, ymax=1)
    

    ############debug################

    return keypoints_info