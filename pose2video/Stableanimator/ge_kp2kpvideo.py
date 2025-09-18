import math
import matplotlib
import cv2
import os
import numpy as np
from DWPose.dwpose_utils.dwpose_detector import dwpose_detector_aligned
import argparse

eps = 0.01

def alpha_blend_color(color, alpha):
    """blend color according to point conf
    """
    return [int(c * alpha) for c in color]

def draw_bodypose(canvas, candidate, subset, score):
    H, W, C = canvas.shape
    candidate = np.array(candidate)
    subset = np.array(subset)

    stickwidth = 4

    limbSeq = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10], \
               [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17], \
               [1, 16], [16, 18], [3, 17], [6, 18]]

    colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

    for i in range(17):
        for n in range(len(subset)):
            index = subset[n][np.array(limbSeq[i]) - 1]
            conf = score[n][np.array(limbSeq[i]) - 1]
            if -1 in index:
                continue
            Y = candidate[index.astype(int), 0] * float(W)
            X = candidate[index.astype(int), 1] * float(H)
            mX = np.mean(X)
            mY = np.mean(Y)
            length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
            polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
            cv2.fillConvexPoly(canvas, polygon, alpha_blend_color(colors[i], conf[0] * conf[1]))

    canvas = (canvas * 0.6).astype(np.uint8)

    for i in range(18):
        for n in range(len(subset)):
            index = int(subset[n][i])
            if index == -1:
                continue
            x, y = candidate[index][0:2]
            conf = score[n][i]
            x = int(x * W)
            y = int(y * H)
            cv2.circle(canvas, (int(x), int(y)), 4, alpha_blend_color(colors[i], conf), thickness=-1)

    return canvas

def draw_handpose(canvas, all_hand_peaks, all_hand_scores):
    H, W, C = canvas.shape

    edges = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [0, 9], [9, 10], \
             [10, 11], [11, 12], [0, 13], [13, 14], [14, 15], [15, 16], [0, 17], [17, 18], [18, 19], [19, 20]]

    for peaks, scores in zip(all_hand_peaks, all_hand_scores):

        for ie, e in enumerate(edges):
            x1, y1 = peaks[e[0]]
            x2, y2 = peaks[e[1]]
            x1 = int(x1 * W)
            y1 = int(y1 * H)
            x2 = int(x2 * W)
            y2 = int(y2 * H)
            score = int(scores[e[0]] * scores[e[1]] * 255)
            if x1 > eps and y1 > eps and x2 > eps and y2 > eps:
                cv2.line(canvas, (x1, y1), (x2, y2),
                         matplotlib.colors.hsv_to_rgb([ie / float(len(edges)), 1.0, 1.0]) * score, thickness=2)

        for i, keyponit in enumerate(peaks):
            x, y = keyponit
            x = int(x * W)
            y = int(y * H)
            score = int(scores[i] * 255)
            if x > eps and y > eps:
                cv2.circle(canvas, (x, y), 4, (0, 0, score), thickness=-1)
    return canvas

def draw_facepose(canvas, all_lmks, all_scores):
    H, W, C = canvas.shape
    for lmks, scores in zip(all_lmks, all_scores):
        for lmk, score in zip(lmks, scores):
            x, y = lmk
            x = int(x * W)
            y = int(y * H)
            conf = int(score * 255)
            if x > eps and y > eps:
                cv2.circle(canvas, (x, y), 3, (conf, conf, conf), thickness=-1)
    return canvas

def draw_pose(pose, H, W, ref_w=2160):
    """vis dwpose outputs

    Args:
        pose (List): DWposeDetector outputs in dwpose_detector.py
        H (int): height
        W (int): width
        ref_w (int, optional) Defaults to 2160.

    Returns:
        np.ndarray: image pixel value in RGB mode
    """
    bodies = pose['bodies']
    faces = pose['faces']
    hands = pose['hands']
    candidate = bodies['candidate']
    subset = bodies['subset']

    sz = min(H, W)
    sr = (ref_w / sz) if sz != ref_w else 1

    ########################################## create zero canvas ##################################################
    canvas = np.zeros(shape=(int(H*sr), int(W*sr), 3), dtype=np.uint8)

    ########################################### draw body pose #####################################################
    canvas = draw_bodypose(canvas, candidate, subset, score=bodies['score'])

    ########################################### draw hand pose #####################################################
    canvas = draw_handpose(canvas, hands, pose['hands_score'])

    ########################################### draw face pose #####################################################
    canvas = draw_facepose(canvas, faces, pose['faces_score'])

    return cv2.cvtColor(cv2.resize(canvas, (W, H)), cv2.COLOR_BGR2RGB).transpose(2, 0, 1)

def get_video_pose(video_path, ref_image_path, poses_folder_path=None):

    ref_image = cv2.imread(ref_image_path)
    ref_image = cv2.cvtColor(ref_image, cv2.COLOR_BGR2RGB)
    height, width, _ = ref_image.shape
    ref_pose = dwpose_detector_aligned(ref_image)
    ref_keypoint_id = [0, 1, 2, 5, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
    ref_keypoint_id = [i for i in ref_keypoint_id \
        if len(ref_pose['bodies']['subset']) > 0 and ref_pose['bodies']['subset'][0][i] >= .0]
    ref_body = ref_pose['bodies']['candidate'][ref_keypoint_id]

    os.makedirs(poses_folder_path, exist_ok=True)
    detected_poses = []
    files = os.listdir(video_path)
    png_files = [f for f in files if f.endswith('.png')]
    png_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
    for sub_name in png_files:
        sub_driven_image_path = os.path.join(video_path, sub_name)
        driven_image = cv2.imread(sub_driven_image_path)
        driven_image = cv2.cvtColor(driven_image, cv2.COLOR_BGR2RGB)
        driven_pose = dwpose_detector_aligned(driven_image)
        detected_poses.append(driven_pose)

    detected_bodies = np.stack(
        [p['bodies']['candidate'] for p in detected_poses if p['bodies']['candidate'].shape[0] == 18])[:,
                      ref_keypoint_id]
    ay, by = np.polyfit(detected_bodies[:, :, 1].flatten(), np.tile(ref_body[:, 1], len(detected_bodies)), 1)
    fh = height
    fw = width
    ax = ay / (fh / fw / height * width)
    bx = np.mean(np.tile(ref_body[:, 0], len(detected_bodies)) - detected_bodies[:, :, 0].flatten() * ax)
    a = np.array([ax, ay])
    b = np.array([bx, by])
    output_pose = []
    # pose rescale
    for detected_pose in detected_poses:
        detected_pose['bodies']['candidate'] = detected_pose['bodies']['candidate'] * a + b
        detected_pose['faces'] = detected_pose['faces'] * a + b
        detected_pose['hands'] = detected_pose['hands'] * a + b
        im = draw_pose(detected_pose, height, width)
        output_pose.append(np.array(im))
    return np.stack(output_pose)

def get_video_pose_given_kp(kp, ref_image_path, poses_folder_path=None):
    # default that kp and ref_image keep same reso
    ref_image = cv2.imread(ref_image_path)
    ref_image = cv2.cvtColor(ref_image, cv2.COLOR_BGR2RGB)
    height, width, _ = ref_image.shape
    ref_pose = dwpose_detector_aligned(ref_image)
    ref_keypoint_id = [0, 1, 2, 5, 8, 11, 14, 15, 16, 17]
    # ref_keypoint_id = [0, 1, 2, 5, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
    ref_keypoint_id = [i for i in ref_keypoint_id \
        if len(ref_pose['bodies']['subset']) > 0 and ref_pose['bodies']['subset'][0][i] >= .0]
    ref_body = ref_pose['bodies']['candidate'][ref_keypoint_id]

    os.makedirs(poses_folder_path, exist_ok=True)
    detected_poses = kp.copy()

    detected_poses_used_align_first_kp = [detected_poses.copy()[0]]

    detected_bodies = np.stack(
        [p['bodies']['candidate'] for p in detected_poses_used_align_first_kp if p['bodies']['candidate'].shape[0] == 18])[:,
                        ref_keypoint_id]
    ay, by = np.polyfit(detected_bodies[:, :, 1].flatten(), np.tile(ref_body[:, 1], len(detected_bodies)), 1)
    fh = height
    fw = width
    ax = ay / (fh / fw / height * width)
    bx = np.mean(np.tile(ref_body[:, 0], len(detected_bodies)) - detected_bodies[:, :, 0].flatten() * ax)
    a = np.array([ax, ay])
    b = np.array([bx, by])
    output_pose = []
    # pose rescale
    for detected_pose in detected_poses:
        detected_pose['bodies']['candidate'] = detected_pose['bodies']['candidate'] * a + b
        detected_pose['faces'] = detected_pose['faces'] * a + b
        detected_pose['hands'] = detected_pose['hands'] * a + b
        im = draw_pose(detected_pose, height, width)
        output_pose.append(np.array(im))
    return np.stack(output_pose)

def get_video_pose_from_kp(kp_path, ref_image_path, poses_folder_path=None, if_smooth=False):
    from utils.smooth_kp import smooth_motion
    import copy
    from utils.preprocess_kp import preprocess_pose, convert_keypoints
    motions_raw = np.load(kp_path, allow_pickle=True)
    motions = preprocess_pose(copy.deepcopy(motions_raw))
    if if_smooth:
        motions = smooth_motion(motions)
    poses_smooth = [convert_keypoints(np.expand_dims(k, axis=0)) for k in motions]
    poses_maps = get_video_pose_given_kp(poses_smooth, ref_image_path, poses_folder_path=poses_folder_path)
    return poses_maps

def get_image_pose(ref_image_path):
    ref_image = cv2.imread(ref_image_path)
    ref_image = cv2.cvtColor(ref_image, cv2.COLOR_BGR2RGB)
    height, width, _ = ref_image.shape
    ref_pose = dwpose_detector_aligned(ref_image)
    pose_img = draw_pose(ref_pose, height, width)
    return np.array(pose_img)

if __name__ == '__main__':
    from tqdm import tqdm
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_path', type=str, default='', help='Path to the folder containing target images.')
    parser.add_argument('--if_smooth', action='store_true', help='If smooth the keypoints.')
    parser.add_argument('--if_gt', action='store_true', help='If the given kp is ground truth. default smooth')
    args = parser.parse_args()
    raw_path = args.raw_path
    for folder in tqdm(sorted(os.listdir(raw_path))):
        if not args.if_gt:
            kp_path = os.path.join(raw_path, folder, 'sample_poses', folder+'.npy')
            if not os.path.exists(kp_path):
                print(f"{folder} does not have kp file, skip")
                continue
            poses_folder_path = os.path.join(raw_path, folder, 'sample_images')
            if os.path.exists(poses_folder_path) and len(os.listdir(poses_folder_path)) > 0:
                print(f"{folder} poses folder is not empty, skip")
                continue
            ref_image_path = os.path.join(raw_path, folder, f'{folder}.png')
            detected_maps = get_video_pose_from_kp(kp_path, ref_image_path, poses_folder_path, if_smooth=args.if_smooth)
            for i in range(detected_maps.shape[0]):
                pose_image = np.transpose(detected_maps[i], (1, 2, 0))
                pose_save_path = os.path.join(poses_folder_path, f"frame_{i}.png")
                cv2.imwrite(pose_save_path, pose_image)
            print(f"{folder} save the pose image in {poses_folder_path}")
        else:
            kp_path = os.path.join(raw_path, folder, 'gt_poses', folder+'.npy')
            if not os.path.exists(kp_path):
                print(f"{folder} does not have kp file, skip")
                continue
            poses_folder_path = os.path.join(raw_path, folder, 'smooth_gt_images')
            if os.path.exists(poses_folder_path) and len(os.listdir(poses_folder_path)) > 0:
                print(f"{folder} poses folder is not empty, skip")
                continue
            ref_image_path = os.path.join(raw_path, folder, f'{folder}.png')
            detected_maps = get_video_pose_from_kp(kp_path, ref_image_path, poses_folder_path, if_smooth=False)
            for i in range(detected_maps.shape[0]):
                pose_image = np.transpose(detected_maps[i], (1, 2, 0))
                pose_save_path = os.path.join(poses_folder_path, f"frame_{i}.png")
                cv2.imwrite(pose_save_path, pose_image)
            print(f"{folder} save the pose image in {poses_folder_path}")