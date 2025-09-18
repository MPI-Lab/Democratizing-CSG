import math
import argparse
import cv2
import os
from tqdm import tqdm
import decord
import numpy as np
import matplotlib

from dwpose_utils.dwpose_detector import dwpose_detector_aligned

eps = 0.01

def alpha_blend_color(color, alpha):
    """blend color according to point conf
    """
    return [int(c * alpha) for c in color]

def draw_bodypose_aligned(canvas, candidate, subset, score):
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
            # if conf[0] < 0.3 or conf[1] < 0.3:
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

def draw_handpose_aligned(canvas, all_hand_peaks, all_hand_scores):
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

def draw_facepose_aligned(canvas, all_lmks, all_scores):
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

def draw_pose_aligned(pose, H, W, ref_w=2160):
    bodies = pose['bodies']
    faces = pose['faces']
    hands = pose['hands']
    candidate = bodies['candidate']
    subset = bodies['subset']
    sz = min(H, W)
    sr = (ref_w / sz) if sz != ref_w else 1
    canvas = np.zeros(shape=(int(H*sr), int(W*sr), 3), dtype=np.uint8)
    canvas = draw_bodypose_aligned(canvas, candidate, subset, score=bodies['score'])
    canvas = draw_handpose_aligned(canvas, hands, pose['hands_score'])
    canvas = draw_facepose_aligned(canvas, faces, pose['faces_score'])

    return cv2.cvtColor(cv2.resize(canvas, (W, H)), cv2.COLOR_BGR2RGB).transpose(2, 0, 1)


def get_image_pose(ref_image_path):
    ref_image = cv2.imread(ref_image_path)
    ref_image = cv2.cvtColor(ref_image, cv2.COLOR_BGR2RGB)
    height, width, _ = ref_image.shape
    ref_pose = dwpose_detector_aligned(ref_image)
    pose_img = draw_pose_aligned(ref_pose, height, width)
    return np.array(pose_img)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Training Skeleton Poses Extraction", add_help=True)
    parser.add_argument("--start", type=int, help="Specify the value of start")
    parser.add_argument("--end", type=int, help="Specify the value of end")
    parser.add_argument("--name", type=str, help="Specify the name of dataset")
    parser.add_argument("--root_path", type=str, help="Specify the root path of dataset")
    args = parser.parse_args()

    start = args.start
    end = args.end
    dataset_name = args.name

    image_root = os.path.join(args.root_path, dataset_name)
    for idx in range(start, end+1):
        subfolder = str(idx).zfill(5)
        subfolder_path = os.path.join(image_root, subfolder)
        images_subfolder_path = os.path.join(subfolder_path, "images")
        print(f"images subfolder path: {images_subfolder_path}")

        pose_subfolder_path = os.path.join(subfolder_path, "poses")
        if not os.path.exists(pose_subfolder_path):
            os.makedirs(pose_subfolder_path)
            print(f"Folder created: {pose_subfolder_path}")
        else:
            print(f"Folder already exists: {pose_subfolder_path}")
        for root, dirs, files in os.walk(images_subfolder_path):
            for file in files:
                if file.endswith('.png'):
                    file_path = os.path.join(root, file)
                    print(file_path)
                    file_name = os.path.splitext(file)[0]
                    image_name = file_name + '.png'
                    image_legal_path = os.path.join(images_subfolder_path, image_name)
                    if os.path.exists(os.path.join(pose_subfolder_path, file_name + '.png')):
                        existed_path = os.path.join(pose_subfolder_path, file_name + '.png')
                        print(f"{existed_path} already exists!")
                        continue
                    detected_map = get_image_pose(image_legal_path)
                    detected_map = np.transpose(detected_map, (1, 2, 0))
                    pose_save_path = os.path.join(pose_subfolder_path, file_name + '.png')
                    cv2.imwrite(pose_save_path, detected_map)
                    print(f"Finish Pose Extraction: {pose_save_path}")
