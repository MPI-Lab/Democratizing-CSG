import os
import warnings
import numpy as np
import scipy.signal as signal
import torch

class SGFilter:
    """savgol_filter lib is from:
    https://docs.scipy.org/doc/scipy/reference/generated/
    scipy.signal.savgol_filter.html.

    Args:
        window_size (float):
                    The length of the filter window
                    (i.e., the number of coefficients).
                    window_length must be a positive odd integer.
        polyorder (int):
                    The order of the polynomial used to fit the samples.
                    polyorder must be less than window_length.

    Returns:
        smoothed poses (np.ndarray, torch.tensor)
    """

    def __init__(self, window_size=11, polyorder=2):
        super(SGFilter, self).__init__()

        # 1-D Savitzky-Golay filter
        self.window_size = window_size
        self.polyorder = polyorder

    def __call__(self, x=None):
        # x.shape: [t,k,c]
        if self.window_size % 2 == 0:
            window_size = self.window_size - 1
        else:
            window_size = self.window_size
        if window_size > x.shape[0]:
            window_size = x.shape[0]
        if window_size <= self.polyorder:
            polyorder = window_size - 1
        else:
            polyorder = self.polyorder
        assert polyorder > 0
        assert window_size > polyorder
        if len(x.shape) != 3:
            warnings.warn('x should be a tensor or numpy of [T*M,K,C]')
        assert len(x.shape) == 3
        x_type = x
        if isinstance(x, torch.Tensor):
            if x.is_cuda:
                x = x.cpu().numpy()
            else:
                x = x.numpy()
        smooth_poses = np.zeros_like(x)
        # smooth at different axis
        C = x.shape[-1]
        for i in range(C):
            smooth_poses[..., i] = signal.savgol_filter(
                x[..., i], window_size, polyorder, axis=0)

        if isinstance(x_type, torch.Tensor):
            # we also return tensor by default
            if x_type.is_cuda:
                smooth_poses = torch.from_numpy(smooth_poses).cuda()
            else:
                smooth_poses = torch.from_numpy(smooth_poses)
        return smooth_poses

def smooth_kp(x,
            smooth_type='savgol',
            window_size=11, 
            polyorder=2
            ):
    """Smooth the array with the specified smoothing type.

    Args:
        x (np.ndarray): Shape should be (frame,K,C).
        smooth_type (str, optional): support savgol.  Defaults to 'savgol'.
        window_size (int, optional): The length of the filter window.  Defaults to 11.
        polyorder (int, optional): The order of the polynomial used to fit the samples.  Defaults to 2.

    Returns:
        np.ndarray: Smoothed data. The shape should be
            (frame,K,C).
    """

    assert smooth_type in ['savgol']

    x = x.copy()
    assert x.ndim == 3

    smooth_func = SGFilter(window_size=window_size, polyorder=polyorder)

    smooth_x = smooth_func(x[...,:2])
    x = np.concatenate([smooth_x,x[...,2:]],axis=-1)

    return x

def smooth_motion_wo_face(motion_data):
    motion_data = motion_data.copy()
    data_to_smooth = np.concatenate([motion_data[:, :23, :], motion_data[:, 91:, :]], axis=1)
    smoothed_data = smooth_kp(data_to_smooth, window_size=11, polyorder=2)
    smoothed_data = np.concatenate(
        [smoothed_data[:, :23, :], motion_data[:, 23:91, :], smoothed_data[:, 23:, :]], axis=1)
    return smoothed_data

def smooth_motion(motion_data):
    # Split the data into parts to smooth and parts to keep
    data_to_smooth = np.concatenate([motion_data[:, :71, :], motion_data[:, 91:, :]], axis=1)
    mouth = motion_data[:, 71:91, :]

    inner_mouth = mouth[:, 12:, :]
    outer_mouth = mouth[:, :12, :]

    # smooth the outer mouth
    smoothed_outer_mouth = smooth_kp(outer_mouth, window_size=7, polyorder=2)

    # smooth the inner mouth
    smoothed_inner_mouth = smooth_kp(inner_mouth, window_size=5, polyorder=2)

    # Smooth the combined parts
    smoothed_data = smooth_kp(data_to_smooth, window_size=11, polyorder=2)

    # Concatenate the smoothed and unsmoothed parts
    smoothed_data = np.concatenate(
        [smoothed_data[:, :71, :], smoothed_outer_mouth, smoothed_inner_mouth, smoothed_data[:, 71:, :]], axis=1
    )
    return smoothed_data

def smooth_face(face_data):
    data_to_smooth = face_data[:, :48, :]
    mouth = face_data[:, 48:68, :]
    inner_mouth = mouth[:, 12:, :]
    outer_mouth = mouth[:, :12, :]
    # smooth the outer mouth
    smoothed_outer_mouth = smooth_kp(outer_mouth, window_size=7, polyorder=2)
    # smooth the inner mouth
    smoothed_inner_mouth = smooth_kp(inner_mouth, window_size=5, polyorder=2)

    # Smooth the combined parts
    smoothed_data = smooth_kp(data_to_smooth, window_size=11, polyorder=2)
    smoothed_data = np.concatenate(
        [smoothed_data[:, :48, :], smoothed_outer_mouth, smoothed_inner_mouth, smoothed_data[:, 48:, :]], axis=1
    )
    return smoothed_data

def smooth_motion_gt(motion_data):
    motion_data = motion_data.copy()
    bodies = []
    hands = []
    faces = []
    for motion in motion_data:
        bodies.append(motion['bodies']['candidate'])
        hands.append(motion['hands'].reshape(42,2))
        faces.append(motion['faces'])
    bodies_cat = np.concatenate(np.expand_dims(bodies, axis=0), axis=0) # (frames, 18, 2)
    hands_cat = np.concatenate(np.expand_dims(hands, axis=0), axis=0) # (frames, 42, 2)
    faces_cat = np.concatenate(faces, axis=0) # (frames, 68, 2)
    # bodies_cat = np.concatenate(bodies, axis=0) # (frames, 18, 2)
    # hands_cat = np.concatenate(hands, axis=0) # (frames, 42, 2)
    # faces_cat = np.concatenate(faces, axis=0)
    bodies_smooth = smooth_kp(bodies_cat, window_size=11, polyorder=2)
    hands_smooth = smooth_kp(hands_cat, window_size=11, polyorder=2).reshape(-1,2,21,2)
    # faces_smooth = smooth_face(faces_cat)
    for i in range(len(motion_data)):
        motion_data[i]['bodies']['candidate'] = bodies_smooth[i]
        motion_data[i]['hands'] = hands_smooth[i]
        # motion_data[i]['faces'] = np.expand_dims(faces_smooth[i], axis=0)
    return motion_data


if __name__ == '__main__':
    motion_folder = '/home/yangxu/tencent-cospeech/_data/_datasets/4b7NfEZWBzw_1h/4b7NfEZWBzw_1h_filtered/pose'
    output_folder = '/home/yangxu/tencent-cospeech/_data/_datasets/4b7NfEZWBzw_1h/4b7NfEZWBzw_1h_filtered/pose_smooth'
    os.makedirs(output_folder, exist_ok=True)
    filter_list = []
    n = 0
    for motion_file in os.listdir(motion_folder):
        if motion_file.endswith('.npy'):
            output_file = os.path.join(output_folder, motion_file)
            motion_file = os.path.join(motion_folder, motion_file)
            data = np.load(motion_file)

            person_num = data.shape[1]

            smoothed_final = np.array([])
            for i in range(person_num):
                motion = data[:, i, :, :2]
                confidence = data[:, i, :, 2]
                # Split the data into parts to smooth and parts to keep
                data_to_smooth = np.concatenate([motion[:, :71, :], motion[:, 91:, :]], axis=1)
                mouth = motion[:, 71:91, :]

                inner_mouth = mouth[:, 12:, :]
                outer_mouth = mouth[:, :12, :]

                # smooth the outer mouth
                smoothed_outer_mouth = smooth_kp(outer_mouth, window_size=7, polyorder=2)

                # smooth the inner mouth
                smoothed_inner_mouth = smooth_kp(inner_mouth, window_size=5, polyorder=2)

                # Smooth the combined parts
                smoothed_data = smooth_kp(data_to_smooth, window_size=11, polyorder=2)

                # Concatenate the smoothed and unsmoothed parts
                smoothed_data = np.concatenate(
                    [smoothed_data[:, :71, :], smoothed_outer_mouth, smoothed_inner_mouth, smoothed_data[:, 71:, :]], axis=1
                )
                smoothed_data = smoothed_data[:, np.newaxis, ...]
                smoothed_data = np.concatenate([smoothed_data, confidence[:, np.newaxis, :, np.newaxis]], axis=-1)

                smoothed_final = np.concatenate([smoothed_final, smoothed_data], axis=1) if smoothed_final.size else smoothed_data

            np.save(output_file, smoothed_final)
            print(f"smooth {output_file}, {n}")