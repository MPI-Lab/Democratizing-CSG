from copy import deepcopy
from functools import partial
from os.path import join as pjoin
import numpy as np
import importlib.util
from torch.utils.data import DataLoader
import os
from utils import dist_utils
import pickle
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from datetime import datetime


def get_dataset_config(config_path):
    module, namespace = config_path.rsplit(".", 1)
    return getattr(importlib.import_module(module, package=None), namespace)


def get_skeleton(dataset):
    return get_dataset_config(dataset).skeleton


def get_landmarks(dataset):
    return get_dataset_config(dataset).landmarks


def get_fps(dataset):
    return get_dataset_config(dataset).fps


def get_dims(dataset):
    return get_dataset_config(dataset).dims


def sample_distance(dataset):
    return get_dataset_config(dataset).distance


def sample_vertical_angle(dataset):
    return get_dataset_config(dataset).sample_elevation_angle()


def get_trajectory(dataset, motion):
    return get_dataset_config(dataset).extract_trajectory(motion)


def get_visualization_scale(dataset):
    return get_dataset_config(dataset).visualization_scale


def get_cond_mode(dataset):
    return get_dataset_config(dataset).cond_mode


def get_dataset(dataset, **kwargs):
    return get_dataset_config(dataset).dataset_class(**kwargs)


def get_data_augmentations(dataset):
    return get_dataset_config(dataset).data_augmentations


def get_dataset_from_args(args, **kwargs):
    default_kwargs = vars(args)
    default_kwargs.update(**kwargs)
    return get_dataset(args.dataset, **default_kwargs)


def get_collate_fn(dataset):
    return get_dataset_config(dataset).collate_fn


def get_dataset_loader(dataset_config, batch_size, **kwargs):
    default_kwargs = deepcopy(vars(get_dataset_config(dataset_config)))
    default_kwargs.update(**kwargs)
    dataset_object = default_kwargs["dataset_class"](**default_kwargs)
    # collate_fn = partial(default_kwargs["collate_fn"], sequence_length=default_kwargs["sequence_length"])

    if not default_kwargs['distributed']:
        if default_kwargs["split"] == "train":
            return DataLoader(
                dataset_object,
                batch_size=batch_size,
                shuffle=True,
                num_workers=16,
                drop_last=True,
                collate_fn=default_kwargs["collate_fn"],
                prefetch_factor=2,
                pin_memory=True,
            )
        else:
            return DataLoader(
                dataset_object,
                batch_size=batch_size,
                shuffle=False,
                num_workers=16,
                drop_last=False,
                collate_fn=default_kwargs["collate_fn"],
                prefetch_factor=2,
                pin_memory=True,
            )
    else:
        rank, world_size = dist_utils.get_dist_info()
        cached_data_path = pjoin(
            default_kwargs["save_dir"], f'cached_data_{default_kwargs["split"]}.pkl'
        )

        if dist_utils.is_main_process() and not os.path.isfile(cached_data_path):
            print(f"Data is being cached at device {rank}")
            pickle.dump(dataset_object, open(cached_data_path, 'wb'))
        if world_size > 1:
            dist.barrier()

        assert os.path.isfile(cached_data_path)
        print(f"Loading cached data on device {rank}")
        dataset_object = pickle.load(open(cached_data_path, 'rb'))

        if default_kwargs["split"] == "train":
            sampler = (
                DistributedSampler(dataset_object, shuffle=True)
                if dist.is_initialized()
                else None
            )
            return DataLoader(
                dataset_object,
                batch_size=batch_size,
                shuffle=False,
                num_workers=16,
                drop_last=True,
                collate_fn=default_kwargs["collate_fn"],
                prefetch_factor=2,
                pin_memory=True,
                sampler=sampler,
            )
        else:
            sampler = (
                DistributedSampler(dataset_object, shuffle=False)
                if dist.is_initialized()
                else None
            )
            return DataLoader(
                dataset_object,
                batch_size=batch_size,
                shuffle=False,
                num_workers=16,
                drop_last=False,
                collate_fn=default_kwargs["collate_fn"],
                prefetch_factor=2,
                pin_memory=True,
                sampler=sampler,
            )


def get_dataset_loader_from_args(args, **kwargs):
    default_args = deepcopy(vars(args))
    default_args.update(**kwargs)
    return get_dataset_loader(**default_args)


def get_datapath(dataset):
    return get_dataset_config(dataset).datapath


def get_mean(dataset):
    return np.load(get_dataset_config(dataset).mean_path)


def get_std(dataset):
    return np.load(get_dataset_config(dataset).std_path)


def get_num_actions(dataset):
    return getattr(get_dataset_config(dataset), "num_actions", 1)


def global_to_local(
    pose,
    face_start=23,
    mouth_start=71,
    left_hand_start=91,
    right_hand_start=112,
    left_shoulder=5,
    right_shoulder=6,
    left_wrist=9,
    right_wrist=10,
    face_scale=1,
    mouth_scale=1,
):
    """
    将姿态数据从全局坐标系转换为局部坐标系，并存储 neck（颈部）坐标到第0个关键点位置。

    Args:
        pose: (batch, frames, keypoints, 3) 的 numpy 数组，包含 x, y 坐标和置信度。

    Returns:
        local_pose: 局部坐标系下的姿态数组，neck 坐标储存在第0个关键点。
    """
    # 计算左右肩关键点的平均值作为 neck 坐标
    if len(pose.shape) == 3:
        pose = pose[None]
    neck_coords = np.ones((pose.shape[0], pose.shape[1], 1, 3))
    neck_coords[:, :, :, :2] = np.mean(
        pose[:, :, [left_shoulder, right_shoulder], :2], axis=2, keepdims=True
    )

    # 获取局部姿态（所有关键点减去 neck 坐标）
    local_pose = pose.copy()
    local_pose[:, :, :, :2] -= neck_coords[:, :, :, :2]  # 减去 neck 坐标

    # 使用各自身根节点调整相应身体部位的局部坐标
    face_end = face_start + 68
    mouth_end = mouth_start + 20
    left_hand_end = left_hand_start + 21
    right_hand_end = right_hand_start + 21

    local_pose[:, :, face_start:face_end, :2] -= local_pose[:, :, 0, :2][
        :, :, None
    ]  # 脸部使用 nose 关键点调整
    local_pose[:, :, face_start:face_end, :2] *= face_scale  # 脸部放大
    local_pose[:, :, mouth_start:mouth_end, :2] *= mouth_scale / face_scale  # 嘴巴放大
    local_pose[:, :, left_hand_start:left_hand_end, :2] -= local_pose[
        :, :, left_wrist, :2
    ][
        :, :, None
    ]  # 左手使用 left_hand_root 关键点调整
    local_pose[:, :, right_hand_start:right_hand_end, :2] -= local_pose[
        :, :, right_wrist, :2
    ][
        :, :, None
    ]  # 右手使用 right_hand_root 关键点调整

    # 将 neck 坐标添加到开头
    local_pose = np.concatenate([neck_coords, local_pose], axis=2)

    return local_pose


def local_to_global(
    local_pose,
    face_start=23,
    mouth_start=71,
    left_hand_start=91,
    right_hand_start=112,
    left_wrist=9,
    right_wrist=10,
    face_scale=1,
    mouth_scale=1,
):
    """
    将局部坐标的姿态数据还原为全局坐标系。

    Args:
        local_pose: 带有局部 x, y 和置信度的 numpy 数组。

    Returns:
        global_pose: 全局坐标系下的姿态数组。
    """
    # 提取存储在第 0 个关键点上的颈部（neck）全局坐标
    if len(local_pose.shape) == 3:
        local_pose = local_pose[None]
    
    # TODO: be commented
    # neck_coords = local_pose[:, :, :1, :]  # neck 信息 (从第 0 个关键点)
    # local_pose = local_pose[:, :, 1:, :]  # 去掉第 0 个点

    global_pose = local_pose.copy()

    face_end = face_start + 68
    mouth_end = mouth_start + 20
    left_hand_end = left_hand_start + 21
    right_hand_end = right_hand_start + 21

    # 恢复各自根节点上的全局坐标
    local_pose[:, :, mouth_start:mouth_end, :2] /= mouth_scale / face_scale  # 嘴巴部分
    global_pose[:, :, face_start:face_end, :2] = (
        local_pose[:, :, face_start:face_end, :2] / face_scale
        + local_pose[:, :, 0, :2][:, :, None]
    )  # 脸部分
    global_pose[:, :, left_hand_start:left_hand_end, :2] += local_pose[
        :, :, left_wrist, :2
    ][
        :, :, None
    ]  # 左手部分
    global_pose[:, :, right_hand_start:right_hand_end, :2] += local_pose[
        :, :, right_wrist, :2
    ][
        :, :, None
    ]  # 右手部分

    # # 恢复相对于颈部的全局坐标
    # global_pose[:, :, :, :2] += neck_coords[:, :, :, :2]

    return global_pose