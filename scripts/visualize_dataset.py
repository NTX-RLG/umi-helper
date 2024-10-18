import sys
import os

cur_file_path = os.path.abspath(__file__)
ROOT_DIR = os.path.dirname(os.path.dirname(cur_file_path))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

import random
import argparse
import zarr
import numpy as np
from tqdm import tqdm

import matplotlib.pyplot as plt

from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs, JpegXl
register_codecs()

def visualize_robot0_eef_pos(robot0_eef_pos, robot0_eef_rot_axis_angle, robot0_gripper_width):
    """
    可视化机器人末端执行器位置和夹爪宽度。

    参数:
    robot0_eef_pos (numpy.ndarray): 形状为 [N, 3] 的数组，表示末端执行器的位置。
    robot0_gripper_width (numpy.ndarray): 形状为 [N, 1] 的数组，表示夹爪的宽度。
    """
    x = robot0_eef_pos[:, 0]
    y = robot0_eef_pos[:, 1]
    z = robot0_eef_pos[:, 2]
    colors = ['b' if width < 0.075 else 'r' for width in robot0_gripper_width]

    fig = plt.figure(figsize=(20, 9))

    # 3D 位置图
    ax1 = fig.add_subplot(141, projection='3d')
    scatter = ax1.scatter(x[1:], y[1:], z[1:], c=colors[1:], label='robot0_eef_pos')
    ax1.scatter(x[0], y[0], z[0], marker='*', s = 300, color='g', label='start')
    ax1.plot(x, y, z)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('Robot0 End Effector Position')
    ax1.legend()
    # 计算三个轴的最小值和最大值
    min_val = min(min(x), min(y), min(z))
    max_val = max(max(x), max(y), max(z))

    # 生成相同的刻度值
    ticks = np.linspace(min_val, max_val, num=5)

    # 设置每个轴的刻度值相同
    ax1.set_xticks(ticks)
    ax1.set_yticks(ticks)
    ax1.set_zticks(ticks)

    # 曲线位置图
    ax2 = fig.add_subplot(142)
    ax2.plot(x, label='X')
    ax2.plot(y, label='Y')
    ax2.plot(z, label='Z')
    ax2.scatter(range(len(x)), x, c='b')
    ax2.scatter(range(len(y)), y, c='g')
    ax2.scatter(range(len(z)), z, c='r')

    ax2.legend()

    # 姿态
    ax3 = fig.add_subplot(143)
    p = robot0_eef_rot_axis_angle[:, 0]
    q = robot0_eef_rot_axis_angle[:, 1]
    r = robot0_eef_rot_axis_angle[:, 2]
    ax3.plot(p, label='p')
    ax3.plot(q, label='q')
    ax3.plot(r, label='r')
    ax3.legend()

    # 夹爪宽度图
    ax4 = fig.add_subplot(144)
    scatter = ax4.scatter(range(len(robot0_gripper_width)), robot0_gripper_width, c=colors, label='Gripper Width')
    ax4.plot(robot0_gripper_width)
    ax4.set_xlabel('Time')
    ax4.set_ylabel('Gripper Width')
    ax4.set_title('Robot0 Gripper Width (red for close)')
    ax4.legend()

    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', help="source folders which contains dataset.zarr.zip", default="data/dataset/fold_pink_towel_lyl_right_v2")
    args = parser.parse_args()

    dataset = args.dataset
    dataset_path = os.path.join(ROOT_DIR, dataset, 'dataset.zarr.zip')
    with zarr.ZipStore(str(dataset_path), mode='r') as zip_store:
        replay_buffer = ReplayBuffer.copy_from_store(
            src_store=zip_store, 
            store=zarr.MemoryStore()
        )
    print(f"There are {replay_buffer.n_episodes} in the dataset")
    index = -1
    while True:
        # input("press Enter to continue")
        index += 1
        index = index % replay_buffer.n_episodes
        result = replay_buffer.get_episode(index)
        robot0_eef_pos = result['robot0_eef_pos']  # [N, 3]
        robot0_eef_rot_axis_angle = result['robot0_eef_rot_axis_angle'] # [N, 3]
        robot0_gripper_width = result['robot0_gripper_width']  # [N, 1]
        visualize_robot0_eef_pos(robot0_eef_pos, robot0_eef_rot_axis_angle, robot0_gripper_width)

if __name__ == "__main__":
    main()