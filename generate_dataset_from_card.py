""" 从SD卡中生成数据集， mapping数据必须大于50s， 其他数据必须小于50s， 30s~50s之间的数据提示进行检查，小于5s的数据自动滤除，第一条小于50s的数据默认为calibration数据， 每一条mapping数据都会生成一个文件夹，一个文件夹是一个数据集。 """
import sys
import os
import argparse
import time
import av
import glob
import shutil
from tqdm import tqdm

cur_file_path = os.path.abspath(__file__)
ROOT_DIR = os.path.dirname(cur_file_path)
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

CARD_DIRS = ['/media/user/021B-FE04', '/media/user/0156-E940']

def main():
    parser = argparse.ArgumentParser(description='Generate dataset from card.')
    parser.add_argument('--task', type=str, required=True, help='The task to do, will set to the dataset name')
    args = parser.parse_args()

    card_dir = None
    for c in CARD_DIRS:
        if os.path.exists(c):
            card_dir = c
            break
    if os.path.exists(card_dir):
        print("-----------------")
        print(f"Card directory {card_dir} existes.")
    else:
        raise Exception(f"Card directory {card_dir} doesn't existes.")
    source_dir = os.path.join(card_dir, 'DCIM', "100GOPRO")

    files = glob.glob(os.path.join(source_dir, "*.MP4"))
    files.sort(key=os.path.getmtime)

    mapping_files = []
    mission_files = []
    calibration_files = []
    for file in files:
        if av.open(file):
            duration = av.open(file).duration
            if duration is None:
                print(f"Warning: File {file} has no duration, check it.")
                continue
            duration = duration / 1000_000
            if duration < 2:
                print(f"Warning: File {file} is too short, less than 2s, check it.")
            if 30 < duration < 50:
                print(f"Warning: File {file} is between 30s and 50s, check it.")
            if duration > 50:
                mapping_files.append(file)
            else:
                if len(calibration_files) == 0:
                    calibration_files.append(file)
                else:
                    mission_files.append(file)
        else:
            print(f"Warning: File {file} is not a valid MP4 file.")

    dataset_dirs = []
    dataset_backup_dirs = []
    time_str = time.strftime("%Y%m%d_%H%M")
    dataset_dir_template = os.path.join(ROOT_DIR, 'data', 'dataset', f'{args.task}_{time_str}')
    dataset_backup_template = os.path.join(ROOT_DIR, 'data', 'dataset_backup', f'{args.task}_{time_str}')
    first_raw_video_dir = None
    for dataset_i in tqdm(range(1, len(mapping_files) + 1)):
        dataset_dir = dataset_dir_template + f'_{dataset_i}'
        raw_video_dir = os.path.join(dataset_dir, 'raw_videos')
        if dataset_i == 1:
            first_raw_video_dir = raw_video_dir
        gripper_calibration_dir = os.path.join(raw_video_dir, 'gripper_calibration')
        dataset_dirs.append(dataset_dir)
        dataset_backup_dirs.append(dataset_backup_template + f'_{dataset_i}')
        assert not os.path.exists(dataset_dir)
        os.makedirs(gripper_calibration_dir)
        calibration_file_name = os.path.basename(calibration_files[0])
        shutil.copy(calibration_files[0], os.path.join(gripper_calibration_dir, calibration_file_name))
        if dataset_i == 1:
            for mission_file in tqdm(mission_files):
                mission_file_name = os.path.basename(mission_file)
                destination = os.path.join(raw_video_dir, mission_file_name)
                shutil.copy(mission_file, destination)
        else:
            for mission_file in tqdm(mission_files):
                mission_file_name = os.path.basename(mission_file)
                source = os.path.join(first_raw_video_dir, mission_file_name)
                destination = os.path.join(raw_video_dir, mission_file_name)
                shutil.copy(source, destination)
        

    for dataset_dir, mapping_file in zip(dataset_dirs, mapping_files):
        raw_video_dir = os.path.join(dataset_dir, 'raw_videos')
        shutil.copy(mapping_file, os.path.join(raw_video_dir, 'mapping.mp4'))

    for dataset_dir, dataset_backup_dir in zip(dataset_dirs, dataset_backup_dirs):
        shutil.copytree(dataset_dir, dataset_backup_dir)

    print("The script to run slam pipeline:")
    print(f"cd {ROOT_DIR}")
    for dataset_dir in dataset_dirs:
        print(f"FILENAME={os.path.basename(dataset_dir)}")
        print(f"python run_slam_pipeline.py universal_manipulation_interface/data/dataset/$FILENAME && \
python scripts_slam_pipeline/07_generate_replay_buffer.py -o data/dataset/$FILENAME/dataset.zarr.zip data/dataset/$FILENAME")

if __name__ == "__main__":
    main()