import os
import pandora
import numpy as np
import cv2
import watch_n_patch
import scipy
import psutil
import argparse
import math
from PIL import Image
from torch.utils.data import Dataset
from scipy import signal

DATASETS_DIR = '/projects/hand_detection'
PANDORA = 'Pandora'
PATCH = 'watch_n_patch'
"""
    SUBDIR -> {DATASETS_DIR}/{PANDORA}/{P_ID}/{P_SUB}
"""
P_ID = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11',
        '12', '13', '14', '15', '16', '17', '18', '19', '20', '21']
P_SUB = ['base_1_ID', 'base_2_ID', 'free_1_ID', 'free_2_ID', 'free_3_ID']
""" 
    mat = scipy.io.loadmat( {DATASETS_DIR}/{PATCH}/data_splits/kitchen.mat )
    kitchen_splits = mat['train_name'][0]
    SUBDIR -> {DATASETS_DIR}/{PATCH}/{W_SUB}/kitchen_splits
    MEAN = 1453.0
    STD = 534.0
    only Pandora
"""
W_SUB = ['kitchen', 'office']
MEAN = 1631.0
STD = 707.0

OFFICE_SPLIT = ['data_03-58-25', 'data_03-25-32', 'data_02-32-08', 'data_03-05-15']
KITCHEN_SPLIT = ['data_04-51-42', 'data_04-52-02', 'data_02-10-35', 'data_03-45-21']


class ComposedDataset(Dataset):
    def __init__(self, root_dir=None, base_transform=None,
                 input_transform=None, label_transform=None,
                 split="train"):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        print("Loader started.")
        self.root_dir = root_dir
        self.base_transform = base_transform
        self.input_transform = input_transform
        self.label_transform = label_transform
        self.P_ID = list()
        self.joints = dict()

        """
        print("Loading Pandora...")
        self.P_ID = os.listdir("{}/{}".format(root_dir, PANDORA))
        pandora_joints = dict()
        # Load Pandora
        for i in self.P_ID:
            for t in P_SUB:
                pandora_joints = {**pandora_joints, **pandora.get_joints("{}/{}/{}/{}{}".format(root_dir, PANDORA, i, t, i),
                                                                         "{}/{}/{}{}".format(PANDORA, i, t, i))}
        print("Done.")
        """

        if split == "train":
            name = 'train_name'
        else:
            root_dir = f"{root_dir}/../train"
            name = 'test_name'

        # Load Watch-n-patch
        print("Loading Watch-n-patch...",)
        mat = scipy.io.loadmat(f"{root_dir}/{PATCH}/data_splits/kitchen_split.mat")
        kitchen_splits = mat[name][0]
        mat = scipy.io.loadmat(f"{root_dir}/{PATCH}/data_splits/office_split.mat")
        office_splits = mat[name][0]
        patch_joints = dict()
        for el in kitchen_splits:
            if el not in KITCHEN_SPLIT:
                continue
            patch_joints = {**patch_joints, **watch_n_patch.get_joints(f"{root_dir}/{PATCH}/kitchen/{el[0]}")}
        for el in office_splits:
            if el not in OFFICE_SPLIT:
                continue
            patch_joints = {**patch_joints, **watch_n_patch.get_joints(f"{root_dir}/{PATCH}/office/{el[0]}")}
        print("Done.")

        # self.size = len(pandora_joints) + len(patch_joints) # + len(cad60_joints) + len(cad120_joints)
        self.size = len(patch_joints)
        # self.size = len(pandora_joints) + len(patch_joints)

        process = psutil.Process(os.getpid())
        byte = process.memory_info().rss
        print("{} images loaded, {}Gb in use.\n".format(self.size, round(byte/1000000000, 3)))
        # print("Pandora dim = {}; Patch dim = {}".format(len(pandora_joints), len(patch_joints)))
        # self.joints = {**patch_joints}
        # self.joints = {**pandora_joints, **patch_joints}
        self.joints = {**patch_joints}


    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        name = list(self.joints.keys())[idx]
        if name.split('.')[-1] == "mat":
            img = scipy.io.loadmat(name)['depth']
            img = Image.fromarray(img)
        else:
            img = Image.open(name)

        arr = np.array(img)
        tmp = np.zeros((arr.shape[0], arr.shape[1], 3))
        tmp[:, :, 0] = arr
        tmp[:, :, 1] = arr
        tmp[:, :, 2] = arr
        img = tmp

        kpts = [i for i in self.joints[name].values()]
        kpts = np.array(kpts)
        kpts = kpts[np.newaxis, :]

        return img, kpts, name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hypes', default=None, type=str,
                        dest='hypes', help='The file of the hyper parameters.')

    args = parser.parse_args()

    d = ComposedDataset("/projects/hand_detection/val", split="validation")

    i = 0
    print("Starting data loader.")
    for data_tuple in d:
        cv2.imshow()

if __name__ == '__main__':
    main()