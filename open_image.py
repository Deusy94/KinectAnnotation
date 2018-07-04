import scipy
from scipy import io
import watch_n_patch
from PIL import Image
import cv2
import numpy as np


class MyImage:
    def __init__(self, path):
        self.path = path
        tmp = path.split('/')
        self.dataset_root = "/".join([x for i, x in enumerate(tmp)
                                      if i != len(tmp) - 1
                                      and i != len(tmp) - 2
                                      and i != len(tmp) - 3
                                      and i != len(tmp) - 4])
        self.dir_path = "/".join([x for i, x in enumerate(tmp)
                                  if i != len(tmp) - 1
                                  and i != len(tmp) - 2])
        self.type = 'patch' if self.path.split('/')[-1].split('.')[-1] == 'mat' else 'pandora'
        if self.type == 'patch':
            images = watch_n_patch.get_joints(f"{self.dir_path}")
            self.kpts = np.array([i for i in images[self.path].values()])
            img = scipy.io.loadmat(path)['depth']
            img = Image.fromarray(img)

            arr = np.array(img)
            tmp = np.zeros((arr.shape[0], arr.shape[1], 3))
            tmp[:, :, 0] = arr
            tmp[:, :, 1] = arr
            tmp[:, :, 2] = arr
            img = tmp

            self.img = img

        elif self.type == 'pandora':
            print("Not implemented")

    def imshow(self, name: str = "img"):
        img = self.img * 255 / np.amax(self.img)
        img = img.astype(np.uint8)
        for el in self.kpts:
            cv2.circle(img, (int(el[0]), int(el[1])), 3, (0, 0, 255), -1)
        cv2.imshow(name, img)


if __name__ == "__main__":
    img = MyImage("/projects/hand_detection/val/../train/watch_n_patch/kitchen/data_01-52-55/depth/0113.mat")
    img.imshow()
    cv2.waitKey(0)