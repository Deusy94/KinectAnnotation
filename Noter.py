from Datasets import ComposedDataset
import numpy as np
from watch_n_patch import WATCH_N_PATCH_JOINTS
import json
import os
import cv2
import tkinter
from tkinter import simpledialog
from torch.utils.data import Dataset
import argparse


class Noter:
    def __init__(self, d: Dataset, ann_path, scale, radius):
        self.radius = radius
        self.is_clicked = False
        self.is_modifying = False
        self.point = [-1, -1]
        self.annotation_path = ann_path
        self.is_adding_joint = False
        self.kpt_idx = -1
        self.obj_idx = -1
        self.master = tkinter.Tk()
        self.info = tkinter.StringVar()
        self.error = tkinter.StringVar()
        self.scale = scale
        tkinter.Label(master=self.master, textvariable=self.info).pack()
        tkinter.Label(master=self.master, textvariable=self.error, width=25).pack()
        self.info.set("....")
        self.error.set("....")
        self.dataset = d
        self.image_list = list()
        self.json_dict = dict()
        if os.path.isfile(self.annotation_path):
            with open(self.annotation_path, 'r') as f:
                self.json_dict = json.load(f)

    def start(self, skip_or_keep: str = "skip"):
        self.master.update()
        next_name = None

        for i, (img, kpts, name) in enumerate(self.dataset):
            if name in self.json_dict:
                if skip_or_keep == "keep":
                    kpts = np.array(self.json_dict[name])
                else:
                    continue

            if next_name:
                if name.split('/')[-3] != next_name:
                    next_name = None
                    cv2.destroyAllWindows()
                else:
                    continue

            rgb_name = name.split('/')
            if name.split('.')[-1] == 'png':
                rgb_name[-2] = 'RGB'
                last_split = rgb_name[-1].split('_')
                last_split[-1] = 'RGB.png'
                rgb_name[-1] = "_".join(last_split)
                rgb_name = "/".join(rgb_name)
            if name.split('.')[-1] == 'mat':
                rgb_name[-2] = 'rgbjpg'
                last_split = rgb_name[-1].split('.')
                last_split[-1] = 'jpg'
                rgb_name[-1] = ".".join(last_split)
                rgb_name = "/".join(rgb_name)

            rgb = cv2.imread(rgb_name)
            rgb = cv2.resize(rgb, None, fx=0.4, fy=0.4, interpolation=cv2.INTER_CUBIC)

            img = img * 255 / np.amax(img)
            img[:, :, 0] = cv2.equalizeHist(cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2GRAY))
            img[:, :, 1] = img[:, :, 0]
            img[:, :, 2] = img[:, :, 0]
            img, kpts = self.upscale(img, kpts)

            tmp = img.astype(np.uint8).copy()
            kpts_back = kpts.copy()

            self.draw_kpts(tmp, kpts, self.radius)
            cv2.namedWindow(name)
            cv2.namedWindow(rgb_name)
            cv2.moveWindow(rgb_name, 900, 300)
            cv2.moveWindow(name, 300, 300)
            cv2.setMouseCallback(name, self.click_left, [name, tmp, kpts])

            while True:
                cv2.imshow(rgb_name, rgb)
                cv2.imshow(name, tmp)
                key = cv2.waitKey(1) & 0xFF

                if key == ord('\n') or key == ord('\r'):
                    _, d_k = self.downscale(kpts=kpts)
                    self.json_dict[name] = d_k.copy().tolist()
                    cv2.destroyAllWindows()
                    with open(self.annotation_path, 'w') as f:
                        json.dump(self.json_dict, f)
                    self.reset()
                    self.info.set("....")
                    self.error.set("....")
                    self.master.update()
                    break

                elif key == ord('r'):
                    tmp[:, :, :] = img.astype(np.uint8).copy()[:, :, :]
                    np.copyto(kpts, kpts_back)
                    self.draw_kpts(tmp, kpts, self.radius)
                    self.reset()

                elif key == ord('c'):
                    with open(self.annotation_path, 'w') as f:
                        json.dump(self.json_dict, f)
                    exit(1)

                elif key == ord('p'):
                    print("Changing sequence.")
                    self.error.set("Changing sequence.")
                    self.master.update()
                    next_name = name.split('/')[-3]
                    self.reset()
                    break

                elif key == ord('n'):
                    tmp[:, :, :] = img.astype(np.uint8).copy()[:, :, :]
                    np.copyto(kpts, kpts_back)
                    self.draw_kpts(tmp, kpts, self.radius)
                    self.reset()

                if self.is_modifying is not True:
                    if key == 27 and self.is_clicked is True:
                        for obj in range(kpts.shape[0]):
                            for el in kpts[obj]:
                                if int(el[0]) == self.point[0] and int(el[1]) == self.point[1]:
                                    el[0] = -1
                                    el[1] = -1
                                    break
                        tmp[:, :, :] = img.astype(np.uint8).copy()[:, :, :]
                        self.draw_kpts(tmp, kpts, self.radius)
                        self.is_clicked = False
                        self.is_modifying = True

                    elif key == ord('a') and self.is_clicked is not True:
                        obj = 0
                        val = 0
                        lgnd = ""
                        while self.kpt_idx < 0 and val is not None and obj is not None:
                            for key, v in enumerate(WATCH_N_PATCH_JOINTS):
                                if key > 20:
                                    break
                                lgnd += "{} for {}".format(key, v)
                                if key != 20:
                                    lgnd += ", "
                                    if (key + 1) % 3 == 0:
                                        lgnd += '\n'
                            val = simpledialog.askinteger("Input", "Insert Joint Number\n{}".format(lgnd),
                                                          parent=self.master,
                                                          minvalue=0, maxvalue=25)
                            self.master.update()
                            if val:
                                if kpts.shape[0] > 1:
                                    obj = simpledialog.askinteger("Input", "Insert obj number",
                                                                  parent=self.master,
                                                                  minvalue=0, maxvalue=kpts.shape[0])
                                if obj is not None:
                                    self.kpt_idx = val
                                    self.obj_idx = obj
                                    if self.kpt_idx < 0 or self.kpt_idx > 20:
                                        self.error.set("Inserisci solo valori tra 0 e 20.")
                                        self.master.update()
                                        self.kpt_idx = -1
                                        continue
                                    self.is_adding_joint = True
                else:
                    if key == ord('y') and self.is_modifying is True:
                        tmp[:, :, :] = img.astype(np.uint8).copy()[:, :, :]
                        np.copyto(kpts_back, kpts)
                        self.draw_kpts(tmp, kpts, self.radius)
                        self.reset()

    def reset(self):
        self.is_modifying = False
        self.is_clicked = False
        self.is_adding_joint = False
        self.obj_idx = -1
        self.kpt_idx = -1
        self.point = [-1, -1]

    @staticmethod
    def draw_kpts(img, kpts, radius):
        for obj in range(kpts.shape[0]):
            for i, el in enumerate(kpts[obj]):
                if i > 20:
                    break
                if el[0] >= 0 and el[1] >= 0:
                    if i == 7 or i == 11 or i == 15 or i == 19:
                        cv2.circle(img, (int(el[0]), int(el[1])), radius, (0, 255, 0), -1)
                    elif i == 13 or i == 17:
                        cv2.circle(img, (int(el[0]), int(el[1])), radius, (0, 255, 255), -1)
                    else:
                        cv2.circle(img, (int(el[0]), int(el[1])), radius, (0, 0, 255), -1)

    def search_near(self, x, y, kpts):
        for obj in range(kpts.shape[0]):
            for i, el in enumerate(kpts[obj]):
                if i > 20:
                    break
                range_x = [el[0] - self.radius + 2, el[0] + self.radius + 2]
                range_y = [el[1] - self.radius + 2, el[1] + self.radius + 2]
                if range_x[0] <= x <= range_x[1] and range_y[0] <= y <= range_y[1]:
                    print(WATCH_N_PATCH_JOINTS[i])
                    self.info.set(WATCH_N_PATCH_JOINTS[i])
                    self.master.update()
                    return int(el[0]), int(el[1])
        return -1, -1

    def click_left(self, event, x, y, flags, param):
        name = param[0]
        img = param[1]
        kpts = param[2]
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.is_modifying is False:
                if self.is_clicked is True:
                    self.is_clicked = False
                    self.is_modifying = True
                    old_x, old_y = self.point
                    self.point = [-1, -1]
                    for obj in range(kpts.shape[0]):
                        for el in kpts[obj]:
                            if int(el[0]) == old_x and int(el[1]) == old_y:
                                el[0] = x
                                el[1] = y
                                break
                    cv2.circle(img, (x, y), self.radius, (255, 0, 0), -1)
                    cv2.imshow(name, img)
                elif self.is_adding_joint is True:
                    kpts[self.obj_idx][self.kpt_idx][0] = x
                    kpts[self.obj_idx][self.kpt_idx][1] = y
                    cv2.circle(img, (x, y), self.radius, (128, 0, 128), -1)
                    cv2.imshow(name, img)
                    self.error.set("{} added.".format(WATCH_N_PATCH_JOINTS[self.kpt_idx]))
                    self.master.update()
                    self.is_modifying = True
                    self.is_adding_joint = False
                    self.kpt_idx = -1
                    self.obj_idx = -1
                else:
                    if (img[y, x][0] == 0 and img[y, x][1] == 0 and img[y, x][2] == 255) \
                            or (img[y, x][0] == 0 and img[y, x][1] == 255 and img[y, x][2] == 0) \
                            or (img[y, x][0] == 0 and img[y, x][1] == 255 and img[y, x][2] == 255):
                        new_x, new_y = self.search_near(x, y, kpts)
                        if new_y > 0 and new_x > 0:
                            cv2.circle(img, (new_x, new_y), self.radius, (255, 0, 0), -1)
                            self.is_clicked = True
                            self.point = [new_x, new_y]
                            cv2.imshow(name, img)
            else:
                print("Accetta la modifica prima.")
                self.error.set("Accetta la modifica prima.")
                self.master.update()

    @staticmethod
    def __resize(img, kpts, scale):
        up = up_k = None
        if img is not None:
            up = cv2.resize(img.copy(), None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        if kpts is not None:
            up_k = kpts.copy()
            for obj in range(up_k.shape[0]):
                for el in up_k[obj]:
                    if el[0] > 0 and el[1] > 0:
                        el[0] *= scale
                        el[1] *= scale
        return up, up_k

    def upscale(self, img=None, kpts=None):
        return self.__resize(img, kpts, self.scale)

    def downscale(self, img=None, kpts=None):
        return self.__resize(img, kpts, 1 / self.scale)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, dest='data_dir', help='Data directory.')
    parser.add_argument('--split', type=str, dest='split', help='Dataset split for custom dataset.')
    parser.add_argument('--out', type=str, default='good_annotations', dest='out', help='Output file path.')
    parser.add_argument('--k', type=str, default='skip',
                        dest='keep_or_skip', help='Keep or skip image already annotated')
    parser.add_argument('--scake', type=float, default=1.5, dest='scale', help='Depth image scale.')
    parser.add_argument('--radius', type=int, default=6, dest='radius', help='Joint annotation radius.')
    parser.add_argument('--add_path', type=str, default='/projects/hand_detection/patch_fixed_joints.json',
                        dest='ann_path', help='Joint annotation path.')
    args = parser.parse_args().__dict__

    n = Noter(ComposedDataset(args['data_dir'], split=args['split']), args['out'], args['scale'], args['radius'])
    n.start(args['keep_or_skip'])
