from Datasets import ComposedDataset
import numpy as np
import ctypes
from watch_n_patch import WATCH_N_PATCH_JOINTS
import json, codecs, os, cv2
import tkinter
from tkinter import messagebox, simpledialog


RADIUS = 3
CLICK = False
MODIFYING = False
POINT = [-1, -1]
NAME = "good_annotations"
ADD_JOINT = False
KPT_IDX = -1
INFO = None
INFO2 = None
MASTER = None


def draw_kpts(img, kpts):
    for i, el in enumerate(kpts[0]):
        if i > 20:
            break
        if el[0] >= 0 and el[1] >= 0:
            if i == 7 or i == 11 or i == 15 or i == 19:
                cv2.circle(img, (int(el[0]), int(el[1])), 3, (0, 255, 0), -1)
            elif i == 13 or i == 17:
                cv2.circle(img, (int(el[0]), int(el[1])), 3, (0, 255, 255), -1)
            else:
                cv2.circle(img, (int(el[0]), int(el[1])), 3, (0, 0, 255), -1)


def search_near(x, y, kpts):
    global INFO
    for i, el in enumerate(kpts[0]):
        range_x = [el[0] - RADIUS + 2, el[0] + RADIUS + 2]
        range_y = [el[1] - RADIUS + 2, el[1] + RADIUS + 2]
        if range_x[0] <= x <= range_x[1] and range_y[0] <= y <= range_y[1]:
            print(WATCH_N_PATCH_JOINTS[i])
            INFO.set(WATCH_N_PATCH_JOINTS[i])
            MASTER.update()
            return int(el[0]), int(el[1])
    return -1, -1


def click_left(event, x, y, flags, param):
    global CLICK, POINT, MODIFYING, ADD_JOINT, KPT_IDX, INFO2
    name = param[0]
    img = param[1]
    kpts = param[2]
    if event == cv2.EVENT_LBUTTONDOWN:
        if MODIFYING is False:
            if CLICK is True:
                CLICK = False
                MODIFYING = True
                old_x, old_y = POINT
                POINT = [-1, -1]
                for el in kpts[0]:
                    if int(el[0]) == old_x and int(el[1]) == old_y:
                        el[0] = x
                        el[1] = y
                        break
                cv2.circle(img, (x, y), 3, (255, 0, 0), -1)
                cv2.imshow(name, img)
            elif ADD_JOINT is True:
                kpts[0][KPT_IDX][0] = x
                kpts[0][KPT_IDX][1] = y
                cv2.circle(img, (x, y), 3, (128, 0, 128), -1)
                cv2.imshow(name, img)
                MODIFYING = True
                ADD_JOINT = False
                KPT_IDX = -1
            else:
                if (img[y, x][0] == 0 and img[y, x][1] == 0 and img[y, x][2] == 255) \
                        or (img[y, x][0] == 0 and img[y, x][1] == 255 and img[y, x][2] == 0) \
                        or (img[y, x][0] == 0 and img[y, x][1] == 255 and img[y, x][2] == 255):
                    new_x, new_y = search_near(x, y, kpts)
                    cv2.circle(img, (new_x, new_y), 3, (255, 0, 0), -1)
                    CLICK = True
                    POINT = [new_x, new_y]
                    cv2.imshow(name, img)
        else:
            print("Accetta la modifica prima.")
            INFO2.set("Accetta la modifica prima.")
            MASTER.update()


def main():
    global MODIFYING, NAME, CLICK, ADD_JOINT, KPT_IDX, INFO, MASTER, INFO2
    d = ComposedDataset("/projects/hand_detection/val", split="validation")
    image_list = list()
    json_dict = dict()
    MASTER = tkinter.Tk()
    INFO = tkinter.StringVar()
    INFO2 = tkinter.StringVar()
    tkinter.Label(master=MASTER, textvariable=INFO).pack()
    tkinter.Label(master=MASTER, textvariable=INFO2, width=25).pack()
    INFO.set("....")
    INFO2.set("....")
    MASTER.update()
    if os.path.isfile(f"./patch_{NAME}.json"):
        with open(f"./patch_{NAME}.json", 'r') as f:
            json_dict = json.load(f)
    NEXT_NAME = None
    for i, (img, kpts, name) in enumerate(d):
        if name in json_dict:
            kpts = np.array(json_dict[name])
            # continue

        if NEXT_NAME:
            if name.split('/')[-3] == NEXT_NAME:
                continue
            NEXT = None

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

        tmp = img.astype(np.uint8).copy()
        kpts_back = kpts.copy()

        draw_kpts(tmp, kpts)
        cv2.namedWindow(name)
        cv2.namedWindow(rgb_name)
        cv2.moveWindow(rgb_name, 900, 300)
        cv2.moveWindow(name, 300, 300)
        cv2.setMouseCallback(name, click_left, [name, tmp, kpts])

        while True:
            cv2.imshow(rgb_name, rgb)
            cv2.imshow(name, tmp)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('\n') or key == ord('\r'):
                json_dict[name] = kpts.copy().tolist()
                cv2.destroyAllWindows()
                break

            elif key == ord('r'):
                tmp[:, :, :] = img.astype(np.uint8).copy()[:, :, :]
                draw_kpts(tmp, kpts)

            elif key == ord('c'):
                with open(f"./patch_{NAME}.json", 'w') as f:
                    json.dump(json_dict, f)
                exit(1)

            elif key == ord('p'):
                print("Changing sequence.")
                INFO2.set("Changing sequence.")
                MASTER.update()
                NEXT_NAME = name.split('/')[-3]
                break

            if MODIFYING is not True:
                if key == 27 and CLICK is True:
                    for el in kpts[0]:
                        if int(el[0]) == POINT[0] and int(el[1]) == POINT[1]:
                            el[0] = -1
                            el[1] = -1
                            break
                    tmp[:, :, :] = img.astype(np.uint8).copy()[:, :, :]
                    draw_kpts(tmp, kpts)
                    CLICK = False
                    MODIFYING = True

                elif key == ord('a') and CLICK is not True:
                    while KPT_IDX < 0:
                        val = simpledialog.askinteger("Input", "Insert Joint Number",
                                                       parent=MASTER,
                                                       minvalue=0, maxvalue=25)
                        try:
                            KPT_IDX = int(val)
                            if KPT_IDX < 0 or KPT_IDX > 20:
                                KPT_IDX = -1
                                continue
                            ADD_JOINT = True
                        except ValueError:
                            continue
            else:
                if key == ord('y') and MODIFYING is True:
                    tmp[:, :, :] = img.astype(np.uint8).copy()[:, :, :]
                    np.copyto(kpts_back, kpts)
                    draw_kpts(tmp, kpts)
                    MODIFYING = False

                elif key == ord('n') and MODIFYING is True:
                    tmp[:, :, :] = img.astype(np.uint8).copy()[:, :, :]
                    np.copyto(kpts, kpts_back)
                    draw_kpts(tmp, kpts)
                    MODIFYING = False



if __name__ == '__main__':
    main()