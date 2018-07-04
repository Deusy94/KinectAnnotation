import json
import os
from watch_n_patch import WATCH_N_PATCH_JOINTS


PANDORA_JOINTS = ['SpineBase', 'SpineMid', 'Neck', 'Head', 'ShoulderLeft', 'ElbowLeft',
                  'WristLeft', 'HandLeft', 'ShoulderRight', 'ElbowRight', 'WristRight',
                  'HandRight', 'SpineShoulder', 'HandTipLeft', 'ThumbLeft', 'HandTipRight', 'ThumbRight']

BAD_JSONS = ["Pandora/21/base_2_ID21",
             "Pandora/06/free_3_ID06", "Pandora/08/free_3_ID08",
             # All 12ID
             "Pandora/12/base_1_ID12", "Pandora/12/base_2_ID12",
             "Pandora/12/free_1_ID12", "Pandora/12/free_2_ID12",
             "Pandora/12/free_3_ID12",
             # All 14ID
             "Pandora/14/base_1_ID14",
             "Pandora/14/base_2_ID14", "Pandora/14/free_1_ID14",
             "Pandora/14/free_2_ID14", "Pandora/14/free_3_ID14",
             # All 16ID
             "Pandora/16/base_1_ID16", "Pandora/16/base_2_ID16",
             "Pandora/16/free_1_ID16", "Pandora/16/free_2_ID16",
             "Pandora/16/free_3_ID16",
             # All 17ID
             "Pandora/17/base_1_ID17", "Pandora/17/base_2_ID17",
             "Pandora/17/free_1_ID17", "Pandora/17/free_2_ID17",
             "Pandora/17/free_3_ID17",
             # All 20ID
             "Pandora/20/base_1_ID20", "Pandora/20/base_2_ID20",
             "Pandora/20/free_1_ID20", "Pandora/20/free_2_ID20",
             "Pandora/20/free_3_ID20",
             # All 21ID
             "Pandora/21/base_1_ID21", "/Pandora/21/base_2_ID21",
             "Pandora/21/free_1_ID21", "Pandora/21/free_2_ID21",
             "Pandora/21/free_3_ID21"
             ]

def get_joint_value(joints: dict, index: int):
    patch_type = WATCH_N_PATCH_JOINTS[index]
    pd_index = -1
    for i in range(len(PANDORA_JOINTS)):
        if PANDORA_JOINTS[i] == patch_type:
            pd_index = i
            break
    if pd_index >= 0:
        return joints[pd_index][0], joints[pd_index][1]
    else:
        return -1, -1


def fill_pandora(joints: dict):
    joints_fixed = dict()
    for key, value in joints.items():
        joints_fixed[key] = dict()
        for k in range(len(WATCH_N_PATCH_JOINTS)):
            joints_fixed[key][k] = get_joint_value(joints[key], k)
    return joints_fixed


def get_joints(data_txt_path: str, root_dir: str=None):
    JSON = json.load(open("{}/data.json".format(data_txt_path)))
    # DEPTH = f"{data_txt_path}/RGB"
    DEPTH = "{}/DEPTH".format(data_txt_path)
    names = get_image_name(DEPTH)
    joints = dict()
    if root_dir in BAD_JSONS:
        for el, v in JSON.items():
            joints["{}/{}_DEPTH.png".format(DEPTH, el)] = dict()
            # joints[f"{DEPTH}/{el}_RGB.png"] = dict()
            tmp = json.loads(v)
            for i in range(len(tmp['joints'])):
                x = tmp['joints'][i][0]
                y = tmp['joints'][i][1]
                joints["{}/{}_DEPTH.png".format(DEPTH, el)][i] = (round(x), round(y))
                # joints[f"{DEPTH}/{el}_RGB.png"][i] = (round(x), round(y))
        return fill_pandora(joints)
    for i in range(len(JSON)):
        curr_name = "{}/{}".format(DEPTH, names[i])
        joints[curr_name] = dict()
        # joints[i] = dict()
        for j in range(len(PANDORA_JOINTS)):
            # x = JSON[i]['joints'][j][0]
            # y = JSON[i]['joints'][j][1]
            # joints[i][j] = (x, y)
            x = JSON[i]['joints'][j][0]
            y = JSON[i]['joints'][j][1]
            joints[curr_name][j] = (round(x), round(y))
    return fill_pandora(joints)


def get_image_name(img_dir: str):
    images = os.listdir(img_dir)
    images.sort()
    return images


def main():
    joints = get_joints('/projects/hand_detection/Pandora/01/base_1_ID01')
    for el, v in joints.items():
        print("{} {}".format(el, v))
        break


if __name__ == '__main__':
    main()