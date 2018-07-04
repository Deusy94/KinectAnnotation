import scipy.io
import os


WATCH_N_PATCH_JOINTS = ['SpineBase', 'SpineMid', 'Neck', 'Head', 'ShoulderLeft', 'ElbowLeft', 'WristLeft', 'HandLeft',
                        'ShoulderRight', 'ElbowRight', 'WristRight', 'HandRight', 'HipLeft', 'KneeLeft', 'AnkleLeft',
                        'FootLeft', 'HipRight', 'KneeRight', 'AnkleRight', 'FootRight', 'SpineShoulder', 'HandTipLeft',
                        'ThumbLeft', 'HandTipRight', 'ThumbRight']
JOINTS_SPLIT = {'Spine': [0, 1, 20], 'Head': [2, 3],
                'Upper': [4, 5, 6, 8, 9, 10], 'Hands': [7, 11, 21, 22, 23, 24],
                'Bottom': [12, 13, 14, 15, 16, 17, 18, 19]}

def get_joints(data_path: str):
    joints = dict()
    body = scipy.io.loadmat(f"{data_path}/body.mat")['body']
    DEPTH = f"{data_path}/depth"
    names = get_image_name(DEPTH)
    for frame in range(len(body)):
        for k in range(6):
            if body[frame][k]['isBodyTracked'] == 1:
                joint_tracked = body[frame][k]['joints']
                joints[f"{data_path}/depth/{names[frame]}"] = dict()
                # joints[frame] = dict()
                for i in range(len(joint_tracked[0][0][0])):
                    if joint_tracked[0][0][0][i]['trackingState'][0][0][0][0] == 0:
                        joints[f"{data_path}/depth/{names[frame]}"][i] = (-1, -1)
                    # elif joint_tracked[0][0][0][i]['trackingState'][0][0][0][0] == 1:
                    #     joints[f"{data_path}/depth/{names[frame]}"][i] = (-1, -1)
                    else:
                        for j in joint_tracked[0][0][0][i]['depth'][0]:
                            x = j[0][0]
                            y = j[1][0]
                            joints[f"{data_path}/depth/{names[frame]}"][i] = (round(x), round(y))
                            # joints[frame][i] = (x, y)
                break
    return joints


def get_image_name(img_dir: str):
    images = os.listdir(img_dir)
    images.sort()
    if any(".DS_Store" in s for s in images):
        images.remove(".DS_Store")
    if any("._.DS_Store" in s for s in images):
        images.remove("._.DS_Store")
    return images


def main():
    mat = scipy.io.loadmat('/projects/hand_detection/watch_n_patch/data_splits/kitchen_split.mat')
    kitchen_splits = mat['train_name'][0]
    mat = scipy.io.loadmat('/projects/hand_detection/watch_n_patch/data_splits/office_split.mat')
    office_splits = mat['train_name'][0]

    joints = get_joints('/projects/hand_detection/watch_n_patch/kitchen/data_01-05-11')
    for el, v in joints.items():
        print(f"{el} {v}")
        break


if __name__ == "__main__":
    main()