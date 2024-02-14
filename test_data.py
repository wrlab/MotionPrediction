import numpy as np
from scipy.spatial.transform import Rotation as R
import torch

def matrot2sixd(pose_matrot):
    pose_6d = torch.cat([pose_matrot[:, :3, 0], pose_matrot[:, :3, 1]], dim=1)
    return pose_6d

data = np.load('data.npy')

head_pos = data[:, 0:3]
head_rot = data[:, 3:6]
head_rot = [R.from_euler('xyz', head_rot[i], False).as_matrix() for i in range(len(head_rot))]

lhand_pos = data[:, 6:9] + head_pos
lhand_rot = data[:, 9:12]
lhand_rot = [R.from_euler('xyz', lhand_rot[i], False).as_matrix() @ head_rot[i] for i in range(len(lhand_rot))]

rhand_pos = data[:, 12:15] + head_pos
rhand_rot = data[:, 15:18]
rhand_rot = [R.from_euler('xyz', rhand_rot[i], False).as_matrix() @ head_rot[i] for i in range(len(rhand_rot))]

head_pos = torch.Tensor(head_pos)
lhand_pos = torch.Tensor(lhand_pos)
rhand_pos = torch.Tensor(rhand_pos)

head_rot = torch.Tensor(head_rot)
lhand_rot = torch.Tensor(lhand_rot)
rhand_rot = torch.Tensor(rhand_rot)

head_vel = head_pos[1:] - head_pos[:-1]
lhand_vel = lhand_pos[1:] - lhand_pos[:-1]
rhand_vel = rhand_pos[1:] - rhand_pos[:-1]

head_rot_6d = matrot2sixd(head_rot)
lhand_rot_6d = matrot2sixd(lhand_rot)
rhand_rot_6d = matrot2sixd(rhand_rot)

head_rot_vel = torch.matmul(torch.inverse(head_rot[:-1]),head_rot[1:])
lhand_rot_vel = torch.matmul(torch.inverse(lhand_rot[:-1]),lhand_rot[1:])
rhand_rot_vel = torch.matmul(torch.inverse(rhand_rot[:-1]),rhand_rot[1:])

head_rot_vel_6d = matrot2sixd(head_rot_vel)
lhand_rot_vel_6d = matrot2sixd(lhand_rot_vel)
rhand_rot_vel_6d = matrot2sixd(rhand_rot_vel)

num_frames = head_pos.shape[0] - 1

sparse_data = torch.cat((
    head_rot_6d[1:].reshape(num_frames, -1),
    lhand_rot_6d[1:].reshape(num_frames, -1),
    rhand_rot_6d[1:].reshape(num_frames, -1),
    head_rot_vel_6d.reshape(num_frames, -1),
    lhand_rot_vel_6d.reshape(num_frames, -1),
    rhand_rot_vel_6d.reshape(num_frames, -1),
    head_pos[1:].reshape(num_frames, -1),
    lhand_pos[1:].reshape(num_frames, -1),
    rhand_pos[1:].reshape(num_frames, -1),
    head_vel.reshape(num_frames, -1),
    lhand_vel.reshape(num_frames, -1),
    rhand_vel.reshape(num_frames, -1)), dim=1)

np_sparse_data = np.array(sparse_data)
np.save('../AGRoL/sparse_data', np_sparse_data)

head_motion = torch.zeros((num_frames, 4, 4))
head_motion[:, 3, 3] = 1.0
head_motion[:, :3, :3] = head_rot[1:]
head_motion[:, :3, 3] = head_pos[1:]

np_head_motion = np.array(head_motion)
np.save('../AGRoL/head_motion', head_motion)

print(num_frames)


