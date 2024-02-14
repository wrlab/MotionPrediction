import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R
import numpy as np
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

figure_limit = 1.9
fig = plt.figure(figsize=(9, 6))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

for i, (hp, hr, lp, lr, rp, rr) in enumerate(zip(head_pos, head_rot, lhand_pos, lhand_rot, rhand_pos, rhand_rot)):
    # head
    def cplot(pos, rot, color):
        origin = np.copy(pos)
        xpoint = np.array([0.1,0,0]) @ rot + origin
        ypoint = np.array([0,0.1,0]) @ rot + origin
        zpoint = np.array([0,0,0.1]) @ rot + origin
        
        plt.plot(xs = [origin[0], xpoint[0]],
                 zs = [origin[1], xpoint[1]],
                 ys = [origin[2], xpoint[2]], c = color, lw = 2.5)
        plt.plot(xs = [origin[0], ypoint[0]],
                 zs = [origin[1], ypoint[1]],
                 ys = [origin[2], ypoint[2]], c = color, lw = 2.5)
        plt.plot(xs = [origin[0], zpoint[0]],
                 zs = [origin[1], zpoint[1]],
                 ys = [origin[2], zpoint[2]], c = color, lw = 2.5)
    
    cplot(hp, hr, 'red')
    cplot(lp, lr, 'blue')
    cplot(rp, rr, 'green')

    ax.set_axis_off()
    ax.set_xlim(-0.6*figure_limit, 0.6*figure_limit)
    ax.set_ylim(-0.6*figure_limit, 0.6*figure_limit)
    ax.set_zlim(-0.2*figure_limit, 1.*figure_limit)
    plt.title('frame: ' + str(i))
    plt.pause(1 / 180)
    ax.cla()
