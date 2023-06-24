import matplotlib.pyplot as plt
import os
import log
from scipy.spatial.transform import Rotation as R
import numpy as np
from math import sin, cos, atan2, sqrt
import numpy.matlib as matlib


class Plotter(object):
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.counter = 0

    def plot(self, plots, xlabel, ylabel, title, labels=None, equal_axes=False, filename=None, callback=None, colors=None):
        if not labels:
            labels_txt = [None] * len(plots)
        else:
            labels_txt = labels
        assert (len(plots) == len(labels_txt))

        plt.clf()
        for i in range(0, len(plots)):
            args = {
                "label": labels_txt[i]
            }
            if colors:
                args["color"] = colors[i]
            plt.plot(*plots[i], linewidth=1, **args)

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        if title:
            plt.title(title)

        if equal_axes:
            plt.axis("equal")

        if labels:
            plt.legend()

        plt.grid()
        if filename is None:
            filename = "%02d_%s.svg" % (self.counter, "_".join(title.lower().split()))

        if callback is not None:
            callback(plt.gcf(), plt.gca())

        plt.savefig(log.Logger.ensure_file_dir_exists(os.path.join(self.output_dir, filename)),  format='svg', bbox_inches='tight', pad_inches=0)
        self.counter += 1

def line2mat(line_data):
    '''
    12 -> 4 x 4
    '''
    mat = np.eye(4)
    mat[0:3,:] = line_data.reshape(3,4)
    return np.matrix(mat)
def pose2motion(data, skip=0):
    '''
    data: N x 12
    all_motion (N-1-skip) x 12
    '''
    data_size = data.shape[0]
    all_motion = np.zeros((data_size-1-skip,12))
    for i in range(0,data_size-1-skip):
        pose_curr = line2mat(data[i,:])
        pose_next = line2mat(data[i+1+skip,:])
        motion = pose_curr.I*pose_next
        motion_line = np.array(motion[0:3,:]).reshape(1,12)
        all_motion[i,:] = motion_line
    return all_motion


def build_se3_transform(xyzrpy):
    """Creates an SE3 transform from translation and Euler angles.

    Args:
        xyzrpy (list[float]): translation and Euler angles for transform. Must have six components.

    Returns:
        numpy.matrixlib.defmatrix.matrix: SE3 homogeneous transformation matrix

    Raises:
        ValueError: if `len(xyzrpy) != 6`

    # """
    # if len(xyzrpy[-1]) != 6:
    #     raise ValueError("Must supply 6 values to build transform")
    se3 = matlib.identity(4)
    se3[0:3, 0:3] = euler_to_so3(xyzrpy[3:6])
    se3[0:3, 3] = np.matrix(xyzrpy[0:3]).transpose()
    return se3


def euler_to_so3(rpy):
    """Converts Euler angles to an SO3 rotation matrix.

    Args:
        rpy (list[float]): Euler angles (in radians). Must have three components.

    Returns:
        numpy.matrixlib.defmatrix.matrix: 3x3 SO3 rotation matrix

    Raises:
        ValueError: if `len(rpy) != 3`.

    """
    if len(rpy) != 3:
        raise ValueError("Euler angles must have three components")

    R_x = np.matrix([[1, 0, 0],
                     [0, cos(rpy[0]), -sin(rpy[0])],
                     [0, sin(rpy[0]), cos(rpy[0])]])
    R_y = np.matrix([[cos(rpy[1]), 0, sin(rpy[1])],
                     [0, 1, 0],
                     [-sin(rpy[1]), 0, cos(rpy[1])]])
    R_z = np.matrix([[cos(rpy[2]), -sin(rpy[2]), 0],
                     [sin(rpy[2]), cos(rpy[2]), 0],
                     [0, 0, 1]])
    R_zyx = R_z * R_y * R_x
    return R_zyx

if __name__ == '__main__':
    import torch
    a = torch.rand(10,6)
    out = []
    for i in range(a.size(0)):
        b = build_se3_transform(a[i,:].squeeze())
        out.append(b)
    print(len(out))
    z = torch.stack(out)
    print(z.shape)