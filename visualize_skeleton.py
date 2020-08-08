import sys
import numpy as np

from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Agg')

if __name__ == '__main__':
    data = np.load(sys.argv[1])

    bones = [
        [0, 1],
        [1, 2],
        [2, 3],

        [0, 4],
        [4, 5],
        [5, 6],

        [0, 7],
        [7, 8],

        [8, 9],
        [9, 10],

        [8, 11],
        [11, 12],
        [12, 13],

        [8, 14],
        [14, 15],
        [15, 16]
    ]

    fig = plt.figure()
    ax = Axes3D(fig)

    START = 0
    ax.view_init(96.625, -8.78125)


    def update(num):
        ax.clear()

        skeleton = data[START + num][:, [2, 0, 1]]
        skeleton[:, 2] = -skeleton[:, 2]
        skeleton[:, 0] = -skeleton[:, 0]

        for i, joint in enumerate(skeleton):
            ax.scatter(*joint)
            ax.text(*joint, '%s' % (str(i)), size=10, zorder=1, color='k')

        for bone in bones:
            ax.plot(*zip(*skeleton[bone]), c='black')

        ax.set_xlim3d([-1.0, 1.0])
        ax.set_xlabel('X')

        ax.set_ylim3d([-1.0, 1.0])
        ax.set_ylabel('Y')

        ax.set_zlim3d([-1, 1])
        ax.set_zlabel('Z')


    N = 100
    update(0)

    fps = 50
    ani = FuncAnimation(fig, update, len(data), interval=1000 / 30, blit=False)
    ani.save(sys.argv[2], writer='ffmpeg')
