from lib.calculateFK import FK
from core.interfaces import ArmController
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fk = FK()

# the dictionary below contains the data returned by calling arm.joint_limits()
limits = [
    {'lower': -2.8973, 'upper': 2.8973},
    {'lower': -1.7628, 'upper': 1.7628},
    {'lower': -2.8973, 'upper': 2.8973},
    {'lower': -3.0718, 'upper': -0.0698},
    {'lower': -2.8973, 'upper': 2.8973},
    {'lower': -0.0175, 'upper': 3.7525},
    {'lower': -2.8973, 'upper': 2.8973}
 ]

# TODO: create plot(s) which visualize the reachable workspace of the Panda arm,
# accounting for the joint limits.
#
# We've included some very basic plotting commands below, but you can find
# more functionality at https://matplotlib.org/stable/index.html

# Number of random samples to draw
N_SAMPLES = 100000

# Arrays to store x,y,z points
points = np.zeros((N_SAMPLES, 3))

for i in range(N_SAMPLES):
    # Randomly sample a valid joint configuration
    q = np.array([
        np.random.uniform(limits[j]['lower'], limits[j]['upper'])
        for j in range(7)
    ])

    # Compute forward kinematics
    joint_positions, T0e = fk.forward(q)

    # The last row in joint_positions is the gripper origin (or the 7th joint),
    # depending on how your code is set up. Often index -1 gives the end-effector.
    # Here we'll just use that as "end-effector" for plotting:
    end_eff_xyz = joint_positions[-1]  # (x, y, z)

    points[i, :] = end_eff_xyz

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# TODO: update this with real results
#ax.scatter(1,1,1) # plot the point (1,1,1)
ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1, c='blue', alpha=0.3)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Approx. Reachable Workspace (Random Sampling)')

plt.show()
