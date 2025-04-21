import numpy as np
from lib.calculateFK import FK
#from calculateFK import FK

def calcJacobian(q_in):
    """
    Calculate the full Jacobian of the end effector in a given configuration
    :param q_in: 1 x 7 configuration vector (of joint angles) [q1,q2,q3,q4,q5,q6,q7]
    :return: J - 6 x 7 matrix representing the Jacobian, where the first three
    rows correspond to the linear velocity and the last three rows correspond to
    the angular velocity, expressed in world frame coordinates
    """

    J = np.zeros((6, 7))

    ## STUDENT CODE GOES HERE

    # Create an FK instance to get joint positions and axes
    fk = FK()

    # Compute forward kinematics to get position of each joint
    joint_positions, T0e = fk.forward(q_in)

    # end-effector in world-frame
    o_ee = joint_positions[7] #T0e

    # Get each joint’s rotation axis z_i in world frame
    z_axes = fk.get_axis_of_rotation(q_in)  # shape (3,7)

    # Build columns of J for i=0..6
    for i in range(7):

        # Position joint i in world frame
        o_i = joint_positions[i]

        # Revolute axis is z_axes[:,i]
        z_i = z_axes[:, i]

        # Linear velocity component = z_i × (o_ee - o_i)
        J[0:3, i] = np.cross(z_i, (o_ee - o_i))

        # Angular velocity component = z_i
        J[3:6, i] = z_i

    return J

if __name__ == '__main__':
    q= np.array([0, 0, 0, -np.pi/2, 0, np.pi/2, np.pi/4])
    print(np.round(calcJacobian(q),3))

    dq = np.array([1, 0, 0, 0, 0, 0, 0])
    #print(np.round(FK))
