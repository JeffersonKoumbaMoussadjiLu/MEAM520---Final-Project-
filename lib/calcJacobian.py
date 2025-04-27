import numpy as np
from lib.calculateFK import FK

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

    fk = FK()
    joint_positions, _ = fk.forward(q_in)
    joint_axis_of_rotations = fk.get_axis_of_rotation(q_in)

    J_v = np.zeros((3, 7))
    J_omega = np.zeros((3, 7))

    end_effector_pos = joint_positions[7]
    
    for i in range(7):
        J_v[:, i] = np.cross(joint_axis_of_rotations[:, i], (end_effector_pos - joint_positions[i]))
        J_omega[:, i] = joint_axis_of_rotations[:, i]
    
    J = np.vstack((J_v, J_omega))

    return J

if __name__ == '__main__':
    # q= np.array([0, 0, 0, -np.pi/2, 0, np.pi/2, np.pi/4])
    q= np.array([0, 0, 0, 0, 0, np.pi, np.pi/4])
    print(np.round(calcJacobian(q),3))
