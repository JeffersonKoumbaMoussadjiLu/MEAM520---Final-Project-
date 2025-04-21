import numpy as np
from lib.IK_velocity import IK_velocity
from lib.calcJacobian import calcJacobian

"""
Lab 3
"""

def IK_velocity_null(q_in, v_in, omega_in, b):
    """
    :param q_in: 1 x 7 vector corresponding to the robot's current configuration.
    :param v_in: The desired linear velocity in the world frame. If any element is
    Nan, then that velocity can be anything
    :param omega_in: The desired angular velocity in the world frame. If any
    element is Nan, then that velocity is unconstrained i.e. it can be anything
    :param b: 7 x 1 Secondary task joint velocity vector
    :return:
    dq + null - 1 x 7 vector corresponding to the joint velocities + secondary task null velocities
    """

    ## STUDENT CODE GOES HERE

    '''
    dq = np.zeros((1, 7))
    null = np.zeros((1, 7))
    b = b.reshape((7, 1))
    v_in = np.array(v_in)
    v_in = v_in.reshape((3,1))
    omega_in = np.array(omega_in)
    omega_in = omega_in.reshape((3,1))
    '''

    '''
    # Calculate primary IK velocity solution
    dq = IK_velocity(q_in, v_in, omega_in)

    # Calculate Jacobian
    J = calcJacobian(q_in)

    # Make sure b is the right shape (7x1)
    b = np.array(b).reshape(7, 1)

    # Calculate pseudoinverse of Jacobian
    J_pinv = np.linalg.pinv(J)

    # Calculate null-space projector (I - J^(+)J)
    I = np.eye(7)  # Identity matrix of size 7x7
    null_space_projector = I - (J_pinv @ J)

    # Project secondary task onto the null space
    null = (null_space_projector @ b).T  # Convert to row vector

    return dq + null
    '''

    dq = IK_velocity(q_in, v_in, omega_in)

    # Setup velocity inputs
    v_in = np.array(v_in).reshape(-1)
    omega_in = np.array(omega_in).reshape(-1)
    b = np.array(b).reshape(7, 1)

    # Construct full 6D velocity target and find which entries are not NaN
    desired_velocity = np.concatenate([v_in, omega_in])
    mask = ~np.isnan(desired_velocity)  # boolean array of shape (6,)

    # Get full Jacobian
    J = calcJacobian(q_in)

    # Slice Jacobian to keep only rows corresponding to non-NaN velocity components
    J_constrained = J[mask, :]

    # Calculate pseudoinverse of constrained Jacobian
    J_pinv = np.linalg.pinv(J_constrained)

    # Calculate null-space projector (I - J⁺J) using the constrained Jacobian
    null_space_projector = np.eye(7) - (J_pinv @ J_constrained)

    # Project secondary task onto the null space
    null = (null_space_projector @ b).T

    return dq + null
