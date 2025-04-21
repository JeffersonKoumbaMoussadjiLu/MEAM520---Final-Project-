import numpy as np
from lib.calcJacobian import calcJacobian



def IK_velocity(q_in, v_in, omega_in):
    """
    :param q_in: 1 x 7 vector corresponding to the robot's current configuration.
    :param v_in: The desired linear velocity in the world frame. If any element is
    Nan, then that velocity can be anything
    :param omega_in: The desired angular velocity in the world frame. If any
    element is Nan, then that velocity is unconstrained i.e. it can be anything
    :return:
    dq - 1 x 7 vector corresponding to the joint velocities. If v_in and omega_in
         are infeasible, then dq should minimize the least squares error. If v_in
         and omega_in have multiple solutions, then you should select the solution
         that minimizes the l2 norm of dq
    """

    ## STUDENT CODE GOES HERE

    dq = np.zeros((1, 7))

    v_in = v_in.reshape((3,1))
    omega_in = omega_in.reshape((3,1))

    # Construct full 6D velocity target: [v_in; omega_in]
    #    shape will be (6,)
    desired_velocity = np.concatenate([v_in.flatten(), omega_in.flatten()])

    # Identify which entries are not NaN
    mask = ~np.isnan(desired_velocity)  # boolean array of shape (6,)

    # Get full Jacobian J (6 x 7)
    J = calcJacobian(q_in)

    # Slice Jacobian and target velocity to keep only valid constraints
    J = J[mask, :]          # shape (k, 7), where k <= 6
    desired_velocity = desired_velocity[mask]  # shape (k,)

    # Solve in a least-squares sense using numpy
    dq, residuals, rank, s = np.linalg.lstsq(J, desired_velocity, rcond=None)

    return dq
