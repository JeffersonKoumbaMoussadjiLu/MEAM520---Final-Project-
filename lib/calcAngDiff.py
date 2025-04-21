import numpy as np


def calcAngDiff(R_des, R_curr):
    """
    Helper function for the End Effector Orientation Task. Computes the axis of rotation
    from the current orientation to the target orientation

    This data can also be interpreted as an end effector velocity which will
    bring the end effector closer to the target orientation.

    INPUTS:
    R_des - 3x3 numpy array representing the desired orientation from
    end effector to world
    R_curr - 3x3 numpy array representing the "current" end effector orientation

    OUTPUTS:
    omega - 0x3 a 3-element numpy array containing the axis of the rotation from
    the current frame to the end effector frame. The magnitude of this vector
    must be sin(angle), where angle is the angle of rotation around this axis
    """
    omega = np.zeros(3)
    ## STUDENT CODE STARTS HERE

    # Relative rotation in the current frame
    # Current orientation (R_curr) to the desired (R_des)
    R_rel = R_curr.T @ R_des

    # Skew-symmetric part of R_rel
    S = 0.5 * (R_rel - R_rel.T)

    # Extract "axis * sin(theta)" in the CURRENT frame
    a = np.array([S[2,1], S[0,2], S[1,0]])  # shape (3,)

    # If sin(theta) = 0 => zero rotation
    sin_theta = np.linalg.norm(a)

    if sin_theta != 0:
        omega = a
    else:
        omega = np.zeros(3)

    # Now we convert a into the WORLD frame
    # a is expressed in current frame => multiply by R_curr to get it in world frame
    omega = R_curr @ a

    return omega
