import numpy as np
from math import pi

class FK():

    def __init__(self):

        # TODO: you may want to define geometric parameters here that will be
        # useful in computing the forward kinematics. The data you will need
        # is provided in the lab handout
        pass

    def get_homogeneous_matrix(self, index, dh_array):
        a = dh_array[index][0]
        alpha = dh_array[index][1]
        d = dh_array[index][2]
        theta = dh_array[index][3]

        return np.array([
            [np.cos(theta), -1 * np.sin(theta) * np.cos(alpha), np.sin(theta) * np.sin(alpha), a * np.cos(theta)],
            [np.sin(theta), np.cos(theta) * np.cos(alpha), -1 * np.cos(theta) * np.sin(alpha), a * np.sin(theta)],
            [0, np.sin(alpha), np.cos(alpha), d],
            [0, 0, 0, 1]
        ])

    def forward(self, q):
        """
        INPUT:
        q - 1x7 vector of joint angles [q0, q1, q2, q3, q4, q5, q6]

        OUTPUTS:
        jointPositions -8 x 3 matrix, where each row corresponds to a rotational joint of the robot or end effector
                  Each row contains the [x,y,z] coordinates in the world frame of the respective joint's center in meters.
                  The base of the robot is located at [0,0,0].
        T0e       - a 4 x 4 homogeneous transformation matrix,
                  representing the end effector frame expressed in the
                  world frame
        """

        # Your Lab 1 code starts here
        theta_1, theta_2, theta_3, theta_4, theta_5, theta_6, theta_7 = q

        # a, alpha, d, theta
        dh_params = np.array([
            [0, -pi/2, 0.333, theta_1],
            [0, pi/2, 0, theta_2],
            [0.0825, pi/2, 0.316, theta_3],
            [0.0825, pi/2, 0, pi + theta_4],
            [0, -pi/2, 0.384, theta_5],
            [0.088, pi/2, 0, theta_6 - pi],
            [0, 0, 0.21, theta_7 - pi/4]
        ])

        joint_offsets = np.array([
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.195, 1.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.125, 1.0],
            [0.0, 0.0, -0.015, 1.0],
            [0.0, 0.0, 0.051, 1.0],
            [0.0, 0.0, 0.0, 1.0]
        ])


        jointPositions = np.zeros((8,3))
        T0e = np.identity(4)

        jointPositions[0] = [0.0, 0.0, 0.141]

        for i in range(7):
            H = self.get_homogeneous_matrix(i, dh_params)
            if i == 0:
                T0e = H
                jointPositions[i+1] = np.dot(T0e, joint_offsets[i])[:3]
            else:
                T0e = np.dot(T0e, H)
                jointPositions[i+1] = np.dot(T0e, joint_offsets[i])[:3]

        # Your code ends here

        return jointPositions, T0e

    # feel free to define additional helper methods to modularize your solution for lab 1

    
    # This code is for Lab 2, you can ignore it ofr Lab 1
    def get_axis_of_rotation(self, q):
        """
        INPUT:
        q - 1x7 vector of joint angles [q0, q1, q2, q3, q4, q5, q6]

        OUTPUTS:
        axis_of_rotation_list: - 3x7 np array of unit vectors describing the axis of rotation for each joint in the
                                 world frame

        """
        # STUDENT CODE HERE: This is a function needed by lab 2
        axis_of_rotation_list = np.zeros((3, 7))
        T0 = np.eye(4)  # Base frame transformation

        # Define the DH parameters (same as in the forward function)
        theta_1, theta_2, theta_3, theta_4, theta_5, theta_6, theta_7 = q
        dh_params = np.array([
            [0, -pi/2, 0.333, theta_1],
            [0,  pi/2, 0,     theta_2],
            [0.0825, pi/2, 0.316, theta_3],
            [0.0825, pi/2, 0,     pi + theta_4],
            [0, -pi/2, 0.384, theta_5],
            [0.088,  pi/2, 0,     theta_6 - pi],
            [0,     0,    0.21,  theta_7 - pi/4]
        ])

        # For each joint, extract the z-axis and update the transformation matrix.
        for i in range(7):
            # The current joint's axis (z-axis of the current frame) in the world frame.
            axis_of_rotation_list[:, i] = T0[:3, 2]
            
            # Get the transformation matrix for the current joint.
            H = self.get_homogeneous_matrix(i, dh_params)
            
            # Update the cumulative transformation.
            T0 = T0.dot(H)
        
        return axis_of_rotation_list
    
    def compute_Ai(self, q):
        """
        INPUT:
        q - 1x7 vector of joint angles [q0, q1, q2, q3, q4, q5, q6]

        OUTPUTS:
        Ai: - 4x4 list of np array of homogenous transformations describing the FK of the robot. Transformations are not
              necessarily located at the joint locations
        """
        # STUDENT CODE HERE: This is a function needed by lab 2

        return()
    
if __name__ == "__main__":

    fk = FK()

    # matches figure in the handout
    q = np.array([0,0,0,-pi/2,0,pi/2,pi/4])
    # q = np.array([0,0,0,0,0,0,0])

    joint_positions, T0e = fk.forward(q)
    
    print("Joint Positions:\n",joint_positions)
    print("End Effector Pose:\n",T0e)