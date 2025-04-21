import numpy as np
from math import pi, cos, sin

class FK():

    def __init__(self):

        # TODO: you may want to define geometric parameters here that will be
        # useful in computing the forward kinematics. The data you will need
        # is provided in the lab handout

        # DH parameters for each joints  (a, alpha, d, theta_offset, q_offset)
        self.joints = [
            {   # Joint 0
                'a': 0,
                'alpha': 0,
                'd': 0.141,
                'theta_offset': 0,
                'q_offset': 0
            },
            {   # Joint 1
                'a': 0,
                'alpha': -np.pi/2,
                'd': 0.192,
                'theta_offset': 0,
                'q_offset': 0
            },
            {   # Joint 2
                'a': 0,
                'alpha': np.pi/2,
                'd': 0,
                'theta_offset': 0,
                'q_offset': 0
            },
            {   # Joint 3
                'a': 0.0825,
                'alpha': np.pi/2,
                'd': 0.195 + 0.121,  # 0.195 + 0.121 = 0.316
                'theta_offset': 0,
                'q_offset': 0
            },
            {   # Joint 4
                'a': 0.0825,
                'alpha': np.pi/2,
                'd': 0,
                'theta_offset': np.pi/2,
                'q_offset': -np.pi/2
            },
            {   # Joint 5
                'a': 0,
                'alpha': -np.pi/2,
                'd': 0.259 + 0.125,  # 0.259 + 0.125 = 0.384
                'theta_offset': 0,
                'q_offset': 0
            },
            {   # Joint 6
                'a': 0.088,
                'alpha': np.pi/2,
                'd': 0,
                'theta_offset': -np.pi/2,
                'q_offset': np.pi/2
            },
            {   # Joint 7
                'a': 0,
                'alpha': 0,
                'd': 0.051 + 0.159,  # 0.051 + 0.159 = 0.210
                'theta_offset': 0,
                'q_offset': np.pi/4
            }
        ]

        # Adjustments for specific joints: {joint_index: offset}
        self.adjustments = {2: 0.195, 4: 0.125, 5: -0.015, 6: 0.051}

    ##############################################################################
    # feel free to define additional helper methods to modularize your solution for lab 1

    # Helper function

    # Function to compute 4x4 homogeneous transformation matrix using DH parameters
    def compute_dh_matrix(self, a, alpha, d, theta):

        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        cos_alpha = np.cos(alpha)
        sin_alpha = np.sin(alpha)

        return np.array([
            [cos_theta, -sin_theta * cos_alpha, sin_theta * sin_alpha, a * cos_theta],
            [sin_theta, cos_theta * cos_alpha, -cos_theta * sin_alpha, a * sin_theta],
            [0,          sin_alpha,             cos_alpha,             d          ],
            [0,          0,                      0,                     1          ]
        ])

    ############################################################################

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

##################################################################################
        #Joints positions and Transformations
        jointPositions = np.zeros((8,3))
        T0e = np.identity(4)

        q = np.insert(q, 0, 0)  # Add 0 angle at the start  for the fixed base joint

        transforms = [] #Store all the transformation matrices [T0_1, 1_2, ... Tn-1_n]

        #Loop through all the joints
        for i in range(8):

            joint = self.joints[i]

            theta = q[i] - joint['q_offset'] + joint['theta_offset'] # Compute theta

            dh = self.compute_dh_matrix(joint['a'], joint['alpha'], joint['d'], theta) #Compute the DH transform for this link (joints)

            T0e = T0e @ dh #Compute T0_i = A0 * A1 * ... * Ai (current_transform T0_i)

            transforms.append(T0e)
            jointPositions[i] = T0e[:3, 3] # Get the translation vector(joint Positions) in the transformations matrix (Last column)

        # Apply adjustments to specific joints (joints 2,4,5)
        for idx, offset in self.adjustments.items():

            rot_matrix = transforms[idx][:3, :3] #Get rotation Matrix from transformation matrix for the given joint

            # Compute adjustment vector by applying the rotation matrix to the offset vector [0, 0, offset].
            # This ensures that the offset is applied in the correct orientation relative to the joint's frame (From Office Hours)
            adjustment = rot_matrix @ np.array([0, 0, offset])

            jointPositions[idx] += adjustment

        # Your code ends here

        return jointPositions, T0e


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

        axes = np.zeros((3,7))

        # Start from identity as T_{base}, ignoring the “fixed base”
        T = np.eye(4)
        q = np.insert(q, 0, 0)

        for i in range(7):
            jnt = self.joints[i]
            # Compute the revolute joint’s axis angle
            theta = q[i] - jnt['q_offset'] + jnt['theta_offset']

            A = self.compute_dh_matrix(jnt['a'], jnt['alpha'], jnt['d'], theta)
            T = T @ A    # T0i

            # The z-axis of the i-th joint is the 3rd column of T0i’s rotation
            z_axis_world = T[:3,2]
            axes[:, i] = z_axis_world

        return axes

        #return()

    def compute_Ai(self, q):
        """
        INPUT:
        q - 1x7 vector of joint angles [q0, q1, q2, q3, q4, q5, q6]

        OUTPUTS:
        Ai: - 4x4 list of np array of homogenous transformations describing the FK of the robot. Transformations are not
              necessarily located at the joint locations
        """
        # STUDENT CODE HERE: This is a function needed by lab 2

        A_list = []
        # Insert dummy 0 for “fixed base”
        q = np.insert(q, 0, 0)

        for i in range(8):
            jnt = self.joints[i]
            theta = q[i] - jnt['q_offset'] + jnt['theta_offset']
            A = self.compute_dh_matrix(jnt['a'], jnt['alpha'], jnt['d'], theta)
            A_list.append(A)

        return A_list

        #return()

if __name__ == "__main__":

    fk = FK()

    # matches figure in the handout
    #q = np.array([0,0,0,-pi/2,0,pi/2,pi/4])
    q = np.array([0,0,0,0,0,0,0])


    joint_positions, T0e = fk.forward(q)

    print("Joint Positions:\n",joint_positions)
    print("End Effector Pose:\n",T0e)
