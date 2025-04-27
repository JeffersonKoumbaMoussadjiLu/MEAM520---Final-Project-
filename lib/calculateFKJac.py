import numpy as np
from math import pi
from .calculateFK import FK # Import the reference FK class using relative import

class FK_Jac():

    def __init__(self):

        # TODO: you may want to define geometric parameters here that will be
        # useful in computing the forward kinematics. The data you will need
        # is provided in the lab 1 and 4 handout
        # Define DH parameters [α, a, d, θ]
        self.dh_params = np.array([
            # [α_i,    a_i,     d_i,     θ_i]
            [-pi/2,    0,      0.3330,   0], # Link 1 -> Frame 1 at J2 axis
            [pi/2,     0,      0,        0], # Link 2 -> Frame 2 at J3 axis
            [pi/2,     0.0825, 0.3160,   0], # Link 3 -> Frame 3 at J4 axis
            [-pi/2,   -0.0825, 0,        0], # Link 4 -> Frame 4 at J5 axis
            [pi/2,     0,      0.3840,   0], # Link 5 -> Frame 5 at J6 axis
            [pi/2,     0.0880, 0,       0], # Link 6 -> Frame 6 at J7 axis
            [0,        0,      0.21,   -pi/4],  # Link 7 -> Frame 7 (EE Frame), d=0.107+0.103, includes fixed rotation -pi/4

        ])

        # Joint center displacements relative to their respective DH frames {i}
        # Point 0: J1 center in {0}
        # Point 1: J2 center in {1}
        # ...
        # Point 6: J7 center in {6}
        # Point 7: EE origin in {7} (EE frame)
        self.joint_center_displacements = np.array([
            [0, 0, 0.141],    # Joint 1 Center relative to Base Frame {0}
            [0, 0, 0],        # Joint 2 Center relative to Frame {1}
            [0, 0, 0.195],    # Joint 3 Center relative to Frame {2}
            [0, 0, 0],        # Joint 4 Center relative to Frame {3}
            [0, 0, 0.125],    # Joint 5 Center relative to Frame {4}
            [0, 0, -0.015],   # Joint 6 Center relative to Frame {5}
            [0, 0, 0.051],    # Joint 7 Center relative to Frame {6}
            [0, 0, 0],        # End effector Origin relative to Frame {7} (EE Frame)
        ])

        # Virtual joint positions relative to End Effector Frame {7}
        self.p_V8_E = np.array([0, 0.100, -0.105]) # Virtual Joint 8 in EE Frame
        self.p_V9_E = np.array([0, -0.100, -0.105]) # Virtual Joint 9 in EE Frame


    def forward_expanded(self, q):
        """
        INPUT:
        q - 1x7 vector of joint angles [q0, q1, q2, q3, q4, q5, q6]

        OUTPUTS:
        jointPositions -10 x 3 matrix, where each row corresponds to a physical or virtual joint of the robot or end effector
                  Each row contains the [x,y,z] coordinates in the world frame of the respective joint's center in meters.
                  The base of the robot is located at [0,0,0].
        T0e       - a 10 x 4 x 4 homogeneous transformation matrix,
                  representing the each joint/end effector frame expressed in the
                  world frame
        """

        # Your code starts here

        jointPositions = np.zeros((10,3))
        T0e = np.zeros((10,4,4))

        T0_current = np.identity(4) # Transformation from world {0} to current frame {i}

        # Calculate position of Joint 1 center (Point 0)
        # Assumes joint_center_displacements[0] is location of J1 center in World Frame {0}
        p_local_J1 = np.append(self.joint_center_displacements[0], 1)
        p_world_J1 = T0_current @ p_local_J1
        jointPositions[0] = p_world_J1[:3]

        # Iterate through each joint (1 to 7)
        for i in range(7): # Corresponds to T_{i, i+1} using q[i]
            alpha, a, d, theta_offset = self.dh_params[i]
            theta = theta_offset + q[i]
            
            # Compute transformation matrix from frame {i} to frame {i+1}
            Ai = self.get_transform_matrix(alpha, a, d, theta)
            
            # Compute transformation from world frame {0} to frame {i+1}
            T0_next = T0_current @ Ai # T_{0, i+1}
            
            # Store T_{0, i+1} (T01, T02, ..., T07)
            T0e[i] = T0_next
            
            # Calculate position of center of Joint i+2 (or EE origin for i=6)
            # joint_center_displacements[i+1] is the displacement in frame {i+1}
            p_local = np.append(self.joint_center_displacements[i+1], 1)
            p_world = T0_next @ p_local
            
            # Store world position for points 1 through 7 (J2 center to EE origin)
            jointPositions[i+1] = p_world[:3]
            
            # Update current transformation for the next iteration
            T0_current = T0_next

        # After loop, T0_current is T07, which is the End Effector frame T0_EE
        T0_EE = T0_current

        # Store T0_EE for points 7 (EE), 8 (V1), 9 (V2)
        T0e[7] = T0_EE
        T0e[8] = T0_EE
        T0e[9] = T0_EE

        # Calculate world positions of virtual joints (Points 8 and 9)
        # Convert local virtual points in EE frame to homogeneous coordinates
        p_V8_E_hom = np.append(self.p_V8_E, 1)
        p_V9_E_hom = np.append(self.p_V9_E, 1)

        # Transform virtual points to world frame {0}
        p_V8_0 = T0_EE @ p_V8_E_hom
        p_V9_0 = T0_EE @ p_V9_E_hom

        # Store world coordinates of virtual joints
        jointPositions[8] = p_V8_0[:3]
        jointPositions[9] = p_V9_0[:3]

        # Your code ends here

        return jointPositions, T0e

    # feel free to define additional helper methods to modularize your solution for lab 1

    def get_transform_matrix(self, alpha, a, d, theta):
        """
        Returns the homogeneous transformation matrix for given DH parameters (standard DH)
        alpha, a, d are for link i-1, theta is for joint i
        This computes T_{i-1, i}
        """
        A = np.array([
            [np.cos(theta), -np.sin(theta)*np.cos(alpha),  np.sin(theta)*np.sin(alpha), a*np.cos(theta)],
            [np.sin(theta),  np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha), a*np.sin(theta)],
            [0,              np.sin(alpha),                np.cos(alpha),               d],
            [0,              0,                           0,                            1]
        ])
        return A

    def compute_Ai(self, q):
        """
        INPUT:
        q - 1x7 vector of joint angles [q0, q1, q2, q3, q4, q5, q6]

        OUTPUTS:
        Ai: - 4x4 list of np array of homogenous transformations describing the FK of the robot. Transformations are not
              necessarily located at the joint locations
        """
        # STUDENT CODE HERE

        return()
    
if __name__ == "__main__":

    fk_jac = FK_Jac()
    fk_ref = FK() # Instantiate the reference FK class

    # matches figure in the handout
    q = np.array([0,0,0,-pi/2,0,pi/2,pi/4])

    # Run the expanded FK
    joint_positions_exp, T0e_exp = fk_jac.forward_expanded(q)
    
    # Run the reference FK
    joint_positions_ref, T0e_ref = fk_ref.forward(q)

    print("--- Testing FK_Jac against FK ---")

    # Compare joint positions (first 8 points)
    print("Comparing Joint Positions (Points 0-7):")
    print("Expanded FK Positions:\n", joint_positions_exp[:8])
    print("Reference FK Positions:\n", joint_positions_ref)
    position_diff = np.linalg.norm(joint_positions_exp[:8] - joint_positions_ref)
    print(f"Difference norm: {position_diff}")
    if position_diff < 1e-6:
        print("Joint positions match!")
    else:
        print("Joint positions DO NOT match!")

    print("\n---")

    # Compare End Effector Pose (T0e[7] vs T0e_ref)
    print("Comparing End Effector Pose (T0e[7]):")
    print("Expanded FK T0e[7]:\n", T0e_exp[7])
    print("Reference FK T0e:\n", T0e_ref)
    pose_diff = np.linalg.norm(T0e_exp[7] - T0e_ref)
    print(f"Difference norm: {pose_diff}")
    if pose_diff < 1e-6:
        print("End effector poses match!")
    else:
        print("End effector poses DO NOT match!")

    print("\n---")

    print("Expanded FK Results:")
    print("Full Joint Positions (including virtual):\n", joint_positions_exp)
    # print("Full T0e stack:\n", T0e_exp) # Optional: print all transforms
    print("Virtual Joint 8 Position:", joint_positions_exp[8])
    print("Virtual Joint 9 Position:", joint_positions_exp[9])