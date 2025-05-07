#!/usr/bin/env python3

import sys
import numpy as np
from copy import deepcopy
from math import pi
from time import sleep

import rospy

# Common interfaces
from core.interfaces import ArmController, ObjectDetector
from core.utils import time_in_seconds

# From your code snippet
from lib.IK_position_null import IK
from lib.calculateFK import FK

##################################################--HELPER FUNCTIONS--##################################################

def calculate_q_via_ik(pos, q_start, verify=True):
    """Use pseudo‐inverse IK with alpha=0.5, from your snippet."""
    q_end, rollout, success, message = ik.inverse(pos, q_start, method='J_pseudo', alpha=0.5)
    if verify:
        if success:
            return q_end
        else:
            print('Failed to find IK Solution:')
            print('pos:', pos)
            print('q_start:', q_start)
            return None
    return q_end

def swap_columns(matrix, col1, col2):
    matrix[:, [col1, col2]] = matrix[:, [col2, col1]]
    return matrix

def rotation_matrix_to_angle_axis(rotation_matrix):
    """
    Adjust the rotation to point its last column along [0,0,1] if possible,
    then measure yaw about z.  From your snippet.
    """
    rotation = rotation_matrix[:3, :3].copy()
    abs_rotation = np.abs(rotation)

    target_column = np.array([0,0,1])
    min_diff = 0.2
    col_to_swap = -1

    for i in range(3):
        diff = np.linalg.norm(abs_rotation[:, i] - target_column)
        if diff < min_diff:
            min_diff = diff
            col_to_swap = i

    rotation = swap_columns(rotation, col_to_swap, 2)

    rz = np.arctan2(rotation[1, 0], rotation[0, 0])
    if rz >= np.pi/4:
        rz -= np.pi/2
    elif rz < -np.pi/4:
        rz += np.pi/2

    return rz

#####################
# Additional Helpers
#####################

def drop_block(distance=0.09, force=10):
    arm.exec_gripper_cmd(distance, force)

def grab_block(distance=0.048, force=52):
    arm.exec_gripper_cmd(distance, force)

###########################
# Block detection & filter
###########################

def comp_filter(curr_reading, prev_reading, alpha):
    return alpha*curr_reading + (1 - alpha)*prev_reading

def get_block_world_comp_filter(q):
    """
    Example from your snippet that does repeated detection merges
    using comp_filter_alpha.  Then it returns a list of transforms
    in base frame, sorted as needed.
    """
    alpha=comp_filter_alpha
    H_ee_camera = detector.get_H_ee_camera()
    _, H_ee_w   = fk.forward(q)
    H_c_w       = H_ee_w @ H_ee_camera

    for i in range(5):
        b_reading_1 = []
        b_reading_2 = []
        block_det = detector.get_detections()

        for (name_1, pose_1) in block_det:
            b_reading_1.append(pose_1)
        b_reading_1 = np.array(b_reading_1)

        for (name_2, pose_2) in block_det:
            b_reading_2.append(pose_2)
        b_reading_2 = np.array(b_reading_2)

        b_reading_comp = comp_filter(b_reading_2, b_reading_2, alpha)
        b_reading_1 = b_reading_2
        b_reading_2 = b_reading_comp

    block_pose_world=[]
    for i in range(b_reading_comp.shape[0]):
        block_pose_world.append(H_c_w @ b_reading_comp[i])

    # filter out blocks not in [0.1, 0.4] in z
    block_pose_world = [b for b in block_pose_world if b[2,3]>0.1 and b[2,3]<0.4]

    # sort them: red vs. blue
    if team=='red':
        block_pose_world = sorted(block_pose_world, key=lambda x: x[1,3], reverse=True)
    else:
        block_pose_world = sorted(block_pose_world, key=lambda x: x[1,3])

    return len(block_pose_world), block_pose_world


def get_block_world(q_current, num_detects=1):
    """
    Example from your snippet that returns a sorted set of block transforms.
    """
    block_world = {}
    H_ee_camera = detector.get_H_ee_camera()

    for _ in range(num_detects):
        for (name, pose) in detector.get_detections():
            ee_block = H_ee_camera @ pose
            _, T0e   = fk.forward(q_current)
            cur_block= T0e @ ee_block
            if name not in block_world:
                block_world[name] = []
            block_world[name].append(cur_block)

    final_block_world=[]
    for key in block_world:
        block_worlds= block_world[key]

        block_worlds_R= [mat[:3,:3] for mat in block_worlds]
        avg_R = np.mean(block_worlds_R, axis=0)
        U,S,Vt= np.linalg.svd(avg_R)
        avg_R = U@Vt

        block_worlds_T= [mat[:3,3] for mat in block_worlds]
        avg_T = np.mean(block_worlds_T, axis=0)

        avg_block = np.eye(4)
        avg_block[:3,:3]= avg_R
        avg_block[:3,3] = avg_T
        final_block_world.append(avg_block)

    if team=='red':
        final_block_world= sorted(final_block_world, key=lambda x:x[1,3], reverse=True)
    else:
        final_block_world= sorted(final_block_world, key=lambda x:x[1,3])

    # filter
    final_block_world= [b for b in final_block_world if b[2,3]>0.1 and b[2,3]<0.4]
    block_count= len(final_block_world)
    return block_count, final_block_world


###################################
# A DirectDynamicGrabber Class
###################################
class DirectDynamicGrabber():
    """
    Does a single direct approach: move vantage-> detect-> final approach-> grab.
    """
    def __init__(self, arm, detector, team, ik, fk):
        self.arm      = arm
        self.detector = detector
        self.team     = team
        self.ik       = ik
        self.fk       = fk

    def grab_direct(self, q_vantage):
        """
        1) Move to vantage (q_vantage).
        2) Detect block once (closest).
        3) Build a single downward transform.
        4) IK -> move -> grab -> lift.
        """
        # vantage
        self.arm.safe_move_to_position(q_vantage)
        rospy.sleep(1.0)

        q_now = self.arm.get_positions()
        # detect
        if use_comp_filter:
            count, block_list = get_block_world_comp_filter(q_now)
        else:
            count, block_list = get_block_world(q_now, num_detects=5)

        if count==0:
            print("No dynamic blocks found from vantage.")
            return False

        # pick the last or first
        block_T = block_list[-1]
        block_pos= block_T[:3,3]

        # Build final approach transform
        T_des= np.eye(4)
        R_down= np.eye(3)
        R_down[2,2]= -1
        T_des[:3,:3]= R_down
        T_des[0,3]  = block_pos[0]
        T_des[1,3]  = block_pos[1]
        T_des[2,3]  = block_pos[2]  # or block_pos[2]+some_offset

        # IK
        q_final= calculate_q_via_ik(T_des, q_now, verify=True)
        if q_final is None:
            print("Direct approach IK failed.")
            return False

        # move + close
        self.arm.safe_move_to_position(q_final)
        grab_block()  # or exec_gripper_cmd(0.03,80)
        rospy.sleep(0.5)

        # lift
        q_lift= deepcopy(q_final)
        q_lift[1]-= 0.3
        self.arm.safe_move_to_position(q_lift)
        print("Direct dynamic block grab success!")
        return True

##################################
# MAIN
##################################
if __name__ == "__main__":
    try:
        team = rospy.get_param("team")  # 'red' or 'blue'
    except KeyError:
        print('Team must be red or blue - run final.launch!')
        sys.exit()

    rospy.init_node("team_script")

    # Arm, Detector
    arm = ArmController()
    detector= ObjectDetector()
    arm.set_arm_speed(0.15)
    arm.set_gripper_speed(0.15)
    arm.open_gripper()

    # IK + FK
    ik= IK()
    fk= FK()

    # Global stuff from snippet
    team= team
    use_comp_filter= False
    comp_filter_alpha= 0.78
    x_offset= 0.015 if team=='blue' else 0.005
    y_offset= 0.0
    z_fixed = 0.225

    # Start pose
    start_position= np.array([
        -0.01779206, -0.76012354,  0.01978261, -2.34205014,
         0.02984053,  1.54119353 + pi/2,  0.75344866
    ])
    arm.safe_move_to_position(start_position)

    print("\n****************")
    if team=='blue':
        print("** BLUE TEAM  **")
    else:
        print("**  RED TEAM  **")
    print("****************")

    input("\nPress ENTER to start.\n")

    # Suppose we do static blocks first (omitted)...

    # Now direct dynamic approach
    direct_grabber= DirectDynamicGrabber(arm, detector, team, ik, fk)

    # define vantage for dynamic
    if team=='blue':
        q_above_rotate= np.array([ 0.84, -0.88, -1.8, -1.15, -0.84, 1.47, -0.42])
    else:
        q_above_rotate= np.array([-0.83, -0.93, 1.72, -1.14, 0.93, 1.43, -1.18])
    # You might need to manually find a valid vantage

    # Attempt direct grab
    success= direct_grabber.grab_direct(q_above_rotate)
    if success:
        # place or do something
        # e.g. arm.open_gripper() after moving to a known place pose
        print("Dynamic block grabbed in one step!")
    else:
        print("Failed direct dynamic approach.")

    # back to start
    arm.safe_move_to_position(start_position)
    print("Done!")
