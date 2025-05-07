#!/usr/bin/env python3

import sys
import numpy as np
from copy import deepcopy
from math import pi
from time import sleep

import rospy
# Common interfaces for interacting with both the simulation and real environments!
from core.interfaces import ArmController
from core.interfaces import ObjectDetector

# for timing that is consistent with simulation or real time as appropriate
from core.utils import time_in_seconds

from lib.IK_position_null import IK
from lib.calculateFK import FK

class StaticGrabber():
    def __init__(self, detector, arm, team, ik, fk):
        self.detector = detector
        self.arm = arm
        self.team = team
        self.ik = ik
        self.fk = fk
        if team == 'blue':
            # Positions to scan for blocks (blue team)
            self.over_blk = np.array([
                [ 0.0337, -0.159 ,  0.1539, -2.0619,  0.0257,  1.9047,  0.9626],
                [ 0.2275, -0.0073,  0.2354, -1.8905,  0.0018,  1.8834,  1.2477],
                [ 0.1383,  0.1481,  0.029 , -1.6939, -0.0044,  1.842 ,  0.9535],
                [ 0.3408,  0.2899,  0.0638, -1.4989, -0.0187,  1.7882,  1.1914]
            ])
            # Positions to place blocks (blue team)
            self.set_point = np.array([
                [-0.0975,  0.2073, -0.1692, -2.0558,  0.0449,  2.2597,  0.4937],
                [-0.1087,  0.1437, -0.1578, -2.0025,  0.0268,  2.1443,  0.506 ],
                [-0.1168,  0.0973, -0.1489, -1.9295,  0.016 ,  2.0256,  0.5133],
                [-0.1218,  0.0688, -0.1432, -1.8359,  0.0104,  1.904 ,  0.5173],
                [-0.1241,  0.0593, -0.1411, -1.7201,  0.0085,  1.7789,  0.5186],
                [-0.1248,  0.0708, -0.1425, -1.578 ,  0.0101,  1.6482,  0.5177],
                [-0.1261,  0.1076, -0.1469, -1.401 ,  0.0158,  1.5075,  0.5143],
                [-0.1344,  0.1814, -0.1515, -1.1668,  0.0279,  1.3463,  0.5082],
            ])
            # Position to scan for blocks during placement
            self.tower_scan = np.array([-0.1344,  0.1814, -0.1515, -1.1668,  0.0279,  1.3463,  0.5082])
        else:
            # Positions to scan for blocks (red team)
            self.over_blk = np.array([
                [-0.1333, -0.1573, -0.0603, -2.0619, -0.01  ,  1.9049,  0.5958],
                [-0.232 , -0.0073, -0.2309, -1.8905, -0.0018,  1.8834,  0.323 ],
                [-0.0606,  0.1489, -0.1127, -1.6939,  0.0173,  1.8418,  0.6087],
                [-0.2217,  0.2946, -0.2052, -1.4986,  0.0606,  1.787 ,  0.3541]
            ])
            # Positions to place blocks (red team)
            self.set_point = np.array([
                [ 0.2318,  0.2045,  0.0301, -2.056 , -0.0079,  2.2604,  1.0517],
                [ 0.1986,  0.1423,  0.0645, -2.0026, -0.0109,  2.1445,  1.0538],
                [ 0.1755,  0.0966,  0.0882, -1.9295, -0.0095,  2.0257,  1.0528],
                [ 0.1619,  0.0684,  0.102 , -1.8359, -0.0074,  1.904 ,  1.0514],
                [ 0.1574,  0.0591,  0.1067, -1.7201, -0.0064,  1.7789,  1.0507],
                [ 0.1628,  0.0705,  0.1026, -1.5781, -0.0072,  1.6482,  1.0512],
                [ 0.1799,  0.107 ,  0.0881, -1.4011, -0.0094,  1.5076,  1.0524],
                [ 0.2123,  0.1799,  0.058 , -1.1669, -0.0106,  1.3465,  1.0525],
            ])
            # Position to scan for blocks during placement
            self.tower_scan = np.array([ 0.2123,  0.1799,  0.058 , -1.1669, -0.0106,  1.3465,  1.0525])

        # Get transform from camera to end effector
        self.H_ee_camera = detector.get_H_ee_camera()

        # Apply calibration adjustments based on team
        if team == 'blue':
            self.H_ee_camera[0,-1] -= 0.035
            self.H_ee_camera[1,-1] += 0.005
            self.H_ee_camera[2,-1] += 0.015
        else:
            self.H_ee_camera[0,-1] += 0.025
            self.H_ee_camera[1,-1] += 0.0
            self.H_ee_camera[2,-1] -= 0.02

    def move_to_over(self, i):
        """Move to a position to scan for blocks"""
        self.arm.safe_move_to_position(self.over_blk[i,:])

    def detect_and_convert(self):
        """Detect blocks and convert their poses to world coordinates"""
        result_list = []
        q = self.arm.get_positions()
        _, current_H = self.fk.forward(q)

        for (_, pose) in self.detector.get_detections():
            # Transform from camera frame to world frame
            result_list.append(np.matmul(current_H, np.matmul(self.H_ee_camera, pose)))

        print(len(result_list), " blocks are detected!!")
        return result_list

    def find_closest(self, transform_matrices):
        """Find the closest block to the robot base"""
        min_distance = np.inf
        closest_index = -1

        for index, matrix in enumerate(transform_matrices):
            displacement = matrix[:3, 3]  # Extract position
            distance = np.linalg.norm(displacement)

            if distance < min_distance:
                min_distance = distance
                closest_index = index

        return transform_matrices[closest_index]

    def find_axis(self, matrix):
        """Find the best axis for approach orientation"""
        target_pos = np.array([0, 0, 1])
        target_neg = np.array([0, 0, -1])

        min_distance = float('inf')
        closest_vector_index = -1

        # Find which column is closest to vertical
        for i in range(3):
            vector = matrix[:, i]
            distance_pos = np.linalg.norm(vector - target_pos)
            distance_neg = np.linalg.norm(vector - target_neg)
            distance = min(distance_pos, distance_neg)

            if distance < min_distance:
                min_distance = distance
                closest_vector_index = i

        # Get the other two vectors
        other_vectors = [matrix[:, i] for i in range(3) if i != closest_vector_index]

        # Choose the vector closest to x-axis
        v1 = other_vectors[0]
        v2 = other_vectors[1]
        chosen_axis = v1 if abs(v1[0]) > abs(v2[0]) else v2

        # Ensure x-axis is positive
        if chosen_axis[0] < 0:
            chosen_axis = -chosen_axis

        x = chosen_axis
        x[2] = 0  # Ensure horizontal
        normed_x = x / np.linalg.norm(x)

        return normed_x

    def find_ee_position(self, T_base_blk):
        """Extract position from transform matrix"""
        return T_base_blk[:3, 3]

    def find_ee_orientation(self, T_base_blk):
        """Compute end effector orientation for grasping"""
        R_base_blk = T_base_blk[:3, :3]
        x = self.find_axis(R_base_blk)
        z = np.array([0, 0, -1])  # Pointing down
        y = np.cross(z, x)        # Perpendicular to x and z
        return np.array([x, y, z]).T

    def find_target_ee_pose(self, T_base_blk):
        """Compute target end effector pose for grasping"""
        t_base_ee = self.find_ee_position(T_base_blk)
        R_base_ee = self.find_ee_orientation(T_base_blk)

        T_base_ee = np.eye(4)
        T_base_ee[:3, :3] = R_base_ee
        T_base_ee[:3, 3] = t_base_ee

        return T_base_ee

    def solve_ik(self, target, seed):
        """Solve inverse kinematics to get joint angles"""
        print("solving IK ...")
        q, _, success_pseudo, message_pseudo = self.ik.inverse(
            target, seed, method='J_pseudo', alpha=0.75)
        print(success_pseudo, message_pseudo)
        return q

    def grab(self, target_H, i):
        """Grab a block at the given pose"""
        # Compute target end effector pose
        target_H_at = self.find_target_ee_pose(target_H)

        # Solve IK for target pose
        target_q_at = self.solve_ik(target_H_at, self.arm.get_positions())
        if target_q_at is None:
            print("IK failed for this block.")
            return

        # Move to target pose and grab block
        print("Moving to grasp position...")
        self.arm.safe_move_to_position(target_q_at)
        self.arm.exec_gripper_cmd(0.04, 80)  # Close gripper

        # Move back to scanning position
        self.arm.safe_move_to_position(self.over_blk[i,:])

    def put(self, i):
        """Place block at the specified position in the stack"""
        # Move to placement position
        self.arm.safe_move_to_position(self.set_point[i,:])



        # Open gripper to release block
        self.arm.open_gripper()

        # Move to safe position above placement
        safe_position = deepcopy(self.set_point[i,:])
        safe_position[3] += 0.8  # Adjust joint 4
        safe_position[5] -= 0.8  # Adjust joint 6
        self.arm.safe_move_to_position(safe_position)

    def go_to_above(self):
        """Move to a safe position above the current position"""
        q = self.arm.get_positions()
        q[1] -= 0.4  # Adjust joint 2
        q[3] += 0.8  # Adjust joint 4
        q[5] -= 0.8  # Adjust joint 6
        self.arm.safe_move_to_position(q)

    def go_to_top_of_tower(self):
        """
        Move the arm to the fixed “top‐of‐tower” joint pose.
        Uses one of two hard-coded configurations depending on team color.
        """

        # move there
        self.arm.safe_move_to_position(self.tower_scan)


class DynamicGrabber():
    """
    A dynamic‐grasp class that
      1) Moves to a pose over the rotating turntable,
      2) Continuously detects the closest moving block,
      3) Steps closer each iteration,
      4) Once 'close enough,' performs a final approach and grabs.
    """
    def __init__(self, detector, arm, team, ik, fk):
        self.detector = detector
        self.arm      = arm
        self.team     = team
        self.ik       = ik
        self.fk       = fk

        # ---------------------------------------------------------
        if self.team == 'blue':
            self.pre_pose = np.array([ 0.7945-0.3, -1.6317, -1.6219, -0.8847,
                                      -0.2694, 1.6976, -0.8327 ])
            self.wait_pose = np.array([ 0.7945, -1.6317, -1.6219, -0.8847,
                                        -0.2694, 1.6976, -0.8327 ])
            # pose after a dynamic grab:
            self.reputpoint = np.array([ 0.2531,  0.1982,  0.0518, -2.0225,
                                        -0.0128, 2.2204,  1.0971])
            # redetection point
            self.redetectpoint = np.array([ 0.1888,  0.0769,  0.1193, -1.6597,
                                           -0.0093, 1.7360,  1.0947])
        else:  # 'red' team
            self.pre_pose = np.array([ 0.9728-0.3,  1.2579,  0.75,  -0.8674,
                                        0.5607, 1.6436, -1.1162 ])
            self.wait_pose = np.array([ 0.9728,  1.2579,  0.75,  -0.8674,
                                        0.5607, 1.6436, -1.1162 ])
            self.reputpoint = np.array([-0.1557,  0.1977, -0.2009, -2.0280,
                                         0.0493,  2.2213,  0.4028])
            self.redetectpoint = np.array([-0.1467,  0.0774, -0.1634, -1.6596,
                                            0.0127,  1.7360,  0.4737])

        # Camera->EE transform from your calibration
        self.H_ee_camera = detector.get_H_ee_camera()

        # Apply calibration adjustments based on team
        if team == 'blue':
            self.H_ee_camera[0,-1] -= 0.035
            self.H_ee_camera[1,-1] += 0.005
            self.H_ee_camera[2,-1] += 0.015
        else:
            self.H_ee_camera[0,-1] += 0.025
            self.H_ee_camera[1,-1] += 0.0
            self.H_ee_camera[2,-1] -= 0.02

    # ---------------------------------------------------------
    #  High-level moves
    # ---------------------------------------------------------
    def move_to_pre_pose(self):
        """Go to the pose for scanning the rotating table."""
        self.arm.safe_move_to_position(self.pre_pose)

    def move_to_wait_pose(self):
        self.arm.safe_move_to_position(self.wait_pose)

    def move_to_initial_pose(self):
        init_pose = np.array([
            -0.01779206, -0.76012354,  0.01978261, -2.34205014,
             0.02984053,  1.54119353+pi/2, 0.75344866
        ])
        self.arm.safe_move_to_position(init_pose)

    def put(self):
        """
        Simple place: move to 'reputpoint', open gripper.
        """
        self.arm.safe_move_to_position(self.reputpoint)
        self.arm.open_gripper()

    def get_over(self):
        """Move to a pose specifically for re-detection."""
        self.arm.safe_move_to_position(self.redetectpoint)

    def go_to_side(self):
        """ shift joint0 sideways a bit to avoid collisions."""
        q = self.arm.get_positions()
        if self.team == 'blue':
            q[0] -= 0.2
        else:
            q[0] += 0.2
        self.arm.safe_move_to_position(q)

    def wait_unitl_grabbed(self):
        """
        Wait to see if it grabbed something
        """
        sleep(0.5)  # Minimal approach


    # ---------------------------------------------------------
    #  The main partial-step dynamic approach
    # ---------------------------------------------------------
    def track_and_grab(self, max_tries=20):
        """
        Repeatedly:
          1. detect the block in camera
          2. partial-step closer
          3. once within threshold, final approach & close
        """
        print("\n*** Starting dynamic pickup ***")
        step_height_above_block = 0.02   # hover 2cm above
        close_dist_thresh       = 0.05   # final approach once <5cm

        for attempt in range(max_tries):
            # Current joint angles + forward kinematics
            q_now = self.arm.get_positions()
            _, H_base_ee_now = self.fk.forward(q_now)

            # Gather detections
            detections = self.detector.get_detections()
            if not detections:
                print(f"[{attempt}] No blocks detected yet, retrying.")
                sleep(0.3)
                continue

            # Convert each detection from camera->base
            T_block_candidates = []
            for _, pose_cam in detections:
                T_block_candidates.append(H_base_ee_now @ self.H_ee_camera @ pose_cam)

            # Pick the block nearest to the robot base
            block_T = self.find_closest(T_block_candidates)
            block_pos = block_T[:3, 3]

            # Build a “hover” transform: we want to be step_height_above_block over it
            T_des = np.eye(4)
            T_des[:3,:3] = self.downward_orientation()
            T_des[0,3]   = block_pos[0]
            T_des[1,3]   = block_pos[1]
            T_des[2,3]   = block_pos[2] + step_height_above_block

            dist = np.linalg.norm(block_pos - H_base_ee_now[:3,3])
            print(f"[{attempt}] Dist to block: {dist:.3f}")

            if dist < close_dist_thresh:
                # final approach
                T_des[2,3] = block_pos[2]
                q_final = self.solve_ik(T_des, q_now)
                if q_final is not None:
                    # Move in, close gripper
                    self.arm.safe_move_to_position(q_final)
                    self.arm.exec_gripper_cmd(0.03, 80)
                    self.wait_unitl_grabbed()

                    # Lift up
                    q_lift = deepcopy(q_final)
                    q_lift[1] -= 0.3  # for instance
                    self.arm.safe_move_to_position(q_lift)
                    print("Block grabbed!")
                    return True
                else:
                    print("Final approach IK failed—retrying.")
            else:
                # partial step
                q_step = self.solve_ik(T_des, q_now)
                if q_step is not None:
                    self.arm.safe_move_to_position(q_step)
                else:
                    print("Partial-step IK failed, retrying.")
            sleep(0.2)

        print("Gave up on dynamic pickup after too many tries.")
        return False

    # ---------------------------------------------------------
    #  Helper methods
    # ---------------------------------------------------------
    def find_closest(self, T_list):
        """Return whichever transform is nearest to the base origin."""
        min_dist = float('inf')
        best_T   = None
        for T in T_list:
            dist = np.linalg.norm(T[:3,3])
            if dist < min_dist:
                min_dist = dist
                best_T   = T
        return best_T

    def downward_orientation(self):
        """
        A rotation that points the end‐effector's Z axis along -Z_base.
        """
        R = np.eye(3)
        R[2,2] = -1
        return R

    def solve_ik(self, T_des, seed):
        """Use the same IK method as your static code."""
        q_sol, _, success, _ = self.ik.inverse(
            T_des, seed, method='J_pseudo', alpha=0.75)
        return q_sol if success else None



if __name__ == "__main__":
    try:
        team = rospy.get_param("team")  # 'red' or 'blue'
    except KeyError:
        print('Team must be red or blue - make sure you are running final.launch!')
        exit()

    rospy.init_node("team_script")

    # Initialize controllers and objects
    arm = ArmController()
    arm.set_arm_speed(0.15)  # set slower speed than before for both arm and gripper
    arm.set_gripper_speed(0.15)
    arm.open_gripper()

    # Get initial gripper state
    gripper_state = arm.get_gripper_state()
    print("gripper_state:", gripper_state['position'])

    # Initialize detectors and solvers
    detector = ObjectDetector()
    ik = IK()
    fk = FK()

    # Initialize both grabbers
    static_grabber = StaticGrabber(detector, arm, team, ik, fk)
    dynamic_grabber = DynamicGrabber(detector, arm, team, ik, fk)

    # Start at a known position
    start_position = np.array([
        -0.01779206, -0.76012354,  0.01978261, -2.34205014,
         0.02984053,  1.54119353+pi/2, 0.75344866
    ])
    arm.safe_move_to_position(start_position)

    print("\n****************")
    if team == 'blue':
        print("** BLUE TEAM  **")
    else:
        print("**  RED TEAM  **")
    print("****************")

    input("\nWaiting for start... Press ENTER to begin!\n")
    print("Go!\n")

    '''
    # 1) Grabbing static blocks
    print("Grabbing static blocks!!\n")
    for i in range(4):
        # move in and grab the i-th block exactly as before
        static_grabber.move_to_over(i)
        sleep(2)
        result_list = static_grabber.detect_and_convert()
        if len(result_list) == 0:
            print("No blocks detected, retrying!")
            i -= 1
            continue
        target_H = static_grabber.find_closest(result_list)
        static_grabber.grab(target_H, i)

        if i == 0:
            static_grabber.put(i)
            print("Placed first block directly at set_point[0].\n")
            continue

        # ─── NEW ────────────────────────────────────────────────────────
        # 2) lift straight up to get a clean view of the tower
        static_grabber.go_to_top_of_tower()
        sleep(1)

        #    re‐detect all blocks in the tower, pick the topmost one
        tower_list = static_grabber.detect_and_convert()
        if not tower_list:
            print("Couldn't see the tower—placing at default height.")
            top_H = None
        else:
            top_H = static_grabber.find_closest(tower_list)

        #    if we did see a top block, build a place‐pose that is exactly 6 cm above it
        if top_H is not None:
            place_H = top_H.copy()
            place_H[2, 3] += 0.06   # lift in Z by one block height + tolerance (6 cm)

            #    compute the gripper pose which aligns to that exact same orientation
            T_ee_place = static_grabber.find_target_ee_pose(place_H)

            #    solve IK and move there
            q_place = static_grabber.solve_ik(T_ee_place, static_grabber.arm.get_positions())
            if q_place is not None:
                static_grabber.arm.safe_move_to_position(q_place)
                static_grabber.arm.open_gripper()
                static_grabber.go_to_above()
            else:
                print("IK failed for aligned place—using default put()")
                static_grabber.put(i)
        else:
            # fallback if no tower seen
            static_grabber.put(i)
        # ─────────────────────────────────────────────────────────────────

        print(f"Successfully grabbed & placed block {i+1}!\n")
    '''

     # ---------------------------
    # 2) Now do the dynamic pickup
    # ---------------------------
    arm.open_gripper()
    arm.safe_move_to_position(start_position)

    print("\nNow let's try to grab the dynamic block!\n")
    # Move the robot to vantage above the rotating table
    dynamic_grabber.move_to_pre_pose()

    # Attempt tracking
    success = dynamic_grabber.track_and_grab()
    if success:
        dynamic_grabber.put()
        print("Dynamic block grabbed and placed successfully!")
    else:
        print("Failed to grab dynamic block.")

    # ---------------------------------------
    # Return to start or end gracefully
    # ---------------------------------------
    arm.safe_move_to_position(start_position)
    print("Done!")
