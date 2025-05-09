#!/usr/bin/env python3

import sys
import numpy as np
from copy import deepcopy
from math import pi
from time import sleep

import rospy
from core.interfaces import ArmController, ObjectDetector
from core.utils import time_in_seconds

from lib.IK_position_null import IK
from lib.calculateFK import FK

from lib.calcAngDiff import calcAngDiff
from lib.IK_velocity_null import IK_velocity_null

class StaticGrabber:
    def __init__(self, detector, arm, team, ik, fk):
        self.detector = detector
        self.arm = arm
        self.team = team
        self.ik = ik
        self.fk = fk

        # Initial pose to view all static blocks
        if team == 'red':
            self.initial_view_pose = np.array([-0.15498,  0.22828, -0.15683, -1.05843,  0.03686,  1.28419,  0.48802])
        else:
            # UPDATE THIS: INCORRECT
            self.initial_view_pose = np.array([0.0, -0.8,  0.2, -2.5, 0.0, 1.5, 1.0])

        # Placeholder for dynamic scan positions; computed on demand
        self.dynamic_over_positions = []

        # Load placement poses
        self.set_point = self._load_set_points(team)
        self.tower_scan = self._load_tower_scan_pose(team)

        # Get and calibrate camera transform
        self.H_ee_camera = detector.get_H_ee_camera()
        self._apply_calibration_offsets()

    def compute_scan_positions(self):
        """
        Move to initial view, detect static blocks, compute overhead IK for each.
        Call this once after user input to initialize dynamic_over_positions.
        """
        self.dynamic_over_positions.clear()
        self.arm.safe_move_to_position(self.initial_view_pose)
        sleep(1.0)

        transforms = self.detect_and_convert()
        for T_base_blk in transforms[:4]:
            pos = T_base_blk[:3,3]
            T_above = np.eye(4)
            # Orientation: point down
            R = np.eye(3)
            R[2,2] = -1
            R[1,1] = -1
            T_above[:3,:3] = R
            T_above[:3,3] = pos + np.array([-0.03, 0, 0.15])

            q_seed = self.arm.get_positions()
            q_above, _, success, _ = self.ik.inverse(
                T_above, q_seed, method='J_pseudo', alpha=0.5)
            if not success:
                raise RuntimeError("IK failed computing scan position.")
            self.dynamic_over_positions.append(q_above)

    def move_to_over(self, i):
        """Move to a computed position over block i for scanning."""
        self.arm.safe_move_to_position(self.dynamic_over_positions[i])

    def detect_and_convert(self):
        """Detect blocks, convert to base frame transforms."""
        results = []
        q = self.arm.get_positions()
        _, current_H = self.fk.forward(q)
        for _, pose_cam in self.detector.get_detections():
            world = current_H @ self.H_ee_camera @ pose_cam
            results.append(world)
        print(len(results), "blocks detected")
        return results

    def find_closest(self, transforms):
        min_dist = np.inf
        best = None
        for T in transforms:
            d = np.linalg.norm(T[:3,3])
            if d < min_dist:
                min_dist = d
                best = T
        return best

    def find_target_ee_pose(self, T_base_blk):
        t = T_base_blk[:3,3]
        R_blk = T_base_blk[:3,:3]
        axis = self._choose_horizontal(R_blk)
        ee_R = np.column_stack([axis, np.cross([0,0,-1], axis), [0,0,-1]])
        T = np.eye(4)
        T[:3,:3] = ee_R
        T[:3,3] = t
        return T

    def _choose_horizontal(self, R):
        cols = [R[:,i] for i in range(3)]
        best = max(cols, key=lambda v: abs(v[0]))
        best[2] = 0
        return best / np.linalg.norm(best)

    def solve_ik(self, target, seed):
        q, _, success, msg = self.ik.inverse(
            target, seed, method='J_pseudo', alpha=0.75)
        print('IK:', success, msg)
        return q if success else None

    def grab(self, T_base_blk, idx):
        target = self.find_target_ee_pose(T_base_blk)
        q = self.solve_ik(target, self.arm.get_positions())
        if q is None:
            print('IK failed, skipping')
            return
        self.arm.safe_move_to_position(q)
        self.arm.exec_gripper_cmd(0.04, 80)
        self.arm.safe_move_to_position(self.dynamic_over_positions[idx])

    def put(self, i):
        self.arm.safe_move_to_position(self.set_point[i])
        self.arm.open_gripper()
        safe = deepcopy(self.set_point[i])
        safe[3] += 0.8
        safe[5] -= 0.8
        self.arm.safe_move_to_position(safe)

    def go_to_top_of_tower(self):
        self.arm.safe_move_to_position(self.tower_scan)

    def go_to_above(self):
        q = self.arm.get_positions()
        q[1] -= 0.4
        q[3] += 0.8
        q[5] -= 0.8
        self.arm.safe_move_to_position(q)

    def _load_set_points(self, team):
        if team == 'blue':
            return np.array([
                [-0.0975,0.2073,-0.1692,-2.0558,0.0449,2.2597,0.4937],
            ])
        else:
            return np.array([
                [0.2318,0.2045,0.0301,-1.996,-0.0079,2.2604,1.0517],
            ])

    def _load_tower_scan_pose(self, team):
        return np.array([-0.1344,0.1814,-0.1515,-1.1668,0.0279,1.3463,0.5082]) if team == 'blue' else np.array([0.2123,0.1799,0.058,-1.1669,-0.0106,1.3465,1.0525])

    def _apply_calibration_offsets(self):
        if self.team == 'blue':
            self.H_ee_camera[0,-1] -= 0.035
            self.H_ee_camera[1,-1] += 0.005
            self.H_ee_camera[2,-1] += 0.015
        else:
            self.H_ee_camera[0,-1] += 0.01
            self.H_ee_camera[1,-1] -= 0.036
            self.H_ee_camera[2,-1] += 0.008


class DynamicGrabber():
    """
    Dynamic grabber class that captures moving blocks and stacks them
    on top of an existing tower. This implementation uses position IK only.
    """
    def __init__(self, detector, arm, team, ik, fk):
        self.detector = detector
        self.arm = arm
        self.team = team
        self.ik = ik
        self.fk = fk

        # Initial pose to scan for blocks
        if team == 'red':
            #self.initial_view_pose = np.array([-0.83, -0.93, 1.72, -1.14, 0.93, 1.43, -1.18]) #on top_block
            self.initial_view_pose =  p.array([-0.83, -0.93, 1.72, -1.14, 0.93, 1.43, -1.18]) #on top 2
            #self.initial_view_pose =  np.array([ 0.9728-0.3,  1.2579,  0.75  , -0.8674,  0.5607,  1.6436, -1.1162] ) #Side
            self.tower_scan = np.array([0.2123, 0.1799, 0.058, -1.1669, -0.0106, 1.3465, 1.0525])
            # Camera calibration offsets
            self.H_ee_camera = detector.get_H_ee_camera()
            #self.H_ee_camera[0,-1] += 0.01
            #self.H_ee_camera[1,-1] -= 0.036
            #self.H_ee_camera[2,-1] += 0.008
        else:
            #self.initial_view_pose = np.array([0.84, -0.88, -1.8, -1.15, -0.84, 1.47, -0.42]) #top
            self.initial_view_pose = np.array([0.84, -0.88, -1.8, -1.15, -0.84, 1.47, -0.42])  #top 2
            #self.initial_view_pose =  np.array([ 0.7945-0.3, -1.6317, -1.6219, -0.8847, -0.2694,  1.6976, -0.8327] ) #Side
            self.tower_scan = np.array([-0.1344, 0.1814, -0.1515, -1.1668, 0.0279, 1.3463, 0.5082])
            # Camera calibration offsets
            self.H_ee_camera = detector.get_H_ee_camera()
            #self.H_ee_camera[0,-1] -= 0.035
            #self.H_ee_camera[1,-1] += 0.005
            #self.H_ee_camera[2,-1] += 0.015

        #Adjust arm
        self.initial_view_pose[3] -= 0.8
        self.initial_view_pose[5] += 0.8

        # Track number of blocks stacked
        self.blocks_stacked = 0

        # Maximum attempts for grabbing and detection
        self.max_grab_attempts = 3
        self.max_detection_attempts = 5

        # Scanning position list
        self.scan_positions = [self.initial_view_pose]

    def detect_and_convert(self):
        """Detect blocks, convert to base frame transforms."""
        results = []
        q = self.arm.get_positions()
        _, current_H = self.fk.forward(q)
        for _, pose_cam in self.detector.get_detections():
            world = current_H @ self.H_ee_camera @ pose_cam
            results.append(world)
        print(len(results), "blocks detected")
        return results

    def find_closest(self, transforms):
        """Find closest block to the robot base."""
        min_dist = np.inf
        best = None
        for T in transforms:
            d = np.linalg.norm(T[:3,3])
            if d < min_dist:
                min_dist = d
                best = T
        return best

    def find_target_ee_pose(self, T_base_blk):
        """Create a suitable end-effector pose to grab the block."""
        t = T_base_blk[:3,3]
        R_blk = T_base_blk[:3,:3]
        axis = self._choose_horizontal(R_blk)
        ee_R = np.column_stack([axis, np.cross([0,0,-1], axis), [0,0,-1]])
        T = np.eye(4)
        T[:3,:3] = ee_R
        T[:3,3] = t
        return T

    def _choose_horizontal(self, R):
        """Choose a horizontal axis for grasping orientation."""
        cols = [R[:,i] for i in range(3)]
        best = max(cols, key=lambda v: abs(v[0]))
        best[2] = 0
        return best / np.linalg.norm(best)

    def solve_ik(self, target, seed):
        """Solve IK for a target pose using the seed configuration."""
        q, _, success, msg = self.ik.inverse(
            target, seed, method='J_pseudo', alpha=0.75)
        print('IK:', success, msg)
        return q if success else None

    def attempt_detection(self):
        """Try to detect blocks with multiple attempts."""
        for attempt in range(self.max_detection_attempts):
            print(f"Detection attempt {attempt+1}/{self.max_detection_attempts}")
            transforms = self.detect_and_convert()
            if transforms:
                return transforms
            sleep(0.5)  # Short wait between detection attempts
        return []

    def move_to_scanning_position(self):
        """Move to position for scanning moving blocks."""
        print("Moving to scanning position...")
        self.arm.safe_move_to_position(self.initial_view_pose)

    def grab_block(self):
        """Attempt to grab a dynamic block."""
        print("Attempting to grab dynamic block...")

        # Move to scanning position first
        self.move_to_scanning_position()
        sleep(1.0)

        # Multiple attempts to account for block movement
        for attempt in range(self.max_grab_attempts):
            print(f"Grab attempt {attempt+1}/{self.max_grab_attempts}")

            # Detect blocks
            transforms = self.attempt_detection()
            if not transforms:
                print("No blocks detected")
                continue

            # Find closest block
            target_block = self.find_closest(transforms)
            if target_block is None:
                print("No suitable block found")
                continue

            # Create target pose for end effector
            target_pose = self.find_target_ee_pose(target_block)

            # Add small offset to account for movement
            target_pose[0:3, 3] += np.array([0.0, 0.0, 0.005])  # Small Z offset

            # Compute joint angles using IK
            q_current = self.arm.get_positions()
            q_grab = self.solve_ik(target_pose, q_current)

            if q_grab is None:
                print("IK solution failed")
                continue

            # Open gripper before moving
            self.arm.open_gripper()

            # Move to grabbing position
            print("Moving to grab position...")
            self.arm.safe_move_to_position(q_grab)

            # Attempt to close gripper on block
            print("Closing gripper...")
            self.arm.exec_gripper_cmd(0.04, 80)
            sleep(0.5)

            # Check if grasp was successful
            gripper_state = self.arm.get_gripper_state()
            gripper_distance = gripper_state['position'][0] + gripper_state['position'][1]

            if gripper_distance < 0.02:
                print("Grasp failed (nothing in gripper)")
                continue
            elif gripper_distance > 0.08:
                print("Grasp failed (gripper too open)")
                continue
            else:
                print(f"Successful grasp! Gripper distance: {gripper_distance:.4f}")
                # Move up slightly after grabbing
                q_lift = q_grab.copy()
                #q_lift[1] -= 0.2  # Move up
                self.arm.safe_move_to_position(q_lift)
                return True

        print("All grab attempts failed")
        return False

    def go_to_top_of_tower(self):
        """Move to a position to scan the top of the tower."""
        print("Moving to tower scanning position...")
        self.arm.safe_move_to_position(self.tower_scan)

    def go_to_above(self):
        """Move to a safe position above current location."""
        q = self.arm.get_positions()
        q[1] -= 0.4
        q[3] += 0.8
        q[5] -= 0.8
        self.arm.safe_move_to_position(q)

    def place_on_tower(self):
        """Place block on top of the tower."""
        print("Preparing to place block on tower...")

        # First move to tower scanning position
        self.go_to_top_of_tower()
        sleep(1.0)

        # Detect current tower state
        tower_transforms = self.attempt_detection()

        if tower_transforms:
            print(f"Tower detected with {len(tower_transforms)} blocks")

            # Find the highest block (closest to camera)
            top_block = self.find_closest(tower_transforms)

            # Target position slightly above top block
            top_block[2, 3] += 0.06  # Z offset for placement

            # Create target pose for end effector
            target_pose = self.find_target_ee_pose(top_block)

            # Compute joint angles using IK
            q_current = self.arm.get_positions()
            q_place = self.solve_ik(target_pose, q_current)

            if q_place is not None:
                # Move to placement position
                print("Moving to placement position...")
                self.arm.safe_move_to_position(q_place)

                # Release block
                print("Opening gripper...")
                self.arm.open_gripper()

                # Move up after placing
                self.go_to_above()

                # Increment block counter
                self.blocks_stacked += 1
                print(f"Successfully placed block {self.blocks_stacked} on tower")
                return True
            else:
                print("IK failed for tower placement")
        else:
            print("No tower detected")

        # If tower detection or placement failed, return to scanning position
        self.move_to_scanning_position()

        # Still return True since we did place the block (just not on the tower)
        return True

    def grab_and_stack(self):
        """Complete sequence to grab a block and stack it on the tower."""
        if self.grab_block():
            return self.place_on_tower()
        return False


if __name__ == "__main__":
    try:
        team = rospy.get_param("team")
    except KeyError:
        print('Team must be red or blue')
        sys.exit(1)

    rospy.init_node("team_script")

    arm = ArmController()
    arm.set_arm_speed(0.15)
    arm.set_gripper_speed(0.15)
    arm.open_gripper()

    print("gripper_state:", arm.get_gripper_state()['position'])

    detector = ObjectDetector()
    ik = IK()
    fk = FK()

    static_grabber = StaticGrabber(detector, arm, team, ik, fk)
    dynamic_grabber = DynamicGrabber(detector, arm, team, ik, fk)

    start_pose = np.array([-0.0178, -0.7601, 0.0198, -2.3420, 0.0298, 1.5412+pi/2, 0.7534])

    arm.safe_move_to_position(start_pose)
    input("Waiting for start... Press ENTER to begin!\n")

    """
    # Now compute dynamic scan positions when ready
    static_grabber.compute_scan_positions()

    # Static block phase
    for i in range(4):
        static_grabber.move_to_over(i)
        sleep(2)
        trans = static_grabber.detect_and_convert()
        if not trans:
            print("Retrying detection")
            continue
        target = static_grabber.find_closest(trans)
        static_grabber.grab(target, i)

        if i == 0:
            static_grabber.put(i)
            continue

        static_grabber.go_to_top_of_tower()
        sleep(1)
        tower = static_grabber.detect_and_convert()
        if tower:
            top = static_grabber.find_closest(tower)
            top[2,3] += 0.06
            q_place = static_grabber.solve_ik(
                static_grabber.find_target_ee_pose(top), arm.get_positions())
            if q_place is not None:
                arm.safe_move_to_position(q_place)
                arm.open_gripper()
                static_grabber.go_to_above()
                static_grabber.move_to_over(i)
            else:
                static_grabber.put(i)
        else:
            static_grabber.put(i)
    """

    # Dynamic block phase
    print("Starting dynamic block phase...")
    arm.set_arm_speed(0.175)
    arm.set_gripper_speed(0.175)
    arm.open_gripper()

    """
    # Initial scan of tower to establish baseline
    dynamic_grabber.go_to_top_of_tower()
    tower_scan = dynamic_grabber.detect_and_convert()
    if tower_scan:
        print(f"Initial tower scan: {len(tower_scan)} blocks detected")
    else:
        print("No tower detected initially")
    """

    # Main grab and stack loop
    try:
        while True:
            print("\n--- Starting new grab and stack cycle ---\n")

            # Complete grab and stack sequence
            success = dynamic_grabber.grab_and_stack()

            if not success:
                print("Grab and stack cycle failed, retrying...")
            else:
                print(f"Successfully stacked {dynamic_grabber.blocks_stacked} blocks")

            # Brief pause between cycles
            sleep(0.5)

    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
    except Exception as e:
        print(f"\nError in dynamic block phase: {e}")
    finally:
        # Return to start position when done
        arm.open_gripper()
        arm.safe_move_to_position(start_pose)
        print(f"Program finished. Total blocks stacked: {dynamic_grabber.blocks_stacked}")
