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
    Example dynamic‐grasp class that continuously tracks a moving block
    and adjusts the arm’s position in ‘real time’ until the block can be
    reliably grasped.
    """
    def __init__(self, detector, arm, team, ik, fk):
        self.detector = detector
        self.arm = arm
        self.team = team
        self.ik = ik
        self.fk = fk
        self.start_time = 0
        self.last_iteration_time = None

        if team == 'blue':
            # Positions to scan for blocks (blue team)
            self.over_blk = np.array([
                [-0.83, -0.93, 1.72, -1.14, 0.93, 1.43, -1.18],
            ])
        else:
            self.over_blk = np.array([
                [0.84, -0.88, -1.8, -1.15, -0.84, 1.47, -0.42],
            ])


        # Initial pose to view all static blocks
        if team == 'red':
            self.initial_view_pose = np.array([-0.83, -0.93, 1.72, -1.14, 0.93, 1.43, -1.18])
        else:
            # UPDATE THIS: INCORRECT
            self.initial_view_pose = np.array([0.84, -0.88, -1.8, -1.15, -0.84, 1.47, -0.42],)

        # Placeholder for dynamic scan positions; computed on demand
        self.dynamic_over_positions = []
        self.dynamic_over_positions.append(self.initial_view_pose)

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

    def solve_ik_grab(self, target, seed):
        print("TARGET: ", target)
        # t = time_in_seconds()
        # T = t - self.start_time
        # print("Time: ", T, t, self.start_time)

        # if T == 0:
        #     T = 1
        T = 5.0
        
        print("seed: ", seed)
        _, T0e = fk.forward(seed)

        # Current joint positions
        R = T0e[:3, :3]
        x = T0e[:3, 3]
        curr_x = np.copy(x.flatten())

        q = seed

        xdes = target[:3, 3].flatten() + np.array([-0.05, -0.03, 0.05]) # destination offset [-0.005, -0.03, 0]) left -y down -z in -x
        delta_x = xdes - curr_x
        print("delta: ", delta_x, xdes, curr_x, T),
        vdes = delta_x / T
        print("vdes: " , vdes)

        Rdes, ang_vdes = np.eye(3), np.zeros(3)

        # Proportional & feed-forward for position
        kp = 1.0  # reduced from 5.0
        v = vdes + kp * (delta_x)

        # Proportional & feed-forward for orientation
        kr = 1.0  # reduced from 5.0
        omega = ang_vdes + kr * calcAngDiff(Rdes, R).flatten()

        # Null-space secondary task
        lower = np.array([-2.8973, -1.7628, -2.8973,
                            -3.0718, -2.8973, -0.0175, -2.8973])
        upper = np.array([ 2.8973,  1.7628,  2.8973,
                            -0.0698,  2.8973,  3.7525,  2.8973])

        # Set q_e to midrange or some comfortable posture
        q_e = lower + (upper - lower)/2.0
        k0 = 1.0  # moderate null-space gain

        # Velocity IK with null-space
        dq = IK_velocity_null(q, v, omega, -k0*(q - q_e)).flatten()
        print("dq: ", dq)
        # Time step
        if self.last_iteration_time is None:
            self.last_iteration_time = time_in_seconds()
        # self.dt = time_in_seconds() - self.last_iteration_time
        self.last_iteration_time = time_in_seconds()

        # 
        new_q = q + dq
        print(new_q)

        return new_q

    #     # Command new pose & velocity
    #     arm.safe_set_joint_positions_velocities(new_q, dq)

    #     # Downsample viz
    #     self.counter += 1
    #     if self.counter == 10:
    #         self.show_ee_position()
    #         self.counter = 0

    # except rospy.exceptions.ROSException:
    #     pass






        # q, _, success, msg = self.ik.IK_velocity_null(
        #     target, seed,)
        # print('IK:', success, msg)
        # return q if success else None
    def solve_ik_place(self, target, seed):
        q, _, success, msg = self.ik.inverse(
            target, seed, method='J_pseudo', alpha=0.75)
        print('IK:', success, msg)
        return q if success else None

    def grab(self, T_base_blk, idx):
        target = self.find_target_ee_pose(T_base_blk)
        q = self.solve_ik_grab(target, self.arm.get_positions())
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
        #safe[3] += 0.8
        #safe[5] -= 0.8
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
        """
        if self.team == 'blue':
            self.H_ee_camera[0,-1] -= 0.035
            self.H_ee_camera[1,-1] += 0.005
            self.H_ee_camera[2,-1] += 0.015
        else:
            self.H_ee_camera[0,-1] += 0.01
            self.H_ee_camera[1,-1] -= 0.036
            self.H_ee_camera[2,-1] += 0.008
        """


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

    dynamic_grabber.go_to_top_of_tower()
    # Scan the tower before starting dynamic block grabbing
    dynamic_grabber.go_to_top_of_tower()
    sleep(1)
    tower_scan = dynamic_grabber.detect_and_convert()
    if tower_scan:
        print(f"Tower detected with {len(tower_scan)} blocks")
    else:
        print("No tower detected")

    # Dynamic block phase
    print("Starting dynamic block phase...")
    arm.set_arm_speed(0.175)
    arm.set_gripper_speed(0.175)
    arm.open_gripper()

    # Move to initial position
    arm.safe_move_to_position(dynamic_grabber.initial_view_pose)
    #arm.safe_move_to_position(start_pose)
    dynamic_grabber.start_time = time_in_seconds()

    try:
        # Run continuously until interrupted
        block_count = 0
        while True:
            print(f"Attempting to grab dynamic block {block_count}")
            sleep(1)  # Brief pause between grab attempts
            
            # Detect blocks
            trans = dynamic_grabber.detect_and_convert()
            if not trans:
                print("No dynamic blocks detected, retrying...")
                continue
                
            # Find closest block to grab
            target = dynamic_grabber.find_closest(trans)
            
            # Try to grab the block
            dynamic_grabber.grab(target, 0)
            
            # After successful grab, scan the tower again to find where to place
            dynamic_grabber.go_to_top_of_tower()
            sleep(1)
            tower = dynamic_grabber.detect_and_convert()
            
            if tower:
                print("Tower detected, placing block on top")
                top = dynamic_grabber.find_closest(tower)
                top[2,3] += 0.06  # Position slightly above tower top
                q_place = dynamic_grabber.solve_ik_place(
                    dynamic_grabber.find_target_ee_pose(top), arm.get_positions())
                if q_place is not None:
                    arm.safe_move_to_position(q_place)
                    arm.open_gripper()
                    dynamic_grabber.go_to_above()
                else:
                    print("IK failed for tower placement, using default position")
                    dynamic_grabber.put(0)  # Use default placement position
            else:
                print("No tower detected, using default position")
                dynamic_grabber.put(0)  # Use default placement position
            

            # Reset start time for velocity calculations
            dynamic_grabber.start_time = time_in_seconds()
            # dynamic_grabber.last_iteration_time = None

            # Return to scanning position
            dynamic_grabber.move_to_over(0)
            block_count += 1
            
    except KeyboardInterrupt:
        print("Dynamic block phase interrupted by user")
    except Exception as e:
        print(f"Error in dynamic block phase: {e}")
    finally:
        # Return to start position when done
        arm.open_gripper()
        arm.safe_move_to_position(start_pose)
        print("Done!")
