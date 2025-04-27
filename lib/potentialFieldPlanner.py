import os
import time
import pandas as pd
import numpy as np
from math import pi, acos
from scipy.linalg import null_space
from copy import deepcopy
from lib.calculateFKJac import FK_Jac
from lib.detectCollision import detectCollision
from lib.loadmap import loadmap


class PotentialFieldPlanner:

    # Parameters
    # Joint limits (from rrt.py)
    lowerLim = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
    upperLim = np.array([ 2.8973,  1.7628,  2.8973, -0.0698,  2.8973,  3.7525,  2.8973])
    # Planner parameters
    max_steps = 1000 # Max iterations
    alpha = 0.02    # Learning rate (step size multiplier) - Recommended value
    tol = 0.1       # Tolerance for reaching goal (radians)
    min_step_size = 1e-4 # Threshold for detecting being stuck

    center = lowerLim + (upperLim - lowerLim) / 2 # compute middle of range of motion of each joint
    fk = FK_Jac()

    def __init__(self):
        # Initialize Forward Kinematics (expanded version)
        self.fk = FK_Jac()

    ######################
    ## Helper Functions ##
    ######################
    # The following functions are provided to you to help you to better structure your code
    # You don't necessarily have to use them. You can also edit them to fit your own situation 

    @staticmethod
    def attractive_force(target, current, zeta=30.0, d_thresh=0.12):
        """
        Helper function for computing the attractive force using a hybrid potential field.
        Uses a parabolic well (linear force) when close to the target and a 
        conic well (constant magnitude force) when far away.

        INPUTS:
        target - 3x1 numpy array representing the desired joint position in the world frame
        current - 3x1 numpy array representing the current joint position in the world frame
        zeta - attractive force scaling factor (float)
        d_thresh - distance threshold for switching between parabolic and conic potential (float)

        OUTPUTS:
        att_f - 3x1 numpy array representing the force vector that pulls the joint 
        from the current position to the target position 
        """

        ## STUDENT CODE STARTS HERE

        # Ensure inputs are numpy arrays (the planner might pass lists or different shapes)
        target = np.asarray(target).reshape(3, 1)
        current = np.asarray(current).reshape(3, 1)

        # Calculate vector and distance from current to target
        diff_vec = target - current
        dist = np.linalg.norm(diff_vec)

        att_f = np.zeros((3, 1)) # Initialize force

        # Avoid division by zero if already at target
        if dist == 0:
            return att_f 

        # Check if within the parabolic region
        if dist <= d_thresh:
            # Parabolic well: F = zeta * (target - current)
            att_f = zeta * diff_vec
        else:
            # Conic well: F = constant_magnitude * unit_vector(target - current)
            # Ensure continuity: Magnitude should match parabolic force at d_thresh.
            magnitude = zeta * d_thresh 
            unit_vec = diff_vec / dist
            att_f = magnitude * unit_vec
            # Previous implementation used zeta * unit_vec, which was likely incorrect.

        ## END STUDENT CODE

        return att_f

    @staticmethod
    def repulsive_force(obstacle, current, unitvec=np.zeros((3,1))):
        """
        Helper function for computing the repulsive force between the current position
        of one joint and one obstacle. Computes the repulsive force vector between the 
        obstacle and the current joint position 

        INPUTS:
        obstacle - 1x6 numpy array representing the an obstacle box in the world frame
        current - 3x1 numpy array representing the current joint position in the world frame
        unitvec - 3x1 numpy array representing the unit vector from the current joint position 
        to the closest point on the obstacle box (Note: This implementation recalculates it)

        OUTPUTS:
        rep_f - 3x1 numpy array representing the force vector that pushes the joint 
        from the obstacle
        """

        ## STUDENT CODE STARTS HERE
        # Parameters (tune these as needed)
        eta = 0.001 # Repulsive force gain (Reset to recommended value)
        rho_0 = 0.12 # Distance of influence (Reset to recommended value)

        # Ensure current is a numpy array for dist_point2box
        current_point = np.asarray(current).reshape(1, 3) # dist_point2box expects nx3

        # Calculate distance (d) and unit vector (grad) from point to box
        d, grad = PotentialFieldPlanner.dist_point2box(current_point, obstacle)
        d = d[0]       # Extract scalar distance
        grad = grad[0] # Extract 1x3 vector

        rep_f = np.zeros((3, 1)) # Initialize force to zero

        # If point is outside, within influence distance, and not on the boundary
        if 0 < d <= rho_0:
            # Calculate magnitude of repulsive force
            force_mag = eta * (1/d - 1/rho_0) * (1 / d**2)
            
            # Force direction is opposite to the gradient (points away from obstacle)
            force_direction = -grad 
            
            # Ensure force_direction is a column vector for broadcasting if needed
            # Although in this case grad is 1x3, making -grad 1x3, we reshape for consistency
            rep_f = (force_mag * force_direction).reshape(3, 1)
        
        # If d=0 (inside/on boundary) or d > rho_0 (too far), rep_f remains [0,0,0]

        ## END STUDENT CODE

        return rep_f

    @staticmethod
    def dist_point2box(p, box):
        """
        Helper function for the computation of repulsive forces. Computes the closest point
        on the box to a given point 
    
        INPUTS:
        p - nx3 numpy array of points [x,y,z]
        box - 1x6 numpy array of minimum and maximum points of box

        OUTPUTS:
        dist - nx1 numpy array of distance between the points and the box
                dist > 0 point outside
                dist = 0 point is on or inside box
        unit - nx3 numpy array where each row is the corresponding unit vector 
        from the point to the closest spot on the box
            norm(unit) = 1 point is outside the box
            norm(unit)= 0 point is on/inside the box

         Method from MultiRRomero
         @ https://stackoverflow.com/questions/5254838/
         calculating-distance-between-a-point-and-a-rectangular-box-nearest-point
        """
        # THIS FUNCTION HAS BEEN FULLY IMPLEMENTED FOR YOU

        # Get box info
        boxMin = np.array([box[0], box[1], box[2]])
        boxMax = np.array([box[3], box[4], box[5]])
        boxCenter = boxMin*0.5 + boxMax*0.5
        p = np.array(p)

        # Get distance info from point to box boundary
        dx = np.amax(np.vstack([boxMin[0] - p[:, 0], p[:, 0] - boxMax[0], np.zeros(p[:, 0].shape)]).T, 1)
        dy = np.amax(np.vstack([boxMin[1] - p[:, 1], p[:, 1] - boxMax[1], np.zeros(p[:, 1].shape)]).T, 1)
        dz = np.amax(np.vstack([boxMin[2] - p[:, 2], p[:, 2] - boxMax[2], np.zeros(p[:, 2].shape)]).T, 1)

        # convert to distance
        distances = np.vstack([dx, dy, dz]).T
        dist = np.linalg.norm(distances, axis=1)

        # Figure out the signs
        signs = np.sign(boxCenter-p)

        # Calculate unit vector and replace with
        unit = distances / dist[:, np.newaxis] * signs
        unit[np.isnan(unit)] = 0
        unit[np.isinf(unit)] = 0
        return dist, unit

    @staticmethod
    def compute_forces(target, obstacle, current):
        """
        Helper function for the computation of forces on every joints. Computes the sum 
        of forces (attactive, repulsive) on each joint. 

        INPUTS:
        target - 3x10 numpy array representing the desired joint/end effector positions 
        in the world frame
        obstacle - nx6 numpy array representing the obstacle box min and max positions
        in the world frame
        current- 3x10 numpy array representing the current joint/end effector positions 
        in the world frame

        OUTPUTS:
        joint_forces - 3x10 numpy array representing the force vectors on each 
        joint/end effector
        """

        ## STUDENT CODE STARTS HERE

        num_points = current.shape[1] # Should be 10
        joint_forces = np.zeros((3, num_points)) 

        # Define attractive parameters based on recommended values
        zeta_physical = 30.0
        zeta_virtual = 12.5 # Using average of 10-15
        d_thresh = 0.12

        # Iterate through each point (0-6 physical joints, 7 EE, 8-9 virtual)
        for i in range(num_points):
            current_pos = current[:, i:i+1] # Ensure column vector (3x1)
            target_pos = target[:, i:i+1]   # Ensure column vector (3x1)

            # Determine attractive force gain zeta based on point index
            if i <= 6:
                zeta = zeta_physical
            else: # Points 7, 8, 9 (EE and virtual)
                zeta = zeta_virtual

            # Calculate attractive force
            F_attr = PotentialFieldPlanner.attractive_force(target_pos, current_pos, zeta, d_thresh)

            # Calculate total repulsive force from all obstacles
            F_rep_total = np.zeros((3, 1))
            if obstacle.size > 0: # Check if there are any obstacles
                for obs_j in obstacle: # Iterate through each obstacle row
                    F_rep_j = PotentialFieldPlanner.repulsive_force(obs_j, current_pos)
                    F_rep_total += F_rep_j
            
            # Calculate total force for joint i
            total_force = F_attr + F_rep_total
            
            # Store total force
            joint_forces[:, i:i+1] = total_force

        ## END STUDENT CODE

        return joint_forces
    
    @staticmethod
    def compute_torques(joint_forces, q):
        """
        Helper function for converting joint forces to joint torques. Computes the sum 
        of torques on each actuated joint using the Jacobian transpose method.

        INPUTS:
        joint_forces - 3x10 numpy array representing the force vectors on each point
                       (0-6: physical joints, 7: EE, 8-9: virtual points)
        q - 1x7 numpy array representing the current joint angles

        OUTPUTS:
        joint_torques - 1x7 numpy array representing the torques on the 7 actuated joints
        """

        ## STUDENT CODE STARTS HERE
        
        # Initialize FK calculator
        fk = FK_Jac()

        # Get current joint positions and transformations
        # jointPositions is 10x3, T0e is 10x4x4
        jointPositions, T0e = fk.forward_expanded(q)

        total_torque = np.zeros((7, 1)) # Torque vector for 7 actuated joints

        num_points = joint_forces.shape[1] # Should be 10
        num_joints = q.shape[0] # Should be 7

        # Loop through each point i where force F_i is applied (i = 0 to 9)
        for i in range(num_points):
            F_i = joint_forces[:, i:i+1] # Force at point i (3x1)
            p_i = jointPositions[i, :].reshape(3, 1) # Position of point i (3x1)
            
            # Compute the Jacobian J_vi for point i (3x7)
            J_vi = np.zeros((3, num_joints))
            
            # Loop through each actuated joint j (j = 0 to 6)
            for j in range(num_joints):
                if j == 0:
                    # Base joint (Joint 0)
                    Z_j = np.array([0, 0, 1])
                    p_j = np.array([0, 0, 0])
                else:
                    # Joints 1 through 6
                    # T0e[k] is T0_{k+1}. So T0_j is T0e[j-1]
                    T0_j = T0e[j-1]
                    Z_j = T0_j[0:3, 2] # Z-axis of frame {j} in world frame {0}
                    p_j = T0_j[0:3, 3] # Origin of frame {j} in world frame {0}

                # Calculate Jacobian column: Z_j x (p_i - p_j)
                # Ensure vectors are flat for cross product, then reshape result
                col_j = np.cross(Z_j.flatten(), (p_i - p_j.reshape(3,1)).flatten())
                J_vi[:, j] = col_j.flatten()
                
            # Calculate torque contribution from F_i: tau_i = J_vi^T * F_i
            tau_i = J_vi.T @ F_i
            
            # Add to total torque
            total_torque += tau_i
            
        joint_torques = total_torque.reshape(1, 7)

        ## END STUDENT CODE

        return joint_torques

    @staticmethod
    def q_distance(target, current):
        """
        Helper function which computes the distance between any two
        vectors.

        This data can be used to decide whether two joint sets can be
        considered equal within a certain tolerance.

        INPUTS:
        target - 1x7 numpy array representing some joint angles
        current - 1x7 numpy array representing some joint angles

        OUTPUTS:
        distance - the distance between the target and the current joint sets 

        """

        ## STUDENT CODE STARTS HERE
        
        # Ensure inputs are numpy arrays
        target = np.asarray(target)
        current = np.asarray(current)
        
        # Calculate the difference vector
        diff = target - current
        
        # Compute the Euclidean distance (L2 norm)
        distance = np.linalg.norm(diff)

        ## END STUDENT CODE

        return distance
    
    @staticmethod
    def compute_gradient(q, target, map_struct):
        """
        Computes the joint gradient step (direction) based on potential field forces.

        INPUTS:
        q - 1x7 numpy array. the current joint configuration.
        target - 1x7 numpy array containing the desired final joint angles.
        map_struct - A map struct containing obstacle information (e.g., map_struct.obstacles)

        OUTPUTS:
        dq - 1x7 numpy array. A unit vector representing the desired joint velocity direction.
        """

        ## STUDENT CODE STARTS HERE
        
        # Initialize FK calculator
        fk = FK_Jac()
        
        # 1. Calculate current Cartesian positions (for repulsive forces)
        current_positions_xyz, _ = fk.forward_expanded(q) # Result is 10x3
        current_positions_3x10 = current_positions_xyz.T # Transpose to 3x10
        
        # 2. Calculate target Cartesian positions (for attractive forces)
        # Assuming 'target' input is final joint angles
        target_positions_xyz, _ = fk.forward_expanded(target) # Result is 10x3
        target_positions_3x10 = target_positions_xyz.T # Transpose to 3x10

        # 3. Get obstacles from map_struct
        obstacles = map_struct.obstacles # Assuming this structure
        
        # 4. Compute 3x10 forces based on current/target Cartesian positions
        joint_forces = PotentialFieldPlanner.compute_forces(target_positions_3x10, obstacles, current_positions_3x10)
        
        # 5. Compute 1x7 joint torques needed to produce these forces
        # Note: compute_torques internally calls fk.forward_expanded(q) again,
        # which is slightly inefficient but keeps functions modular.
        joint_torques_1x7 = PotentialFieldPlanner.compute_torques(joint_forces, q)

        # 6. Normalize the torque vector to get the gradient direction dq
        torque_norm = np.linalg.norm(joint_torques_1x7)
        
        dq = np.zeros((1, 7))
        if torque_norm > 1e-6: # Avoid division by zero/near-zero
            dq = joint_torques_1x7 / torque_norm
            
        ## END STUDENT CODE

        return dq.flatten() # Return as 1D array (7,)

    ###############################
    ### Potential Feild Solver  ###
    ###############################

    def plan(self, map_struct, start, goal):
        """
        Uses potential field to move the Panda robot arm from the starting configuration to
        the goal configuration.

        INPUTS:
        map_struct - A map struct containing obstacle information (e.g., map_struct.obstacles)
        start - 1x7 numpy array representing the starting joint angles for a configuration 
        goal - 1x7 numpy array representing the desired joint angles for a configuration

        OUTPUTS:
        q_path - nx7 numpy array of joint angles [q0, q1, q2, q3, q4, q5, q6]. This should contain
        all the joint angles throughout the path of the planner. The first row of q should be
        the starting joint angles and the last row of q should be the goal joint angles.
        Returns empty array if start/goal is invalid or path not found within max_steps.
        """
        
        # Ensure inputs are numpy arrays and flattened to (7,)
        start = np.asarray(start).flatten()
        goal = np.asarray(goal).flatten()
        obstacles = map_struct.obstacles

        # Initial checks for start/goal validity
        if not (np.all(start >= self.lowerLim) and np.all(start <= self.upperLim)):
            print("Start configuration out of limits.")
            return np.array([]).reshape(0,7)
        if not (np.all(goal >= self.lowerLim) and np.all(goal <= self.upperLim)):
            print("Goal configuration out of limits.")
            return np.array([]).reshape(0,7)
        if self._check_collision_segment(start, start, obstacles): # Check start collision
             print("Start configuration is in collision.")
             return np.array([]).reshape(0,7)
        if self._check_collision_segment(goal, goal, obstacles):   # Check goal collision
             print("Goal configuration is in collision.")
             return np.array([]).reshape(0,7)

        # Check for immediate direct path
        print("Checking direct path from start to goal...")
        if not self._check_collision_segment(start, goal, obstacles):
            print("Direct path found immediately!")
            return np.array([start, goal])
        else:
            print("Direct path is blocked. Starting planner...")

        # Initialize path
        q_path = [start.copy()] # Start with the start configuration
        q_curr = start.copy()

        print("Starting Potential Field Planner...")
        # Main planning loop
        for step in range(self.max_steps):
            
            # Check if goal reached
            dist_to_goal = self.q_distance(q_curr, goal)
            if dist_to_goal < self.tol:
                print(f"Goal reached within tolerance {self.tol} at step {step}.")
                q_path.append(goal.copy()) # Ensure goal is the last point
                return np.array(q_path)

            # Compute gradient direction
            dq = self.compute_gradient(q_curr, goal, map_struct)

            # Compute step
            q_step = self.alpha * dq
            step_norm = np.linalg.norm(q_step)

            # Check for local minimum / being stuck
            if step_norm < self.min_step_size:
                print(f"Step size ({step_norm:.2e}) below minimum ({self.min_step_size:.2e}) at step {step}.")
                # Check if actually at goal
                if dist_to_goal < self.tol:
                    print("Confirmed at goal.")
                    q_path.append(goal.copy())
                    return np.array(q_path)
                else:
                    # Not at goal, but step is tiny. Try direct connection.
                    print("Step size too small. Checking direct path to goal...")
                    if not self._check_collision_segment(q_curr, goal, obstacles):
                        print("Direct path to goal is collision-free! Appending goal.")
                        q_path.append(goal.copy())
                        return np.array(q_path)
                    else:
                        # Direct path failed, now try random walk
                        print("Direct path to goal collided. Attempting random walk...")
                        # --- Random Walk Logic --- 
                        found_escape_step = False
                        random_walk_attempts = 20
                        random_walk_step_size = 2.5 # Increased step size significantly (Radians)

                        for attempt in range(random_walk_attempts):
                            # Generate random direction
                            rand_dir = np.random.randn(7) # Random 7D vector
                            norm = np.linalg.norm(rand_dir)
                            if norm > 1e-8:
                                rand_dir = rand_dir / norm # Normalize
                            else:
                                continue # Skip if zero vector

                            # Calculate and clip random step
                            q_rand_next_raw = q_curr + random_walk_step_size * rand_dir
                            q_rand_next = np.clip(q_rand_next_raw, self.lowerLim, self.upperLim)
                            
                            # Check collision for the random step
                            if not self._check_collision_segment(q_curr, q_rand_next, obstacles):
                                # Found a valid escape step
                                print(f"  Random walk step successful (attempt {attempt+1}).")
                                q_curr = q_rand_next # Update current position
                                q_path.append(q_curr.copy()) # Add escaped position to path
                                found_escape_step = True
                                break # Exit random walk attempts loop
                            # else: 
                                # print(f"  Random walk attempt {attempt+1} collided.")
                                
                        # If random walk failed after all attempts
                        if not found_escape_step:
                            print(f"Random walk failed after {random_walk_attempts} attempts. Checking direct path to goal as last resort...")
                            # Check direct path one last time
                            if not self._check_collision_segment(q_curr, goal, obstacles):
                                print("Direct path to goal is collision-free! Appending goal.")
                                q_path.append(goal.copy())
                                return np.array(q_path)
                            else:
                                 print("Direct path also collided. Planner truly stuck.")
                                 return np.array(q_path) # Return path found so far
                        
                        # If random walk succeeded, continue to next main loop iteration
                        continue # Go compute gradient from the new q_curr
                        # ------------------------- 

            # Calculate next potential configuration
            # q_curr is (7,), dq is (7,), alpha is scalar -> q_next_raw is (7,)
            q_next_raw = q_curr + q_step

            # Clamp to joint limits
            q_next = np.clip(q_next_raw, self.lowerLim, self.upperLim)

            # --- Collision Handling with Sub-Stepping ---
            took_step = False # Flag to track if any step (full or sub) was taken
            if self._check_collision_segment(q_curr, q_next, obstacles):
                # Full step collides, try smaller steps
                print(f"Collision detected for step {step}. Trying sub-steps...")
                step_fractions = [0.5, 0.25, 0.125] # Try progressively smaller steps
                for fraction in step_fractions:
                    q_step_sub = fraction * q_step
                    q_next_sub_raw = q_curr + q_step_sub
                    q_next_sub = np.clip(q_next_sub_raw, self.lowerLim, self.upperLim)
                    
                    if not self._check_collision_segment(q_curr, q_next_sub, obstacles):
                        # Found a valid smaller step
                        print(f"  Sub-step ({fraction*100:.1f}%) successful.")
                        q_curr = q_next_sub # Update current position
                        q_path.append(q_curr.copy()) # Add accepted position to path
                        took_step = True
                        break # Exit sub-stepping loop
                
                if not took_step:
                    print(f"  All sub-steps failed. Skipping gradient update for step {step}.")
                    # continue # Skip to next main loop iteration if no step was taken
                    # Alternatively, we could trigger random walk here immediately if desired
            else:
                # Full step is collision-free
                q_curr = q_next # Update current configuration
                q_path.append(q_curr.copy()) # Add accepted position to path
                took_step = True
            # ---------------------------------------------
            
            # Optional: Print progress (Only print if a step was actually taken)
            # if took_step and step % 50 == 0: 
            #     dist_to_goal = self.q_distance(q_curr, goal) # Recalculate for print
            #     print(f"Step: {step}, Dist to Goal: {dist_to_goal:.4f}")

        print(f"Max steps ({self.max_steps}) reached.")
        
        # Check distance one last time
        final_dist = self.q_distance(q_curr, goal)
        if final_dist < self.tol:
            print(f"Final position within tolerance ({final_dist:.4f}).")
        else:
             print(f"Final position error ({final_dist:.4f}) outside tolerance {self.tol}.")
        
        # Force append goal, trusting simulation might handle the last step
        print("Forcing goal configuration onto the end of the path.")
        if np.linalg.norm(q_path[-1] - goal) > 1e-6: # Avoid duplicate goal if already there
             q_path.append(goal.copy())
            
        return np.array(q_path)

    def _check_collision_segment(self, q1, q2, obstacles):
        """
        Checks if the path segment between q1 and q2 is collision-free.
        Uses FK to get joint positions and detectCollision for checks.
        Only checks the 7 physical links (points 0-7) like in RRT.
        Returns True if collision IS detected, False otherwise.
        """
        dist = np.linalg.norm(q2 - q1)
        # Reduce sensitivity: Check less frequently (e.g., every ~0.15 rad)
        steps = max(int(dist / 0.15), 2) 
        for t in np.linspace(0, 1, steps):
            q = q1 + t * (q2 - q1)
            if not (np.all(q >= self.lowerLim) and np.all(q <= self.upperLim)):
                return True # Collision with joint limits
                
            # Get positions of the 7 physical joints + EE origin (Points 0-7)
            # forward_expanded returns 10x3 positions
            try:
                # Ensure q is 1D (7,) before passing to FK
                joint_positions_all, _ = self.fk.forward_expanded(q.flatten())
                joint_positions = joint_positions_all[:8] # Select first 8 points (0-7)
            except Exception as e:
                print(f"FK Error during collision check at q={q}: {e}")
                return True # Treat FK error as collision

            # Check collisions between links (line segments) and obstacles
            # Links: 0->1, 1->2, ..., 6->7
            p1 = joint_positions[:-1] # Start points of links (0-6)
            p2 = joint_positions[1:]  # End points of links (1-7)
            
            for box in obstacles:
                if np.any(detectCollision(p1, p2, box)):
                    return True # Collision detected
                    
        return False # No collision found along segment

################################
## Simple Testing Environment ##
################################
"""
if __name__ == "__main__":

    np.set_printoptions(suppress=True,precision=5)

    planner = PotentialFieldPlanner()
    
    # inputs 
    map_struct = loadmap("../maps/map1.txt")
    start = np.array([0,-1,0,-2,0,1.57,0])
    goal =  np.arra    map_struct = loadmap("../maps/map1.txt")
    start = np.array([0,-1,0,-2,0,1.57,0])
    goal =  np.array([-1.2, 1.57 , 1.57, -2.07, -1.57, 1.57, 0.7])
    
    # potential field planning
    q_path = planner.plan(deepcopy(map_struct), deepcopy(start), deepcopy(goal))
    
    # show results
    for i in range(q_path.shape[0]):
        error = PotentialFieldPlanner.q_distance(q_path[i, :], goal)
        print('iteration:',i,' q =', q_path[i, :], ' error={error}'.format(error=error))

    print("q path: ", q_path)y([-1.2, 1.57 , 1.57, -2.07, -1.57, 1.57, 0.7])
    
    # potential field planning
    q_path = planner.plan(deepcopy(map_struct), deepcopy(start), deepcopy(goal))
    
    # show results
    for i in range(q_path.shape[0]):
        error = PotentialFieldPlanner.q_distance(q_path[i, :], goal)
        print('iteration:',i,' q =', q_path[i, :], ' error={error}'.format(error=error))

    print("q path: ", q_path)
"""

def evaluate_path(q_path, goal):
    if q_path.shape[0] == 0:
        return False, []
    
    errors = []
    for i in range(q_path.shape[0]):
        error = PotentialFieldPlanner.q_distance(q_path[i, :], goal)
        errors.append(error)
        print(f"iteration {i}: q = {q_path[i, :]}, error = {error:.5f}")

    return True, errors

if __name__ == "__main__":
    np.set_printoptions(suppress=True, precision=5)

    # Test maps and configs
    maps = [
        ("simple_open.txt", np.array([0, -1, 0, -2, 0, 1.57, 0]), np.array([-1.2, 1.57, 1.57, -2.07, -1.57, 1.57, 0.7])),
        ("single_block.txt", np.array([0, -1, 0, -2, 0, 1.57, 0]), np.array([-1.2, 1.57, 1.57, -2.07, -1.57, 1.57, 0.7])),
        ("narrow_passage.txt", np.array([0, -0.5, 0, -2, 0, 1.5, 0]), np.array([1.5, 0.5, 1.5, -2, -1.5, 1.5, 0.5])),
        ("u_shaped_trap.txt", np.array([0, 0, 0, -2, 0, 1.57, 0]), np.array([1.0, 0.0, 1.5, -2, -1.5, 1.0, 0.5])),
        ("maze_corridor.txt", np.array([0, 0, 0, -1.5, 0, 1.5, 0]), np.array([1.5, 0.5, 1.5, -2.0, -1.5, 1.5, 0.5])),
        ("cluttered_blocks.txt", np.array([0, -0.5, 0, -2, 0, 1.2, 0]), np.array([1.2, 0.5, 1.5, -2, -1.5, 1.5, 0.8])),
        ("vertical_wall.txt", np.array([0, 0, 0, -2, 0, 1.5, 0]), np.array([2.0, 0, 1.5, -2, -1.5, 1.5, 0.5])),
        ("small_exit.txt", np.array([0, 0, 0, -1.8, 0, 1.57, 0]), np.array([1.0, 0.0, 1.2, -2.0, -1.2, 1.0, 0.4]))
    ]

    # Number of trials
    num_trials = 5
    map_folder = "../maps/"

    planner = PotentialFieldPlanner()

    results = []

    for map_name, start, goal in maps:
        map_path = os.path.join(map_folder, map_name)
        map_struct = loadmap(map_path)

        for trial in range(1, num_trials + 1):
            print("\n==============================")
            print(f"Running Potential Fields Planner | Map: {map_name} | Trial {trial}")
            print("==============================")

            start_time = time.time()
            q_path = planner.plan(deepcopy(map_struct), deepcopy(start), deepcopy(goal))
            elapsed_time = time.time() - start_time

            success, errors = evaluate_path(q_path, goal)

            results.append({
                "Map": map_name,
                "Trial": trial,
                "Success": success,
                "Time (s)": elapsed_time,
                "Final Error": errors[-1] if errors else None,
                "Steps": len(errors)
            })

    # Save results
    df = pd.DataFrame(results)
    df.to_csv("planner_results.csv", index=False)
    print("\n✅ All tests completed. Results saved to planner_results.csv")