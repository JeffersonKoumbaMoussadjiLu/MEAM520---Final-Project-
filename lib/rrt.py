import os
import time
import pandas as pd
import numpy as np
import random
from lib.detectCollision import detectCollision
from lib.loadmap import loadmap
from copy import deepcopy
from lib.calculateFK import FK


fk = FK()


lowerLim = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
upperLim = np.array([ 2.8973,  1.7628,  2.8973, -0.0698,  2.8973,  3.7525,  2.8973])


class Node:
    def __init__(self, q, parent=None):
        self.q      = np.array(q, dtype=float)  # 7-vector
        self.parent = parent                    # another Node or None

def in_limits(q):
    return np.all(q >= lowerLim) and np.all(q <= upperLim)

def clip_limits(q):
    return np.clip(q, lowerLim, upperLim)

def checkCollision(q1, q2, obstacles):
    """
    Return True if ANY configuration along the segment from q1 to q2 collides.
    """
    dist  = np.linalg.norm(q2 - q1)
    steps = max(int(dist / 0.01), 2)
    for q in np.linspace(q1, q2, steps):
        if not in_limits(q):
            return True
        joints = fk.forward(q)[0]  # joint positions
        # check each link against every obstacle
        for box in obstacles:
            if np.any(detectCollision(joints[:-1], joints[1:], box)):
                return True
    return False

def build_path(goal_node):
    """
    Trace back from goal_node to the root via parent pointers,
    return an (N×7) numpy array of configurations.
    """
    path = []
    node = goal_node
    while node is not None:
        path.append(node.q)
        node = node.parent
    path.reverse()
    return np.array(path)


def rrt(map, start, goal):
    """
    Implement RRT algorithm in this file.
    :param map:         the map struct
    :param start:       start pose of the robot (0x7).
    :param goal:        goal pose of the robot (0x7).
    :return:            returns an mx7 matrix, where each row consists of the configuration of the Panda at a point on
                        the path. The first row is start and the last row is goal. If no path is found, PATH is empty
    """

    # initialize path
    path = []

    # RRT hyper‑parameters
    max_iters   = 500    # max samples
    step_size   = 2      # radian step per extension
    goal_bias   = 0.15     # P(sample==goal)
    goal_thresh = 0.5      # radian threshold to consider “at goal”
    number_tries = 10      # number of tries to find a path

    # get joint limits
    lowerLim = np.array([-2.8973,-1.7628,-2.8973,-3.0718,-2.8973,-0.0175,-2.8973])
    upperLim = np.array([2.8973,1.7628,2.8973,-0.0698,2.8973,3.7525,2.8973])

    start = np.array(start, dtype=float)
    goal  = np.array(goal,  dtype=float)

    if not in_limits(start) or not in_limits(goal):
        return np.empty((0,7))
    if checkCollision(start, start, map.obstacles) or checkCollision(goal, goal, map.obstacles):
        return np.empty((0,7))
    
    #  Check if there is a direct path
    if not checkCollision(start, goal, map.obstacles):
        return np.vstack([start, goal])

    for try_num in range(number_tries):
        print("try: ", try_num)
        # Initialize tree with the start node
        nodes = [Node(start, parent=None)]

        for iter_num in range(max_iters):
            # Sample
            if random.random() < goal_bias:
                q_rand = goal.copy()
            else:
                q_rand = np.random.uniform(lowerLim, upperLim)

            # Nearest neighbor
            nearest = min(nodes, key=lambda n: np.linalg.norm(n.q - q_rand))
            dir_vec = q_rand - nearest.q
            norm   = np.linalg.norm(dir_vec)
            if norm > 1e-8:
                dir_vec = dir_vec / norm
            q_new = clip_limits(nearest.q + step_size * dir_vec)

            # Reject trivial steps
            if np.linalg.norm(q_new - nearest.q) < 1e-6:
                continue

            # Collision along the small segment
            if checkCollision(nearest.q, q_new, map.obstacles):
                continue

            # Add new node
            new_node = Node(q_new, parent=nearest)
            nodes.append(new_node)

            # Check if we’ve reached the goal region
            if np.linalg.norm(q_new - goal) < goal_thresh:
                # try final connect
                if not checkCollision(q_new, goal, map.obstacles):
                    goal_node = Node(goal, parent=new_node)
                    print("total iterations: ", iter_num)
                    return build_path(goal_node)

    # Failed to find path
    return np.empty((0,7))

"""
if __name__ == '__main__':
    map_struct = loadmap("../maps/map1.txt")
    start = np.array([0,-1,0,-2,0,1.57,0])
    goal =  np.array([-1.2, 1.57 , 1.57, -2.07, -1.57, 1.57, 0.7])
    path = rrt(deepcopy(map_struct), deepcopy(start), deepcopy(goal))
    print(path)
"""

def evaluate_path(q_path, goal):
    if q_path.shape[0] == 0:
        return False, []
    
    errors = []
    for i in range(q_path.shape[0]):
        error = np.linalg.norm(q_path[i, :] - goal)
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

    results = []

    for map_name, start, goal in maps:
        map_path = os.path.join(map_folder, map_name)
        map_struct = loadmap(map_path)

        for trial in range(1, num_trials + 1):
            print("\n==============================")
            print(f"Running RRT Planner | Map: {map_name} | Trial {trial}")
            print("==============================")

            start_time = time.time()
            q_path = rrt(deepcopy(map_struct), deepcopy(start), deepcopy(goal))
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
    df.to_csv("planner_results_rrt.csv", index=False)
    print("\n✅ All tests completed. Results saved to planner_results_rrt.csv")