import sys
import numpy as np
import rospy
from math import cos, sin, pi
import matplotlib.pyplot as plt
import geometry_msgs.msg
import visualization_msgs.msg
from tf.transformations import quaternion_from_matrix

from core.interfaces import ArmController
from core.utils import time_in_seconds

from lib.IK_velocity_null import IK_velocity_null
from lib.calculateFK import FK
from lib.calcAngDiff import calcAngDiff
from lib.calcManipulability import calcManipulability

#####################
## Rotation Helper ##
#####################

def rotvec_to_matrix(rotvec):
    theta = np.linalg.norm(rotvec)
    if theta < 1e-9:
        return np.eye(3)

    # Normalize to get rotation axis
    k = rotvec / theta
    K = np.array([
        [0, -k[2], k[1]],
        [k[2], 0, -k[0]],
        [-k[1], k[0], 0]
    ])
    R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * np.dot(K, K)
    return R


##################
## Follow class ##
##################

class JacobianDemo():
    """
    Demo class for testing Jacobian and Inverse Velocity Kinematics.
    Contains trajectories and controller callback function.
    """
    active = False      # When to stop commanding the arm
    start_time = 0      # Start time
    dt = 0.03           # How to turn velocities into positions
    fk = FK()
    point_pub = rospy.Publisher('/vis/trace',
                                geometry_msgs.msg.PointStamped,
                                queue_size=10)
    ellipsoid_pub = rospy.Publisher('/vis/ellip',
                                    visualization_msgs.msg.Marker,
                                    queue_size=10)
    counter = 0
    x0 = np.array([0.307, 0, 0.487])  # Corresponds to neutral position
    last_iteration_time = None

    ##################
    ## TRAJECTORIES ##
    ##################

    def eight(t, fx=1.5, fy=0.6, rx=0.10, ry=0.07):
        """
        A figure-8 path in x-y with a mild rotation around z.
        Lower frequencies and smaller amplitudes to avoid safety aborts.
        """
        from math import sin, cos

        # Base position
        x0 = np.array([0.307, 0, 0.487])

        # Lissajous-based figure-8 in x-y
        xdes = x0 + np.array([
            rx * sin(fx * t),
            ry * sin(fy * t),
            0.0
        ])
        vdes = np.array([
            rx * fx * cos(fx * t),
            ry * fy * cos(fy * t),
            0.0
        ])

        # Orientation: a gentle oscillation about z-axis
        angle_amplitude = 0.1
        angle = angle_amplitude * sin(fy * t)
        dro_dt = angle_amplitude * fy * cos(fy * t)
        #rotvec = np.array([0.0, 0.0, angle])
        #rotvec = np.array([angle, 0.0, 0.0])
        rotvec = np.array([0.0, angle, 0.0])

        Rdes = rotvec_to_matrix(rotvec)
        #ang_vdes = np.array([0.0, 0.0, dro_dt])
        #ang_vdes = np.array([dro_dt, 0.0, 0.0])
        ang_vdes = np.array([0.0, dro_dt, 0.0])

        return Rdes, ang_vdes, xdes, vdes

    def ellipse(t, f=1.5, ry=0.10, rz=0.05):
        """
        An ellipse in y-z with a small 'wobble' about x-axis.
        Also using lower frequency and smaller amplitude to avoid large jumps.
        """
        from math import sin, cos

        # Center of ellipse
        x0 = np.array([0.307, 0, 0.487])

        # Ellipse param in y-z
        x = x0[0]
        y = x0[1] + ry * cos(f * t)
        z = x0[2] + rz * sin(f * t)
        xdes = np.array([x, y, z])

        # Linear velocity
        vdes = np.array([
            0.0,
            -ry * f * sin(f * t),
            rz * f * cos(f * t)
        ])

        # Small rotation about x-axis
        ang_amp = 0.2
        ang = ang_amp * sin(f * t)
        rotvec = ang * np.array([1.0, 0.0, 0.0])
        Rdes = rotvec_to_matrix(rotvec)

        # Angular velocity
        ang_v = ang_amp * f * cos(f * t)
        ang_vdes = ang_v * np.array([1.0, 0.0, 0.0])

        return Rdes, ang_vdes, xdes, vdes

    def line(t, f=0.2, L=0.10):
        """
        A simple line motion along z plus a small rotation around x.
        Reduced amplitude/frequency for safety.
        """
        from math import sin, cos, pi

        x0 = np.array([0.307, 0, 0.487])

        # Sine motion in z
        xdes = x0 + np.array([
            0.0,
            0.0,
            L * sin(2 * pi * f * t)
        ])
        vdes = np.array([
            0.0,
            0.0,
            L * (2 * pi * f) * cos(2 * pi * f * t)
        ])

        # Rotation around x-axis
        angle = -pi + (pi/4.0) * sin(2 * pi * f * t)
        rotvec = angle * np.array([1.0, 0.0, 0.0])
        Rdes = rotvec_to_matrix(rotvec)

        dangle = (pi/4.0) * f * cos(2 * pi * f * t)
        ang_vdes = dangle * np.array([1.0, 0.0, 0.0])

        return Rdes, ang_vdes, xdes, vdes

    ###################
    ## VISUALIZATION ##
    ###################

    def show_ee_position(self):
        msg = geometry_msgs.msg.PointStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = 'endeffector'
        msg.point.x = 0
        msg.point.y = 0
        msg.point.z = 0
        self.point_pub.publish(msg)

    ################
    ## CONTROLLER ##
    ################

    def follow_trajectory(self, state, trajectory):

        if self.active:
            try:
                t = time_in_seconds() - self.start_time

                # Get desired trajectory position & velocity
                Rdes, ang_vdes, xdes, vdes = trajectory(t)

                # Current joint positions
                q = state['position']
                _, T0e = self.fk.forward(q)
                R = T0e[:3, :3]
                x = T0e[:3, 3]
                curr_x = np.copy(x.flatten())

                # Proportional & feed-forward for position
                kp = 2.0  # reduced from 5.0
                v = vdes + kp * (xdes - curr_x)

                # Proportional & feed-forward for orientation
                kr = 2.0  # reduced from 5.0
                omega = ang_vdes + kr * calcAngDiff(Rdes, R).flatten()

                # Null-space secondary task
                lower = np.array([-2.8973, -1.7628, -2.8973,
                                  -3.0718, -2.8973, -0.0175, -2.8973])
                upper = np.array([ 2.8973,  1.7628,  2.8973,
                                  -0.0698,  2.8973,  3.7525,  2.8973])

                # Set q_e to midrange or some comfortable posture
                q_e = lower + (upper - lower)/2.0
                k0 = 0.3  # moderate null-space gain

                # Velocity IK with null-space
                dq = IK_velocity_null(q, v, omega, -k0*(q - q_e)).flatten()

                # Time step
                if self.last_iteration_time is None:
                    self.last_iteration_time = time_in_seconds()
                self.dt = time_in_seconds() - self.last_iteration_time
                self.last_iteration_time = time_in_seconds()

                new_q = q + self.dt * dq

                # Command new pose & velocity
                arm.safe_set_joint_positions_velocities(new_q, dq)

                # Downsample viz
                self.counter += 1
                if self.counter == 10:
                    self.show_ee_position()
                    self.counter = 0

            except rospy.exceptions.ROSException:
                pass


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("usage:\n\tpython follow.py line\n\tpython follow.py ellipse\n\tpython follow.py eight")
        sys.exit()

    rospy.init_node("follower")

    JD = JacobianDemo()

    traj_name = sys.argv[1]
    if traj_name == 'line':
        callback = lambda st: JD.follow_trajectory(st, JacobianDemo.line)
    elif traj_name == 'ellipse':
        callback = lambda st: JD.follow_trajectory(st, JacobianDemo.ellipse)
    elif traj_name == 'eight':
        callback = lambda st: JD.follow_trajectory(st, JacobianDemo.eight)
    else:
        print("invalid option")
        sys.exit()

    arm = ArmController(on_state_callback=callback)

    # reset arm
    print("resetting arm...")
    arm.safe_move_to_position(arm.neutral_position())

    # start tracking trajectory
    JD.active = True
    JD.start_time = time_in_seconds()

    input("Press Enter to stop")

