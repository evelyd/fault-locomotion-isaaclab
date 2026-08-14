import os

# Fail-safe: se non è stato scelto esplicitamente altro,
# ROS 2 comunica solamente sulla macchina locale.
os.environ.setdefault("ROS_LOCALHOST_ONLY", "1")

print(
    "ROS 2 network mode:",
    "LOCALHOST" if os.environ["ROS_LOCALHOST_ONLY"] == "1" else "NETWORK",
)

import sys
import shlex
import subprocess
from pathlib import Path

dir_path = Path(__file__).resolve().parent
sys.path.append(str(dir_path / ".."))

ros_ws = dir_path / "ros2_ws"
setup_bash = ros_ws / "install" / "setup.bash"

if not setup_bash.exists():
    print("Building the msgs first...")
    subprocess.run(["colcon", "build"], cwd=ros_ws, check=True)

if os.environ.get("FAULT_LOCOMOTION_ROS2_SOURCED") != "1":
    print("Sourcing ROS2 workspace and restarting script...")
    cmd = (
        f"source {shlex.quote(str(setup_bash))} && "
        "export FAULT_LOCOMOTION_ROS2_SOURCED=1 && "
        f"exec {shlex.quote(sys.executable)} "
        + " ".join(shlex.quote(arg) for arg in [str(Path(__file__).resolve()), *sys.argv[1:]])
    )
    os.execv("/bin/bash", ["bash", "-c", cmd])


import rclpy
from rclpy.node import Node
from dls2_interface.msg import BaseState, BlindState, Imu, TrajectoryGenerator, FeetContactState
from visualization_msgs.msg import Marker, MarkerArray
from gym_quadruped.sensors.heightmap import HeightMap
from gym_quadruped.utils.mujoco.visual import render_sphere

import time
import numpy as np
np.set_printoptions(precision=3, suppress=True)

# Gym and Simulation related imports
import mujoco
from gym_quadruped.quadruped_env import QuadrupedEnv
from gym_quadruped.utils.quadruped_utils import LegsAttr

# Config imports
import config as cfg

import os
dir_path = os.path.dirname(os.path.realpath(__file__))
# Set the priority of the process
pid = os.getpid()
print("PID: ", pid)
os.system("renice -n -21 -p " + str(pid))
os.system("echo -20 > /proc/" + str(pid) + "/autogroup")
#for real time, launch it with chrt -r 99 python3 run_controller.py


USE_SCHEDULER = True # Use the scheduler to compute the control signal
SCHEDULER_FREQ = 500 # Frequency of the scheduler
RENDER_FREQ = 30

# Shell for the controllers ----------------------------------------------
class Simulator_Node(Node):
    def __init__(self):
        super().__init__('Simulator_Node')

        # Subscribers and Publishers
        self.publisher_base_state = self.create_publisher(BaseState,"base_state", 1)
        self.publisher_blind_state = self.create_publisher(BlindState,"blind_state", 1)
        self.publisher_imu = self.create_publisher(Imu,"imu", 1)
        self.publisher_feet_contact_state = self.create_publisher(FeetContactState,"feet_contact_state", 1)

        self.subscriber_trajectory_generator = self.create_subscription(TrajectoryGenerator,"trajectory_generator", self.get_trajectory_generator_callback, 1)

        self.timer = self.create_timer(1.0/SCHEDULER_FREQ, self.compute_simulator_step_callback)

        # Timing stuff
        self.loop_time = 0.002
        self.last_start_time = None
        self.last_mpc_loop_time = 0.0


        # Mujoco env
        self.env = QuadrupedEnv(
            robot=cfg.robot,
            scene=cfg.scene,
            sim_dt=1.0/SCHEDULER_FREQ,
            base_vel_command_type="human"
        )
        self.env.reset(random=False)

        self.publisher_heightmap = self.create_publisher(MarkerArray, "/height_scan_markers", 1)

        pattern_cfg = cfg.training_env["height_scanner2"]["pattern_cfg"]
        resolution_heightmap = pattern_cfg["resolution"]
        self.num_rows_heightmap = round(pattern_cfg["size"][0] / resolution_heightmap) + 1
        self.num_cols_heightmap = round(pattern_cfg["size"][1] / resolution_heightmap) + 1

        self.heightmap = HeightMap(
            num_rows=self.num_rows_heightmap,
            num_cols=self.num_cols_heightmap,
            dist_x=resolution_heightmap,
            dist_y=resolution_heightmap,
            mj_model=self.env.mjModel,
            mj_data=self.env.mjData,
        )
        self.heightmap_publish_counter = 0


        self.last_render_time = time.time()
        self.env.render()
        self.env.viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
        self.env.viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = False

        # Desired PD
        self.desired_joints_position = LegsAttr(*[np.zeros((int(self.env.mjModel.nu/4), 1)) for _ in range(4)])
        self.desired_joints_velocity = LegsAttr(*[np.zeros((int(self.env.mjModel.nu/4), 1)) for _ in range(4)])

        # Desired gains
        self.Kp = np.zeros(12)
        self.Kd = np.zeros(12)


    def get_trajectory_generator_callback(self, msg):

        joints_position = np.array(msg.joints_position)

        self.desired_joints_position.FL = joints_position[0:3]
        self.desired_joints_position.FR = joints_position[3:6]
        self.desired_joints_position.RL = joints_position[6:9]
        self.desired_joints_position.RR = joints_position[9:12]

        self.Kp = np.array(msg.kp)
        self.Kd = np.array(msg.kd)


    def compute_simulator_step_callback(self):

        qpos, qvel = self.env.mjData.qpos, self.env.mjData.qvel
        base_lin_vel = self.env.base_lin_vel(frame='world')
        base_ang_vel = self.env.base_ang_vel(frame='base')
        base_pos = self.env.base_pos

        # Publish Heightmap ------------------------------------------------
        self.heightmap_publish_counter += 1
        if self.heightmap_publish_counter >= 10:  # 500Hz / 10 = 50Hz
            self.heightmap_publish_counter = 0

            base_pos = self.env.base_pos
            yaw = self.env.base_ori_euler_xyz[2]
            self.heightmap.update_height_map(base_pos, yaw=yaw)

            # 1. Calculate inverse rotation to convert World -> Local
            base_quat_wxyz = self.env.mjData.qpos[3:7]
            rotation = np.empty(9, dtype=np.float64)
            mujoco.mju_quat2Mat(rotation, base_quat_wxyz)
            # Transpose the rotation matrix to get its inverse
            R_inv = rotation.reshape(3, 3).T

            marker_array = MarkerArray()
            marker_id = 0

            for i in range(self.num_rows_heightmap):
                for j in range(self.num_cols_heightmap):
                    marker = Marker()
                    marker.header.frame_id = "base_link" # Set semantically correct frame
                    marker.header.stamp = self.get_clock().now().to_msg()
                    marker.ns = "heightmap"
                    marker.id = marker_id
                    marker.type = Marker.SPHERE
                    marker.action = Marker.ADD

                    pos = self.heightmap.data[i][j]
                    if len(pos.shape) > 1:
                        pos = pos[0]

                    # 2. Transform the world point into a local point
                    pos_local = R_inv @ (pos - base_pos)

                    # 3. Publish the local point
                    marker.pose.position.x = float(pos_local[0])
                    marker.pose.position.y = float(pos_local[1])
                    marker.pose.position.z = float(pos_local[2])

                    marker.scale.x = 0.02
                    marker.scale.y = 0.02
                    marker.scale.z = 0.02

                    marker.color.r = 0.0
                    marker.color.g = 1.0
                    marker.color.b = 0.0
                    marker.color.a = 0.5

                    marker_array.markers.append(marker)
                    marker_id += 1

            self.publisher_heightmap.publish(marker_array)

        # Publish Base State ------------------------------------------------
        base_state_msg = BaseState()
        base_state_msg.pose.position = base_pos
        base_state_msg.pose.orientation = np.roll(self.env.mjData.qpos[3:7],-1)
        base_state_msg.velocity.linear = base_lin_vel
        base_state_msg.velocity.angular = base_ang_vel
        self.publisher_base_state.publish(base_state_msg)


        # Publish Blind State ------------------------------------------------
        blind_state_msg = BlindState()
        blind_state_msg.joints_position = self.env.mjData.qpos[7:].tolist()
        blind_state_msg.joints_velocity = self.env.mjData.qvel[6:].tolist()
        blind_state_msg.joints_effort = self.env.mjData.qfrc_actuator[6:].tolist()
        self.publisher_blind_state.publish(blind_state_msg)


        # Publish IMU ------------------------------------------------
        imu_msg = Imu()
        imu_msg.linear_acceleration = self.env.mjData.sensordata[0:3]
        imu_msg.angular_velocity = self.env.mjData.sensordata[3:6]
        imu_msg.orientation = self.env.mjData.sensordata[9:13]
        self.publisher_imu.publish(imu_msg)


        # Publish Feet Contact State ------------------------------------------------
        _, _, feet_GRF = self.env.feet_contact_state(ground_reaction_forces=True)
        feet_contact_state_msg = FeetContactState()
        feet_contact_state_msg.feet_name = ["FL", "FR", "RL", "RR"]
        feet_contact_state_msg.linear_grf_feet = np.concatenate([feet_GRF["FL"], feet_GRF["FR"], feet_GRF["RL"], feet_GRF["RR"]]).tolist()
        feet_contact_state_msg.angular_grf_feet = np.concatenate([feet_GRF["FL"]*0.0, feet_GRF["FR"]*0.0, feet_GRF["RL"]*0.0, feet_GRF["RR"]*0.0]).tolist()
        self.publisher_feet_contact_state.publish(feet_contact_state_msg)


        # Step the environment --------------------------------------------------------------------------------
        joints_pos = LegsAttr(*[np.zeros((1, int(self.env.mjModel.nu/4))) for _ in range(4)])
        joints_pos.FL = qpos[self.env.legs_qpos_idx.FL]
        joints_pos.FR = qpos[self.env.legs_qpos_idx.FR]
        joints_pos.RL = qpos[self.env.legs_qpos_idx.RL]
        joints_pos.RR = qpos[self.env.legs_qpos_idx.RR]


        joints_vel = LegsAttr(*[np.zeros((1, int(self.env.mjModel.nu/4))) for _ in range(4)])
        joints_vel.FL = qvel[self.env.legs_qvel_idx.FL]
        joints_vel.FR = qvel[self.env.legs_qvel_idx.FR]
        joints_vel.RL = qvel[self.env.legs_qvel_idx.RL]
        joints_vel.RR = qvel[self.env.legs_qvel_idx.RR]


        action = np.zeros(self.env.mjModel.nu)
        action[self.env.legs_tau_idx.FL] = self.Kp[0:3]*(self.desired_joints_position.FL.reshape(-1) - joints_pos.FL) - self.Kd[0:3]*(joints_vel.FL)
        action[self.env.legs_tau_idx.FR] = self.Kp[3:6]*(self.desired_joints_position.FR.reshape(-1) - joints_pos.FR) - self.Kd[3:6]*(joints_vel.FR)
        action[self.env.legs_tau_idx.RL] = self.Kp[6:9]*(self.desired_joints_position.RL.reshape(-1) - joints_pos.RL) - self.Kd[6:9]*(joints_vel.RL)
        action[self.env.legs_tau_idx.RR] = self.Kp[9:12]*(self.desired_joints_position.RR.reshape(-1) - joints_pos.RR) - self.Kd[9:12]*(joints_vel.RR)
        self.env.step(action=action)


        # Render only at a certain frequency -----------------------------------------------------------------
        if time.time() - self.last_render_time > 1.0 / RENDER_FREQ:
            self.env.render()
            self.last_render_time = time.time()

            if self.heightmap.data is not None:
                for i in range(self.heightmap.data.shape[0]):
                    for j in range(self.heightmap.data.data.shape[1]):
                        self.heightmap.geom_ids[i, j] = render_sphere(
                            viewer=self.env.viewer,
                            position=([self.heightmap.data[i][j][0][0], self.heightmap.data[i][j][0][1], self.heightmap.data[i][j][0][2]]),
                            diameter=0.02,
                            color=[0, 1, 0, 0.5],
                            geom_id=self.heightmap.geom_ids[i, j],
                        )

def main():
    print('Hello from the gym_quadruped simulator.')
    rclpy.init()

    simulator_node = Simulator_Node()

    rclpy.spin(simulator_node)
    simulator_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
