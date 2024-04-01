# Copyright (c) 2018-2023, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import numpy as np
import os
import torch
import sys

from isaacgym import gymutil, gymtorch, gymapi
from .base.vec_task import VecTask
from isaacgymenvs.utils.torch_jit_utils import quat_mul, to_torch, tensor_clamp, torch_rand_float, get_axis_params, quat_apply

class DoorOpenRobotArm(VecTask):

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        self.train_count = 0 # Variable of how many times to test one model
        self.test_pattern_array = np.array([[0.01, 0.0], [-0.01, 0.0], [0.0, 0.01], [0.0, -0.01]]) # Position of the base of the robot arm during testing
        self.test_pattern_count = 0 # Select the position to place the base of the robot arm in order from No. 0
        # Initialization of the robot arm base position and measurements
        self.x = 0.0
        self.y = 0.0

        self.max_episode_length = 350.0
        self.cfg = cfg
        self.dt = self.cfg["sim"]["dt"]
        self.actuator_max_output_torque = self.cfg["env"]["maxEffort"]
        self.control = self.cfg["env"]["control"]                   # 0:Fixed gain, 1:Variable gain
        self.stiffness_offset = self.cfg["env"]["stiffness_offset"] # Maximum value of Kp
        self.damping_offset = self.cfg["env"]["damping_offset"]     # Maximum value of Kd
        self.randomize_env = self.cfg["env"]["randomize_env"]
        self.test_flag = self.cfg["env"]["test_flag"]

        self.cfg["env"]["numObservations"] = 12
        if self.control == 1:
            self.cfg["env"]["numActions"] = 24
        elif self.control == 0:
            self.cfg["env"]["numActions"] = 8
        
        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)

        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)

        dof_force_tensor  = self.gym.acquire_dof_force_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.dof_force_tensor = gymtorch.wrap_tensor(dof_force_tensor)

        self.dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)
        self.dof_state = gymtorch.wrap_tensor(self.dof_state_tensor)
        np.shape(self.dof_state)

        rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_tensor).view(self.num_envs, -1, 13)
        self.num_bodies = self.rigid_body_states.shape[1]

        self.robot_arm_dof_state = self.dof_state.view(self.num_envs, -1, 2)[:, :self.robot_arm_num_dof]
        self.robot_arm_dof_pos = self.robot_arm_dof_state[..., 0]
        self.robot_arm_dof_vel = self.robot_arm_dof_state[..., 1]

        self.door_dof_state = self.dof_state.view(self.num_envs, -1, 2)[:, self.robot_arm_num_dof:]
        self.door_dof_pos = self.door_dof_state[..., 0]
        self.door_dof_vel = self.door_dof_state[..., 1]

        self.robot_arm_default_dof_pos = to_torch([0.0, -0.09, 0.0, -1.6, 0.0, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0], device=self.device)
        self.actions_tensor = torch.zeros(self.num_envs, self.robot_arm_num_dof+self.door_num_dof, dtype=torch.float32, device=self.device, requires_grad=False)
        self.dof_stiffness_target = torch.zeros(self.num_envs, self.robot_arm_num_dof, dtype=torch.float32, device=self.device, requires_grad=False)
        self.dof_damping_target = torch.zeros(self.num_envs, self.robot_arm_num_dof, dtype=torch.float32, device=self.device, requires_grad=False)

        self.extras = {}
        num_actors = 2 # RobotArm, Door
        self.root_states = gymtorch.wrap_tensor(actor_root_state).view(self.num_envs, num_actors, 13)
        self.initial_root_states = self.root_states.clone()
        robot_arm_state = [self.x, self.y, 0.20, 0.0, 0.0, 0.707107, 0.707107, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        door_state = [0.42, 0.4, 0.51, 0.7071068286895752, 0.0, -0.7071068286895752, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]        
        self.dof_position_target = torch.zeros(self.num_envs, self.robot_arm_num_dof+self.door_num_dof, dtype=torch.float32, device=self.device, requires_grad=False)
        state = [robot_arm_state, door_state]
        self.initial_root_states[:] = to_torch(state, device=self.device, requires_grad=False)
        
        self.all_actor_indices = torch.arange(num_actors * self.num_envs, dtype=torch.int32, device=self.device).view(self.num_envs, num_actors)
        self.env = num_actors * torch.arange(self.num_envs, dtype=torch.int32, device=self.device)
        self.global_indices = torch.arange(self.num_envs * num_actors, dtype=torch.int32, device=self.device).view(self.num_envs, -1)

        stash_num = 5
        self.obs_buf_stash = torch.zeros(self.num_envs, stash_num, dtype=torch.float32, device=self.device, requires_grad=False)

        # reward episode sums
        torch_zeros = lambda : torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.episode_sums = {"hand_pos": torch_zeros(), "knob_ang": torch_zeros(), "door_ang": torch_zeros(), "force_reward_2": torch_zeros(), "episode_finished_door_reward": torch_zeros()}

        self.reset_idx(torch.arange(self.num_envs, device=self.device))

    def create_sim(self):
        self.up_axis = self.cfg["sim"]["up_axis"]
        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0) if self.up_axis == 'z' else gymapi.Vec3(0.0, 1.0, 0.0)
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0) if self.up_axis == 'z' else gymapi.Vec3(0.5 * -spacing, 0.0, -spacing)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        robot_arm_asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../assets")
        robot_arm_asset_file = "/urdf/door_open_robot_arm/robot/crane_x7.urdf"
        door_asset_file = "/urdf/door_open_robot_arm/robot/door_pattern_1.urdf"

        if "asset" in self.cfg["env"]:
            robot_arm_asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.cfg["env"]["asset"].get("assetRoot", robot_arm_asset_root)) # パスの結合
            robot_arm_asset_file = self.cfg["env"]["asset"].get("assetFileName", robot_arm_asset_file)
            door_asset_file = self.cfg["env"]["asset"].get("door_assetFileName", door_asset_file)

        robot_arm_asset_path = os.path.join(robot_arm_asset_root, robot_arm_asset_file)
        robot_arm_asset_root = os.path.dirname(robot_arm_asset_path)
        robot_arm_asset_file = os.path.basename(robot_arm_asset_path)

        door_asset_path = os.path.join(robot_arm_asset_root, door_asset_file)
        door_asset_root = os.path.dirname(door_asset_path)
        door_asset_file = os.path.basename(door_asset_path)

        robot_arm_asset_options = gymapi.AssetOptions()
        robot_arm_asset_options.flip_visual_attachments = False
        robot_arm_asset_options.fix_base_link = True
        robot_arm_asset_options.collapse_fixed_joints = False
        robot_arm_asset_options.thickness = 0.0
        robot_arm_asset_options.density = 1000.0
        robot_arm_asset_options.armature = 0.0
        robot_arm_asset_options.use_physx_armature = True
        robot_arm_asset_options.linear_damping = 0.0
        robot_arm_asset_options.max_linear_velocity = 1000.0
        robot_arm_asset_options.angular_damping = 0.0
        robot_arm_asset_options.max_angular_velocity = 64

        door_asset_options = gymapi.AssetOptions()
        door_asset_options.flip_visual_attachments = False
        door_asset_options.fix_base_link = True
        door_asset_options.thickness = 0.0
        door_asset_options.armature = 0.0
        door_asset_options.use_physx_armature = True
        door_asset_options.linear_damping = 0.0
        door_asset_options.max_linear_velocity = 1000.0
        door_asset_options.angular_damping = 0.0
        door_asset_options.max_angular_velocity = 64.0
        door_asset_options.disable_gravity = False
        door_asset_options.enable_gyroscopic_forces = True
        door_asset_options.use_mesh_materials = False

        robot_arm_asset = self.gym.load_asset(self.sim, robot_arm_asset_root, robot_arm_asset_file, robot_arm_asset_options)
        door_asset = self.gym.load_asset(self.sim, door_asset_root, door_asset_file, door_asset_options)
        self.robot_arm_num_dof = self.gym.get_asset_dof_count(robot_arm_asset)
        self.door_num_dof = self.gym.get_asset_dof_count(door_asset)

        robot_arm_props = self.gym.get_asset_rigid_shape_properties(robot_arm_asset)
        robot_arm_props[10].friction = 0.0 # crane_x7_lower_arm_revolute_part_link
        robot_arm_props[11].friction = 0.0 # crane_x7_wrist_link
        robot_arm_props[12].friction = 0.0 # crane_x7_gripper_base_link
        robot_arm_props[13].friction = 0.2 # crane_x7_gripper_finger_a_link
        robot_arm_props[14].friction = 0.2 # gripper_a
        robot_arm_props[15].friction = 0.2 # gripper_a
        robot_arm_props[16].friction = 0.2 # gripper_a
        robot_arm_props[18].friction = 0.2 # crane_x7_gripper_finger_b_link
        robot_arm_props[19].friction = 0.2 # gripper_b
        robot_arm_props[20].friction = 0.2 # gripper_b
        robot_arm_props[21].friction = 0.2 # gripper_b
        self.gym.set_asset_rigid_shape_properties(robot_arm_asset, robot_arm_props)
        door_props = self.gym.get_asset_rigid_shape_properties(door_asset)
        door_props[3].friction = 0.1 # knob
        door_props[4].friction = 0.1 # knob
        door_props[5].friction = 0.0 # latch
        door_props[6].friction = 0.0 # wall
        self.gym.set_asset_rigid_shape_properties(door_asset, door_props)

        pose = gymapi.Transform()
        if self.up_axis == 'z':
            pose.p.z = 0.25 
            pose.r = gymapi.Quat(0.0, 0.0, 0.707107, 0.707107)
        else:
            pose.p.y = 2.0
            pose.r = gymapi.Quat(-np.sqrt(2)/2, 0.0, 0.0, np.sqrt(2)/2)

        door_pattern_1_pose = gymapi.Transform()
        door_pattern_1_pose.p = gymapi.Vec3(0.43, 0.38, 0.5)
        door_pattern_1_pose.r = gymapi.Quat(0.707106, 0.0, -0.707106, 0.0)

        wall_pose = gymapi.Transform()
        wall_pose.p = gymapi.Vec3(-0.326, 0.38, 0.5)
        wall_pose.r = gymapi.Quat(0.707106, 0.0, -0.707106, 0.0)

        self.envs = []
        self.robot_arm_handles = []
        self.cartpole_handle_2 = []
        self.env_ptr_2 = []
        self.door_handles = []
        self.dof_lower_limit = []
        self.dof_upper_limit = []
        self.dof_lower_limits = []
        self.dof_upper_limits = []

        for i in range(self.num_envs):
            self.env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)
            self.robot_arm_handle = self.gym.create_actor(self.env_ptr, robot_arm_asset, pose, "robot_arm", i, 1, 0)
            self.door_handle = self.gym.create_actor(self.env_ptr, door_asset, door_pattern_1_pose, "door", i, 0, 0)
            
            self.gym.enable_actor_dof_force_sensors(self.env_ptr, self.robot_arm_handle)

            self.robot_arm_dof_props = self.gym.get_actor_dof_properties(self.env_ptr, self.robot_arm_handle)
            self.robot_arm_dof_props['driveMode'].fill(gymapi.DOF_MODE_EFFORT)
            self.robot_arm_dof_props['effort'].fill(0.0)
            self.gym.set_actor_dof_properties(self.env_ptr, self.robot_arm_handle, self.robot_arm_dof_props)

            self.door_dof_props = self.gym.get_actor_dof_properties(self.env_ptr, self.door_handle)
            self.door_dof_props['driveMode'].fill(gymapi.DOF_MODE_EFFORT)
            self.door_dof_props['friction'].fill(0.0)
            self.door_dof_props['effort'].fill(0.0)
            self.gym.set_actor_dof_properties(self.env_ptr, self.door_handle, self.door_dof_props)

            self.envs.append(self.env_ptr)
            self.robot_arm_handles.append(self.robot_arm_handle)
            self.door_handles.append(self.door_handle)
        for j in range(self.robot_arm_num_dof):
            self.dof_lower_limit.append(self.robot_arm_dof_props['lower'][j])
            self.dof_upper_limit.append(self.robot_arm_dof_props['upper'][j])
        for f in range(self.door_num_dof):
            self.dof_lower_limit.append(self.door_dof_props['lower'][f])
            self.dof_upper_limit.append(self.door_dof_props['upper'][f])
        self.dof_lower_limits = to_torch(self.dof_lower_limit, device=self.device)
        self.dof_upper_limits = to_torch(self.dof_upper_limit, device=self.device)

        self.base_handle = self.gym.find_actor_rigid_body_handle(self.env_ptr, self.robot_arm_handle, "crane_x7_mounting_plate_link")
        self.endeffector_handle = self.gym.find_actor_rigid_body_handle(self.env_ptr, self.robot_arm_handle, "endeffector_position")
        self.door_handle = self.gym.find_actor_rigid_body_handle(self.env_ptr, self.door_handle, "board_3")


    def compute_observations(self, env_ids=None):
        if env_ids is None:
            env_ids = np.arange(self.num_envs)

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # Robot arm joint angles
        self.obs_buf[env_ids, 0:8] = self.robot_arm_dof_pos[env_ids, 0:8]
        # Measurement errors
        if self.test_flag:
            self.obs_buf[env_ids, 8:9] = self.obs_buf_stash[env_ids, 0:1] + (-1*self.x)
            self.obs_buf[env_ids, 9:10] = self.obs_buf_stash[env_ids, 1:2] + (-1*self.y)
            self.obs_buf[env_ids, 10:11] = self.obs_buf_stash[env_ids, 2:3]
        else:
            self.obs_buf[env_ids, 8:9] = self.obs_buf_stash[env_ids, 0:1] + self.obs_buf_stash[env_ids, 3:4]
            self.obs_buf[env_ids, 9:10] = self.obs_buf_stash[env_ids, 1:2] + self.obs_buf_stash[env_ids, 4:5]
            self.obs_buf[env_ids, 10:11] = self.obs_buf_stash[env_ids, 2:3]
        # Door angle
        self.obs_buf[env_ids, 11:12] = self.door_dof_pos[env_ids, 0:1]

        return self.obs_buf
    

    def compute_reward(self):
        door_ang = self.door_dof_pos[:, 0]
        knob_ang = self.door_dof_pos[:, 1]
        dof_force_link = (self.dof_force_tensor.view(-1, 11)[:, 0:7])
        dof_force_gripper_a = self.dof_force_tensor[7::11]
        dof_force_gripper_b = self.dof_force_tensor[8::11]
        hand_pos_x = self.rigid_body_states[:, self.endeffector_handle][:, 0]
        hand_pos_y = self.rigid_body_states[:, self.endeffector_handle][:, 1]
        hand_pos_z = self.rigid_body_states[:, self.endeffector_handle][:, 2]
        door_pos_x = self.rigid_body_states[:, self.door_handle][:, 0]
        door_pos_y = self.rigid_body_states[:, self.door_handle][:, 1]
        door_pos_z = self.rigid_body_states[:, self.door_handle][:, 2]

        self.rew_buf[:], self.reset_buf[:], rew_hand_pos, rew_knob_ang, rew_door_ang, force_reward_2, episode_finished_door_reward = impedance_reward(
            door_ang, knob_ang, dof_force_link, dof_force_gripper_a, dof_force_gripper_b,
            hand_pos_x, hand_pos_y, hand_pos_z, door_pos_x, door_pos_y, door_pos_z,
            self.reset_buf, self.progress_buf, self.max_episode_length,
            self.test_flag,
        )
        self.episode_sums["hand_pos"] += rew_hand_pos
        self.episode_sums["knob_ang"] += rew_knob_ang
        self.episode_sums["door_ang"] += rew_door_ang
        self.episode_sums["force_reward_2"] += force_reward_2
        self.episode_sums["episode_finished_door_reward"] += episode_finished_door_reward

    def reset_idx(self, env_ids):
        
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)

        robot_arm_pos = tensor_clamp(self.robot_arm_default_dof_pos.unsqueeze(0), self.dof_lower_limits, self.dof_upper_limits)
        self.dof_position_target[env_ids, :] = robot_arm_pos
        self.door_dof_pos[env_ids, :] = robot_arm_pos[:, self.robot_arm_num_dof:]
        self.robot_arm_dof_pos[env_ids, :] = robot_arm_pos[:, :self.robot_arm_num_dof]
        self.robot_arm_dof_vel[env_ids, :] = torch.zeros_like(self.robot_arm_dof_vel[env_ids])
        self.door_dof_state[env_ids, :] = torch.zeros_like(self.door_dof_state[env_ids])

        in_env_ids_to_number_actor = self.gym.get_sim_actor_count(self.sim)
        x_in_env_ids_to_number_actor = torch.arange(in_env_ids_to_number_actor, dtype=torch.int32, device=self.device).view(self.num_envs, -1)
        env_ids_int32 = x_in_env_ids_to_number_actor.flatten()

        self.root_states[env_ids] = to_torch(self.initial_root_states[env_ids], device=self.device, requires_grad=False)

        if self.randomize_env:
            # Base travel of the robot arm
            self.root_states[env_ids, 0, 0:1] += torch_rand_float(-0.01, 0.01, (len(env_ids), 1), device=self.device)
            self.root_states[env_ids, 0, 1:2] += torch_rand_float(-0.01, 0.01, (len(env_ids), 1), device=self.device)
            # Measurement errors
            self.obs_buf_stash[env_ids, 3:4] = torch_rand_float(-0.01, 0.01, (len(env_ids), 1), device=self.device)
            self.obs_buf_stash[env_ids, 4:5] = torch_rand_float(-0.01, 0.01, (len(env_ids), 1), device=self.device)

        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(self.episode_sums[key][env_ids] / self.progress_buf[env_ids])
            self.episode_sums[key][env_ids] = 0.

        if self.test_flag:
            print("position pattern :", self.test_pattern_count)
            print("train count :", self.train_count)
            print(">>>>>>>>>>>>>>")
            if self.train_count == 10:
                self.train_count = 0
                if self.test_pattern_count == 4:
                    sys.exit()
                self.test_pattern_count+=1
            self.train_count += 1
            # Base travel of the robot arm
            self.root_states[env_ids, 0, 0:1] = torch.tensor(np.full(self.num_envs,self.test_pattern_array[self.test_pattern_count, 0]).reshape(self.num_envs,1),dtype=torch.float32, device=self.device)
            self.root_states[env_ids, 0, 1:2] = torch.tensor(np.full(self.num_envs,self.test_pattern_array[self.test_pattern_count, 1]).reshape(self.num_envs,1),dtype=torch.float32, device=self.device)
            # Measurement errors
            self.x = self.test_pattern_array[self.test_pattern_count, 0]
            self.y = self.test_pattern_array[self.test_pattern_count, 1]

        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        env_ids_int32 = self.global_indices[env_ids, :2].flatten()

        self.gym.set_dof_position_target_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(self.dof_position_target),
                                                        gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                                gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        self.obs_buf_stash[env_ids, 0] = self.root_states[env_ids, 0, 0]
        self.obs_buf_stash[env_ids, 1] = self.root_states[env_ids, 0, 1]
        self.obs_buf_stash[env_ids, 2] = 0.0
        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0


    def pre_physics_step(self, actions):
        self.actions = actions.clone().to(self.device)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        target = self.dof_position_target[..., :]
        target[..., :8] = self.robot_arm_dof_pos[..., :8] + self.dt * self.actions[..., :8] * 4.8171
        target[..., 8] = target[..., 7]
        door_target = [0.0, 0.0]
        target[..., self.robot_arm_num_dof:] = torch.tensor(door_target, device=self.device)
        self.dof_position_target[..., :] = tensor_clamp(target, self.dof_lower_limits, self.dof_upper_limits)

        if self.control == 0:
            self.dof_stiffness_target[..., :8] = self.stiffness_offset
            self.dof_damping_target[..., :8] = self.damping_offset
        elif self.control == 1:
            self.dof_stiffness_target[..., :8] = (((self.actions[..., 8:16] + 1) / 2) * self.stiffness_offset)
            self.dof_damping_target[..., :8] = (((self.actions[..., 16:24] + 1) / 2) * self.damping_offset)
        else:
            pass

        ## 角度誤差
        diff_link = self.dof_position_target[..., :8] - self.robot_arm_dof_pos[..., :8]
        # PD制御
        Kp_control = self.dof_stiffness_target[..., :8] * diff_link
        Kd_control = self.dof_damping_target[..., :8] * self.robot_arm_dof_vel[..., :8]
        pd_control = Kp_control + Kd_control
        clamped_pd_control = tensor_clamp(pd_control, torch.tensor(-self.actuator_max_output_torque, device=self.device), torch.tensor(self.actuator_max_output_torque, device=self.device))

        self.actions_tensor[..., :8] = clamped_pd_control
        self.actions_tensor[..., 8] = clamped_pd_control[..., 7]
        self.actions_tensor[..., 9] = torch.tensor(int(-1.0), device=self.device)
        self.actions_tensor[..., 10] = torch.tensor(int(-0.7), device=self.device)
        target_forces = gymtorch.unwrap_tensor(self.actions_tensor)
        self.gym.set_dof_actuation_force_tensor(self.sim, target_forces)

    def post_physics_step(self):
        self.progress_buf += 1

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        self.compute_observations()
        self.compute_reward()

#####################################################################
###=========================jit functions=========================###
#####################################################################

@torch.jit.script # 報酬計算
def impedance_reward(
                door_ang, knob_ang,
                dof_force_link, dof_force_gripper_a, dof_force_gripper_b, 
                hand_pos_x, hand_pos_y, hand_pos_z, door_pos_x, door_pos_y, door_pos_z,
                reset_buf, progress_buf, max_episode_length,
                test_flag,
                    ):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, float, bool) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]

    goal_door_ang = 0.25
    goal_knob_ang = 0.52

    tau_max = 4.0
    L2 = 0.0
    delta = 0.1 
    Vp = 500.0
    Vd = 110.0
    Vk = 10.0
    Vf = -1.0
    Vc = 5000.0

    # Reward for distance between end-effector and doorknob
    dx, dy, dz = door_pos_x - hand_pos_x, door_pos_y - hand_pos_y, door_pos_z - hand_pos_z
    L1 = torch.sqrt(dx**2 + dy**2 + dz**2)
    a = L1 - L2
    huberloss = 1 / 2 * a**2
    huberloss = torch.where(torch.abs(a) > delta, delta * (torch.abs(a) - (1 / 2 * delta)), huberloss)
    rew_hand_pos = -huberloss * Vp

    # Reward for door angle
    door_ang[door_ang <= 0] = 0
    knob_ang[knob_ang <= 0] = 0
    rew_door_ang = (1 - torch.abs(goal_door_ang - door_ang) / goal_door_ang) * Vd
    rew_knob_ang = (1 - torch.abs(goal_knob_ang - knob_ang) / goal_knob_ang) * Vk

    # Reward for successful door opening
    door = (1 - torch.abs(goal_door_ang - door_ang) / goal_door_ang)
    finish_door_reward = door * Vc
    rew_success = torch.where((door_ang > 0.2) & (progress_buf >= (max_episode_length - 1)), finish_door_reward, torch.zeros_like(rew_door_ang))

    # Reward for torque applied to joints
    abs_dof_force_link = torch.abs(dof_force_link)
    rew_force = (abs_dof_force_link.sum(dim=1) + torch.abs(dof_force_gripper_a) + torch.abs(dof_force_gripper_b)) / tau_max
    rew_force *= Vf

    reward = rew_hand_pos + rew_knob_ang + rew_door_ang + rew_force + rew_success

    if test_flag == False:
        reset_buf = torch.where(((progress_buf >= 200) & (door_ang <= 0.1)), torch.ones_like(reset_buf), reset_buf)
    reset_buf = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), reset_buf)

    return reward, reset_buf, rew_hand_pos, rew_knob_ang, rew_door_ang, rew_force, rew_success