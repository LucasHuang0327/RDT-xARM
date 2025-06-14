# code by Boxjod LinCC111 2025.1.13 Box2AI-Robotics copyright 盒桥智能 版权所有

import os
import mujoco
import mujoco.viewer
import numpy as np
import time
import threading

# for joycon
from lerobot_kinematics import lerobot_IK, lerobot_FK, get_robot
from joyconrobotics import JoyconRobotics
import math

np.set_printoptions(linewidth=200)
os.environ["MUJOCO_GL"] = "egl"

JOINT_NAMES = ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw"]

xml_path = "./examples/scene.xml"
mjmodel = mujoco.MjModel.from_xml_path(xml_path)
qpos_indices = np.array([mjmodel.jnt_qposadr[mjmodel.joint(name).id] for name in JOINT_NAMES])
mjdata = mujoco.MjData(mjmodel)

JOINT_INCREMENT = 0.01  
POSITION_INSERMENT = 0.002

robot = get_robot('so100')

control_glimit = [[0.125, -0.4,  0.046, -3.1, -1.5, -1.5], 
                  [0.380,  0.4,  0.23,  3.1,  1.5,  1.5]]

init_qpos = np.array([0.0, -3.14, 3.14, 0.0, -1.57, -0.157])
target_qpos = init_qpos.copy() 
init_gpos = lerobot_FK(init_qpos[1:5], robot=robot)
target_gpos = init_gpos.copy() 

lock = threading.Lock()
target_gpos_last = init_gpos.copy()
direction_data_r = [[], [], []]  

offset_position_m = init_gpos[0:3]
joyconrobotics_right = JoyconRobotics(device="right", 
                                      horizontal_stick_mode='yaw_diff', 
                                      close_y=True, 
                                      limit_dof=True, 
                                      glimit = control_glimit,
                                      offset_position_m=offset_position_m, 
                                    #   offset_euler_rad = offset_euler_rad,
                                      common_rad=False,
                                      lerobot = True,
                                      pitch_down_double=True)

t = 0
try:
    with mujoco.viewer.launch_passive(mjmodel, mjdata) as viewer:
        start = time.time()
        while viewer.is_running() and time.time() - start < 1000:
            step_start = time.time()
            if t ==0 :
                mjdata.qpos[qpos_indices] = init_qpos
                mujoco.mj_step(mjmodel, mjdata)
                with viewer.lock():
                    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(mjdata.time % 2)
                viewer.sync()
                
            t = t + 1
            target_pose, gripper_state_r, _ = joyconrobotics_right.get_control()# update()
            
            print("target_pose:", [f"{x:.3f}" for x in target_pose])
            
            for i in range(6):
                target_pose[i] = control_glimit[0][i] if target_pose[i] < control_glimit[0][i] else (control_glimit[1][i] if target_pose[i] > control_glimit[1][i] else target_pose[i])
    
            x_r = target_pose[0] # init_gpos[0] + 
            z_r = target_pose[2] # init_gpos[2] + 
            _, _, _, roll_r, pitch_r, yaw_r = target_pose
            y_r = 0.01
            pitch_r = -pitch_r 
            roll_r = roll_r - math.pi/2 # lerobo末端旋转90度
            
            right_target_gpos = np.array([x_r, y_r, z_r, roll_r, pitch_r, 0.0])
            print("robot_target_pose", right_target_gpos)
            qpos_inv,IK_success = lerobot_IK(mjdata.qpos[qpos_indices][1:5], right_target_gpos, robot=robot)
            
            if IK_success:
                target_qpos = np.concatenate(([yaw_r,], qpos_inv[:4], [gripper_state_r,])) # 使用陀螺仪控制yaw
                mjdata.qpos[qpos_indices] = target_qpos
                # mjdata.ctrl[qpos_indices] = target_qpos
                
                mujoco.mj_step(mjmodel, mjdata)
                with viewer.lock():
                    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(mjdata.time % 2)
                viewer.sync()
                
                target_gpos_last = right_target_gpos.copy() 
            else:
                right_target_gpos = target_gpos_last.copy()
                joyconrobotics_right.set_position = right_target_gpos[0:3]
            
            time_until_next_step = mjmodel.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
            time.sleep(0.001)
except KeyboardInterrupt:
    viewer.close()
