#!/home/lin/software/miniconda3/envs/aloha/bin/python
# -- coding: UTF-8
"""
#!/usr/bin/python3
"""

import argparse
import sys
import threading
import time
import yaml
from collections import deque

import numpy as np
#import rospy
import torch
#from cv_bridge import CvBridge
# from geometry_msgs.msg import Twist
# from nav_msgs.msg import Odometry
from PIL import Image as PImage
# from sensor_msgs.msg import Image, JointState
# from std_msgs.msg import Header
import cv2

from scripts.xArm_model import create_model
from xarm_sys.scripts.xarm_joycon.xarm_api import XArmAPI 
# sys.path.append("./")

# DONE
CAMERA_NAMES = ['cam_high','cam_wrist']
observation_window = None
lang_embeddings = None

# debug
preload_images = None

# DONE
# Initialize the model
def make_policy(args):
    with open(args.config_path, "r") as fp:
        config = yaml.safe_load(fp)
    args.config = config
    
    # pretrained_text_encoder_name_or_path = "google/t5-v1_1-xxl"
    pretrained_vision_encoder_name_or_path = "google/siglip-so400m-patch14-384"
    model = create_model(
        args=args.config, 
        dtype=torch.bfloat16,
        pretrained=args.pretrained_model_name_or_path,
        # pretrained_text_encoder_name_or_path=pretrained_text_encoder_name_or_path,
        pretrained_vision_encoder_name_or_path=pretrained_vision_encoder_name_or_path,
        control_frequency=args.ctrl_freq,
    )

    return model

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

# Interpolate the actions to make the robot move smoothly
def interpolate_action(args, prev_action, cur_action):
    steps = np.concatenate((np.array(args.arm_steps_length), np.array(args.arm_steps_length)), axis=0)
    diff = np.abs(cur_action - prev_action)
    step = np.ceil(diff / steps).astype(int)
    step = np.max(step)
    if step <= 1:
        return cur_action[np.newaxis, :]
    new_actions = np.linspace(prev_action, cur_action, step + 1)
    return new_actions[1:]

def get_config(args):
    config = {
        'episode_len': args.max_publish_step,
        'state_dim': 7,  # HARDCODED
        'chunk_size': args.chunk_size,
        'camera_names': CAMERA_NAMES,
    }
    return config

# DONE
def get_observation(args,ros_operator):
    print_flag = True

    while True and not rospy.is_shutdown():
        result = ros_operator.get_frame()
        '''
        get_frame return(img_high, img_wrist, img_high_depth, img_wrist_depth, arm_state) 
        '''
        if not result:
            if print_flag:
                print("syn fail when get_ros_observation")
                print_flag = False
            rate.sleep()
            continue
        print_flag = True
        (img_high, img_wrist, img_high_depth, img_wrist_depth, arm_state) = result
        # print(f"sync success when get_ros_observation")
        return (img_high, img_wrist, img_high_depth, img_wrist_depth, arm_state) 

# DONE
# Update the observation window buffer
def NO_ROS_update_observation_window(args, config, operator):
    
    # JPEG transformation
    # Align with training
    def jpeg_mapping(img):
        img = cv2.imencode('.jpg', img)[1].tobytes()
        img = cv2.imdecode(np.frombuffer(img, np.uint8), cv2.IMREAD_COLOR)
        return img
    
    global observation_window
    if observation_window is None:
        observation_window = deque(maxlen=2)
    
        # Append the first dummy image
        observation_window.append(
            {
                'qpos': None,
                'images':
                    {
                        config["camera_names"][0]: None,
                        config["camera_names"][1]: None,
                    },
            }
        )
        
    img_high, img_wrist, arm_state = get_observation(args,operator)
    img_high = jpeg_mapping(img_high)
    img_wrist = jpeg_mapping(img_wrist)
    
    qpos = state
    qpos = torch.from_numpy(qpos).float().cuda()
    
    observation_window.append(
        {
            'qpos': qpos,
            'images':
                {
                    config["camera_names"][0]: img_high,
                    config["camera_names"][1]: img_wrist,
                },
        }
    )

# DONE
# RDT inference, return actions
def NO_ROS_inference_fn(args, config, policy, t):
    global observation_window
    global lang_embeddings
    
    # print(f"Start inference_thread_fn: t={t}")
    while True:
        time1 = time.time()     

        # fetch images in sequence [front, right, left]
        image_arrs = [
            observation_window[-2]['images'][config['camera_names'][0]],
            observation_window[-2]['images'][config['camera_names'][1]],
            
            observation_window[-1]['images'][config['camera_names'][0]],
            observation_window[-1]['images'][config['camera_names'][1]]
        ]
        
        # fetch debug images in sequence [front, right, left]
        # image_arrs = [
        #     preload_images[config['camera_names'][0]][max(t - 1, 0)],
        #     preload_images[config['camera_names'][2]][max(t - 1, 0)],
        #     preload_images[config['camera_names'][1]][max(t - 1, 0)],
        #     preload_images[config['camera_names'][0]][t],
        #     preload_images[config['camera_names'][2]][t],
        #     preload_images[config['camera_names'][1]][t]
        # ]
        # # encode the images
        # for i in range(len(image_arrs)):
        #     image_arrs[i] = cv2.imdecode(np.frombuffer(image_arrs[i], np.uint8), cv2.IMREAD_COLOR)
        # proprio = torch.from_numpy(preload_images['qpos'][t]).float().cuda()
        
        images = [PImage.fromarray(arr) if arr is not None else None
                  for arr in image_arrs]
        
        # for i, pos in enumerate(['f', 'r', 'l'] * 2):
        #     images[i].save(f'{t}-{i}-{pos}.png')
        
        # get last qpos in shape [7, ]
        proprio = observation_window[-1]['qpos']
        # unsqueeze to [1, 7]
        proprio = proprio.unsqueeze(0)
        
        # actions shaped as [1, 64, 14] in format [left, right]
        actions = policy.step(
            proprio=proprio,
            images=images,
            text_embeds=lang_embeddings 
        ).squeeze(0).cpu().numpy()
        # print(f"inference_actions: {actions.squeeze()}")
        
        print(f"Model inference time: {time.time() - time1} s")
        
        # print(f"Finish inference_thread_fn: t={t}")
        return actions

# DONE
def NO_ROS_model_inference(args, config, operator):
    # Initialize
    global lang_embeddings
    
    # Load rdt model
    policy = make_policy(args)
    
    lang_dict = torch.load(args.lang_embeddings_path)
    print(f"Running with instruction: \"{lang_dict['instruction']}\" from \"{lang_dict['name']}\"")
    lang_embeddings = lang_dict["embeddings"]
    
    max_publish_step = config['episode_len']
    chunk_size = config['chunk_size']

    ### Initialize xArm position and previous action ###
    # init xArm position every rollout DONE
    init_qpos = np.array([14.1, -8, -24.7, 196.9, 62.3, -8.8, 730]) #joints + gripper
    init_qpos = np.radians(init_qpos)

    print("init xArm position")
    operator.move_xArm(init_qpos)
    input("Press enter to continue")

    # Initialize the previous action to be the initial robot state
    pre_action = np.zeros(config['state_dim'])
    # 填入對應的state_dim
    pre_action[:6] = np.array([14.1, -8, -24.7, 196.9, 62.3, -8.8]) #joints + gripper
    pre_action[7] = 730 # IDX是7 還是 10？
    action = None
    
    # Inference loop
    with torch.inference_mode():
        # NO ROS, using dev to read
        while True: # 一直跑
            # The current time step
            t = 0
            action_buffer = np.zeros([chunk_size, config['state_dim']])
            max_steps = 50
            while step in range(max_steps): # 一直跑
                # Update observation window, get image, state, lang
                NO_ROS_update_observation_window(args, config, operator)
                
                # When coming to the end of the action chunk
                if t % chunk_size == 0:
                    # Start inference and get action array
                    action_buffer = NO_ROS_inference_fn(args, config, policy, t).copy()
                
                raw_action = action_buffer[t % chunk_size]
                action = raw_action

                #使用中間插值
                # Interpolate the original action sequence
                if args.use_actions_interpolation:
                    # print(f"Time {t}, pre {pre_action}, act {action}")
                    interp_actions = interpolate_action(args, pre_action, action)
                else:
                    interp_actions = action[np.newaxis, :]

                # Execute the interpolated actions one by one
                for act in interp_actions:
                    action = act[:7]
                    
                    # execute action
                    if not args.disable_puppet_arm:
                        operater.move_xArm(action)# DONE，from Non_ROS_Operater

                    # No robotbase
                    # Operating vel action, TODO watch whether vel is needed or not
                    # if args.use_robot_base:
                    #     vel_action = act[7:9] 
                    #     # operater vel 
                    #     # from Non_ROS_Operater

                    time.sleep(1/120)
                    print(f"doing action: {act}")
                t += 1
                
                print("Published Step", t)
                pre_action = action.copy()

# WORK: TODO
class NO_RosOperater:
    # DONE
    def __init__(self, args):
        self.args = args
        
        self.arm_state_deque = None
        self.img_high_deque = None
        self.img_wrist_deque = None

        # CV
        self.bridge = None
        self.cam_high = None
        self.cam_wrist = None
         
        # init xArm
        self.arm = None
        
        self.init()
        self.init_xarm()
        self.init_dev()
        
        ## 停用
        self.args.use_depth_image = False
        # self.img_high_depth_deque = None
        # self.img_right_depth_deque = None
        ## 未知用途
        # self.puppet_arm_right_publisher = None
        # self.puppet_arm_publish_thread = None
        # self.puppet_arm_publish_lock = None

    # DONE
    def init_xarm(self):
        self.arm = XArmAPI('192.168.1.222')
        self.arm.motion_enable(enable=True)
        self.arm.set_gripper_enable(enable=True)
        self.arm.set_mode(0)
        self.arm.set_state(0)
        
    # DONE
    def init(self):
        # cv
        self.deque = deque()
        self.img_wrist_deque = deque()
        self.img_high_deque = deque()
        
        
        ## 停用
        # self.depth_deque = deque()
        # self.img_right_depth_deque = deque()
        # self.img_high_depth_deque = deque()

        self.arm_state_deque = deque()
        
        ## 未知用途
        # self.puppet_arm_publish_lock = threading.Lock()
        # self.puppet_arm_publish_lock.acquire()

    ## 移動機械臂, DONE
    def move_xArm(self, angle):
        xarm_joint = angle[:5]
        xarm_gripper = angle[6]
        arm.set_servo_angle(angle=xarm_joint,speed=8,is_radian=False)
        arm.set_gripper_position(pos=xarm_gripper, wait=False)

    ## 同步, TODO
    def get_frame(self):
        '''
        return
        (img_high, img_wrist, img_high_depth, img_wrist_depth, arm_state)
        '''
        if len(self.img_wrist_deque) == 0 or len(self.img_high_deque) == 0 or \
                (self.args.use_depth_image or len(self.img_right_depth_deque) == 0 ):
            return False
        if self.args.use_depth_image:
            frame_time = min([self.img_wrist_deque[-1].header.stamp.to_sec(), self.img_high_deque[-1].header.stamp.to_sec(),
                               self.img_right_depth_deque[-1].header.stamp.to_sec(), self.img_high_depth_deque[-1].header.stamp.to_sec()])
        else:
            frame_time = min([self.img_wrist_deque[-1].header.stamp.to_sec(), self.img_high_deque[-1].header.stamp.to_sec()])

        if len(self.img_wrist_deque) == 0 or self.img_wrist_deque[-1].header.stamp.to_sec() < frame_time:
            return False
        if len(self.img_high_deque) == 0 or self.img_high_deque[-1].header.stamp.to_sec() < frame_time:
            return False
        if len(self.arm_state_deque) == 0 or self.arm_state_deque[-1].header.stamp.to_sec() < frame_time:
            return False
        if self.args.use_depth_image and (len(self.img_right_depth_deque) == 0 or self.img_right_depth_deque[-1].header.stamp.to_sec() < frame_time):
            return False
        if self.args.use_depth_image and (len(self.img_high_depth_deque) == 0 or self.img_high_depth_deque[-1].header.stamp.to_sec() < frame_time):
            return False
            
        while self.img_wrist_deque[0].header.stamp.to_sec() < frame_time:
            self.img_wrist_deque.popleft()
        img_right = self.bridge.imgmsg_to_cv2(self.img_wrist_deque.popleft(), 'passthrough')

        while self.img_high_deque[0].header.stamp.to_sec() < frame_time:
            self.img_high_deque.popleft()
        img_high = self.bridge.imgmsg_to_cv2(self.img_high_deque.popleft(), 'passthrough')

        while self.arm_state_deque[0].header.stamp.to_sec() < frame_time:
            self.arm_state_deque.popleft()
        arm_state = self.arm_state_deque.popleft()

        img_right_depth = None
        if self.args.use_depth_image:
            while self.img_right_depth_deque[0].header.stamp.to_sec() < frame_time:
                self.img_right_depth_deque.popleft()
            img_right_depth = self.bridge.imgmsg_to_cv2(self.img_right_depth_deque.popleft(), 'passthrough')

        img_high_depth = None
        if self.args.use_depth_image:
            while self.img_high_depth_deque[0].header.stamp.to_sec() < frame_time:
                self.img_high_depth_deque.popleft()
            img_high_depth = self.bridge.imgmsg_to_cv2(self.img_high_depth_deque.popleft(), 'passthrough')

        return (img_high, img_wrist, img_high_depth, img_wrist_depth, arm_state)

    ## 串口讀入，運行即讀串口並輸出
    # DONE 有一個cam_wrist串口類待實現或傳入, 用self 傳入
    def img_wrist_callback(self):
        # Capture images from the cameras
        if len(self.img_wrist_deque) >= 2000:
            self.img_wrist_deque.popleft()
            
        ret_wrist, wrist_img = self.cam_wrist.read()
        if not ret_wrist:
            raise OSError(f"Error: Failed to capture images from high.")
        self.img_wrist_deque.append(wrist_img)

    # DONE
    def img_high_callback(self):
        # Capture images from the cameras
        if len(self.img_high_deque) >= 2000:
            self.img_high_deque.popleft()
            
        ret_high, high_img = self.cam_high.read()
        if not ret_high:
            raise OSError(f"Error: Failed to capture images from wrist")
        self.img_high_deque.append(high_img)
        
    '''    
    def img_wrist_depth_callback(self, msg):
        if len(self.img_right_depth_deque) >= 2000:
            self.img_right_depth_deque.popleft()
        self.img_right_depth_deque.append(msg)

    def img_high_depth_callback(self, msg):
        if len(self.img_high_depth_deque) >= 2000:
            self.img_high_depth_deque.popleft()
        self.img_high_depth_deque.append(msg)
    '''
    # DONE, 
    def arm_callback(self):
        if len(self.arm_state_deque) >= 2000:
            self.arm_state_deque.popleft()
         # Get qpos and normalize
        _, curr_angles_rad = arm.get_servo_angle(is_radian=True)  # xArm6 has 7 DOF
        _, curr_gripper_state = arm.get_gripper_position()
        curr_state = np.append(np.rad2deg(curr_angles_rad[:-1]), curr_gripper_state)
        
        self.arm_state_deque.append(curr_state)

    # DONE
    def init_dev(self):
        # OpenCV video capture for cameras
        self.cam_high = cv2.VideoCapture("/dev/video0")  # Base camera
        self.cam_wrist = cv2.VideoCapture("/dev/video6")  # Wrist camera
        if not self.cam_high.isOpened() or not self.cam_wrist.isOpened():
            raise OSError("Unable to open one or both cameras.")
            exit()
            

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_publish_step', action='store', type=int, 
                        help='Maximum number of action publishing steps', default=10000, required=False)
    parser.add_argument('--seed', action='store', type=int, 
                        help='Random seed', default=None, required=False)

    parser.add_argument('--img_high_topic', action='store', type=str, help='img_high_topic',
                        default='/camera_f/color/image_raw', required=False)
    parser.add_argument('--img_left_topic', action='store', type=str, help='img_left_topic',
                        default='/camera_l/color/image_raw', required=False)
    parser.add_argument('--img_right_topic', action='store', type=str, help='img_right_topic',
                        default='/camera_r/color/image_raw', required=False)
    
    parser.add_argument('--img_high_depth_topic', action='store', type=str, help='img_high_depth_topic',
                        default='/camera_f/depth/image_raw', required=False)
    parser.add_argument('--img_left_depth_topic', action='store', type=str, help='img_left_depth_topic',
                        default='/camera_l/depth/image_raw', required=False)
    parser.add_argument('--img_right_depth_topic', action='store', type=str, help='img_right_depth_topic',
                        default='/camera_r/depth/image_raw', required=False)
    
    parser.add_argument('--puppet_arm_left_cmd_topic', action='store', type=str, help='puppet_arm_left_cmd_topic',
                        default='/master/joint_left', required=False)
    parser.add_argument('--puppet_arm_right_cmd_topic', action='store', type=str, help='puppet_arm_right_cmd_topic',
                        default='/master/joint_right', required=False)
    parser.add_argument('--puppet_arm_left_topic', action='store', type=str, help='puppet_arm_left_topic',
                        default='/puppet/joint_left', required=False)
    parser.add_argument('--puppet_arm_right_topic', action='store', type=str, help='puppet_arm_right_topic',
                        default='/puppet/joint_right', required=False)
    
    parser.add_argument('--robot_base_topic', action='store', type=str, help='robot_base_topic',
                        default='/odom_raw', required=False)
    parser.add_argument('--robot_base_cmd_topic', action='store', type=str, help='robot_base_topic',
                        default='/cmd_vel', required=False)
    parser.add_argument('--use_robot_base', action='store_true', 
                        help='Whether to use the robot base to move around',
                        default=False, required=False)
    parser.add_argument('--publish_rate', action='store', type=int, 
                        help='The rate at which to publish the actions',
                        default=30, required=False)
    parser.add_argument('--ctrl_freq', action='store', type=int, 
                        help='The control frequency of the robot',
                        default=25, required=False)
    
    parser.add_argument('--chunk_size', action='store', type=int, 
                        help='Action chunk size',
                        default=64, required=False)
    parser.add_argument('--arm_steps_length', action='store', type=float, 
                        help='The maximum change allowed for each joint per timestep',
                        default=[0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.2], required=False)

    parser.add_argument('--use_actions_interpolation', action='store_true',
                        help='Whether to interpolate the actions if the difference is too large',
                        default=False, required=False)
    parser.add_argument('--use_depth_image', action='store_true', 
                        help='Whether to use depth images',
                        default=False, required=False)
    
    parser.add_argument('--disable_puppet_arm', action='store_true',
                        help='Whether to disable the puppet arm. This is useful for safely debugging',default=False)
    
    parser.add_argument('--config_path', type=str, default="configs/base.yaml", 
                        help='Path to the config file')
    # parser.add_argument('--cfg_scale', type=float, default=2.0,
    #                     help='the scaling factor used to modify the magnitude of the control features during denoising')
    parser.add_argument('--pretrained_model_name_or_path', type=str, required=True, help='Name or path to the pretrained model')
    
    parser.add_argument('--lang_embeddings_path', type=str, required=True, 
                        help='Path to the pre-encoded language instruction embeddings')
    
    args = parser.parse_args()
    return args

# DONE
def main():
    args = get_arguments()              # Parser
    operator = NO_RosOperater(args)    # 初始化ROS operator from args
    if args.seed is not None:           
        set_seed(args.seed)
    config = get_config(args)           # get config from args
    NO_ROS_model_inference(args, config, operator) # 主循環

if __name__ == '__main__':
    main()
