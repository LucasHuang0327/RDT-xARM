import os

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from configs.state_vec import STATE_VEC_IDX_MAPPING
from models.multimodal_encoder.siglip_encoder import SiglipVisionTower
from models.multimodal_encoder.t5_encoder import T5Embedder
from models.rdt_runner import RDTRunner

# DONE
# The indices that the raw vector should be mapped to in the unified action vector
XARM_STATE_INDICES = [
    STATE_VEC_IDX_MAPPING[f"right_arm_joint_{i}_pos"] for i in range(6)
] + [
    STATE_VEC_IDX_MAPPING[f"right_gripper_open"] 
]

# Create the RDT model
def create_model(args, **kwargs):
    model = RoboticDiffusionTransformerModel(args, **kwargs)
    pretrained = kwargs.get("pretrained", None)
    if (
        pretrained is not None 
        and os.path.isfile(pretrained)
    ):
        model.load_pretrained_weights(pretrained)
    return model

class RoboticDiffusionTransformerModel(object):
    """A wrapper for the RDT model, which handles
            1. Model initialization
            2. Encodings of instructions
            3. Model inference
    """
    def __init__(
        self, args, 
        device='cuda',
        dtype=torch.bfloat16,
        image_size=None,
        control_frequency=25,
        pretrained=None,
        pretrained_vision_encoder_name_or_path=None,
    ):
        self.args = args
        self.dtype = dtype
        self.image_size = image_size
        self.device = device
        self.control_frequency = control_frequency
        # We do not use the text encoder due to limited GPU memory
        # self.text_tokenizer, self.text_model = self.get_text_encoder(pretrained_text_encoder_name_or_path)
        self.image_processor, self.vision_model = self.get_vision_encoder(pretrained_vision_encoder_name_or_path)
        self.policy = self.get_policy(pretrained) # path
        
        self.reset()

    # 初始化模型/policy，返回RDTRunner類，DONE
    def get_policy(self, pretrained):
        """Initialize the model."""
        # Initialize model with arguments
        # 判斷是否有pretrain path
        if (
            pretrained is None
            or os.path.isfile(pretrained)
        ):
            img_cond_len = (self.args["common"]["img_history_size"] 
                            * self.args["common"]["num_cameras"] 
                            * self.vision_model.num_patches)
            
            # 用了RDTRunner
            _model = RDTRunner(
                action_dim=self.args["common"]["state_dim"],
                pred_horizon=self.args["common"]["action_chunk_size"],
                config=self.args["model"],
                lang_token_dim=self.args["model"]["lang_token_dim"],
                img_token_dim=self.args["model"]["img_token_dim"],
                state_token_dim=self.args["model"]["state_token_dim"],
                max_lang_cond_len=self.args["dataset"]["tokenizer_max_length"],
                img_cond_len=img_cond_len,
                img_pos_embed_config=[
                    # No initial pos embed in the last grid size
                    # since we've already done in ViT
                    ("image", (self.args["common"]["img_history_size"], 
                        self.args["common"]["num_cameras"], 
                        -self.vision_model.num_patches)),  
                ],
                lang_pos_embed_config=[
                    # Similarly, no initial pos embed for language
                    ("lang", -self.args["dataset"]["tokenizer_max_length"]),
                ],
                dtype=self.dtype,
            )
        else:
            _model = RDTRunner.from_pretrained(pretrained)
        return _model

    # Ignore
    def get_text_encoder(self, pretrained_text_encoder_name_or_path):
        text_embedder = T5Embedder(from_pretrained=pretrained_text_encoder_name_or_path, 
                                   model_max_length=self.args["dataset"]["tokenizer_max_length"], 
                                   device=self.device)
        tokenizer, text_encoder = text_embedder.tokenizer, text_embedder.model
        return tokenizer, text_encoder

    # 輸出image_processor和self.vision_model，DONE
    def get_vision_encoder(self, pretrained_vision_encoder_name_or_path):
        # 調用siglip encoder
        vision_encoder = SiglipVisionTower(vision_tower=pretrained_vision_encoder_name_or_path, args=None)
        image_processor = vision_encoder.image_processor
        return image_processor, vision_encoder

    # 只用於把模型和Policy設為eval模式，DONE
    def reset(self):
        """Set model to evaluation mode.
        """
        device = self.device
        weight_dtype = self.dtype
        self.policy.eval()
        # self.text_model.eval()
        self.vision_model.eval()

        self.policy = self.policy.to(device, dtype=weight_dtype)
        # self.text_model = self.text_model.to(device, dtype=weight_dtype)
        self.vision_model = self.vision_model.to(device, dtype=weight_dtype)

    # DONE
    def load_pretrained_weights(self, pretrained=None):
        if pretrained is None:
            return 
        print(f'Loading weights from {pretrained}')
        filename = os.path.basename(pretrained)
        if filename.endswith('.pt'):
            checkpoint =  torch.load(pretrained)
            self.policy.load_state_dict(checkpoint["module"])
        elif filename.endswith('.safetensors'):
            from safetensors.torch import load_model
            load_model(self.policy, pretrained)
        else:
            raise NotImplementedError(f"Unknown checkpoint format: {pretrained}")

    # instruction:str, device:cuda, pred: embedding of each token; DONE
    def encode_instruction(self, instruction, device="cuda"):
        """Encode string instruction to latent embeddings.

        Args:
            instruction: a string of instruction
            device: a string of device
        
        Returns:
            pred: a tensor of latent embeddings of shape (text_max_length, 512)
        """
        tokens = self.text_tokenizer(
            instruction, return_tensors="pt",
            padding="longest",
            truncation=True
        )["input_ids"].to(device)

        tokens = tokens.view(1, -1)
        with torch.no_grad():
            pred = self.text_model(tokens).last_hidden_state.detach()

        return pred

    # DONE: convert between robot raw actions and unified action vectors that RDT accepts.
    def _format_joint_to_state(self, joints):
        """
        Format the joint proprioception into the unified action vector.

        Args:
            joints (torch.Tensor): The joint proprioception to be formatted. 
                qpos ([B, N, 14]).

        Returns:
            state (torch.Tensor): The formatted vector for RDT ([B, N, 128]). 
        """
        # DONE
        # Rescale the gripper to the range of [0, 1]
        joints = joints / torch.tensor(
            [[[1, 1, 1, 1, 1, 1, 800, 1, 1, 1, 1, 1, 1, 800]]], # 800 as xArm gripper maxium range
            device=joints.device, dtype=joints.dtype
        )
        
        B, N, _ = joints.shape # B is batchsize, N is state_num
        state = torch.zeros(
            (B, N, self.args["model"]["state_token_dim"]), 
            device=joints.device, dtype=joints.dtype
        )
        # Fill into the unified state vector
        state[:, :, XARM_STATE_INDICES] = joints

        # Assemble the mask indicating each dimension's availability 
        state_elem_mask = torch.zeros(
            (B, self.args["model"]["state_token_dim"]),
            device=joints.device, dtype=joints.dtype
        )

        state_elem_mask[:, XARM_STATE_INDICES] = 1
        return state, state_elem_mask

    # DONE: convert between robot raw actions and unified action vectors that RDT accepts.
    def _unformat_action_to_joint(self, action):
        """
        Unformat the unified action vector into the joint action to be executed.

        Args:
            action (torch.Tensor): The unified action vector to be unformatted. 
                ([B, N, 128])
        
        Returns:
            joints (torch.Tensor): The unformatted robot joint action. 
                qpos ([B, N, 14]).
        """
        action_indices = XARM_STATE_INDICES
        joints = action[:, :, action_indices]
        
        # Rescale the gripper back to the action range
        # Note that the action range and proprioception range are different
        # for Mobile ALOHA robot
        joints = joints * torch.tensor(
            [[[1, 1, 1, 1, 1, 1, 800, 1, 1, 1, 1, 1, 1, 800]]], # 800 as xArm gripper maxium range
            device=joints.device, dtype=joints.dtype
        )
        
        return joints

    # TODO
    @torch.no_grad()
    def step(self, proprio, images, text_embeds):
        """
        Predict the next action chunk given the 
        proprioceptive states, images, and instruction embeddings.

        Args:
            proprio: proprioceptive states
            images: RGB images, the order should be
                [cam_high{t-1}, cam_wrist{t-1}, 
                cam_high{t}, cam_wrist{t}]
            text_embeds: instruction embeddings

        Returns:
            action: predicted action
        """
        device = self.device
        dtype = self.dtype
        
        # The background image used for padding
        background_color = np.array([
            int(x*255) for x in self.image_processor.image_mean
        ], dtype=np.uint8).reshape(1, 1, 3)
        
        # 逐元素點乘
        background_image = np.ones((
            self.image_processor.size["height"], 
            self.image_processor.size["width"], 3), dtype=np.uint8
        ) * background_color
        
        # Preprocess the images by order and encode them
        image_tensor_list = []
        for image in images:
            if image is None:
                # Replace it with the background image
                image = Image.fromarray(background_image)
            
            # 大小調整
            if self.image_size is not None:
                image = transforms.Resize(self.data_args.image_size)(image)
            
            # 調整亮度
            if self.args["dataset"].get("auto_adjust_image_brightness", False):
                pixel_values = list(image.getdata())
                average_brightness = sum(sum(pixel) for pixel in pixel_values) / (len(pixel_values) * 255.0 * 3)
                if average_brightness <= 0.15:
                    image = transforms.ColorJitter(brightness=(1.75,1.75))(image)
                    
            # 調整為正方形
            if self.args["dataset"].get("image_aspect_ratio", "pad") == 'pad':
                def expand2square(pil_img, background_color):
                    width, height = pil_img.size
                    if width == height:
                        return pil_img
                    elif width > height:
                        result = Image.new(pil_img.mode, (width, width), background_color)
                        result.paste(pil_img, (0, (width - height) // 2))
                        return result
                    else:
                        result = Image.new(pil_img.mode, (height, height), background_color)
                        result.paste(pil_img, ((height - width) // 2, 0))
                        return result
                image = expand2square(image, tuple(int(x*255) for x in self.image_processor.image_mean))
            
            # 預處理並投影到物理空間
            image = self.image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            
            #保留原image順序
            image_tensor_list.append(image)

        # 按零維度拼接
        image_tensor = torch.stack(image_tensor_list, dim=0).to(device, dtype=dtype)

        # 輸入vision encoder得到embedding
        image_embeds = self.vision_model(image_tensor).detach()
        image_embeds = image_embeds.reshape(-1, self.vision_model.hidden_size).unsqueeze(0)

        # Prepare the proprioception states and the control frequency
        joints = proprio.to(device).unsqueeze(0)   # (1, 1, 7)
        states, state_elem_mask = self._format_joint_to_state(joints)    # (1, 1, 128), (1, 128)
        states, state_elem_mask = states.to(device, dtype=dtype), state_elem_mask.to(device, dtype=dtype)
        states = states[:, -1:, :]  # (1, 1, 128)
        
        # create mask of control
        ctrl_freqs = torch.tensor([self.control_frequency]).to(device)
        
        text_embeds = text_embeds.to(device, dtype=dtype)
        
        # Predict the next action chunk given the inputs
        # predict_action函數來自RDT類定義
        trajectory = self.policy.predict_action(
            lang_tokens=text_embeds,
            lang_attn_mask=torch.ones(
                text_embeds.shape[:2], dtype=torch.bool,
                device=text_embeds.device),
            img_tokens=image_embeds,
            state_tokens=states,
            action_mask=state_elem_mask.unsqueeze(1),  
            ctrl_freqs=ctrl_freqs
        )
        trajectory = self._unformat_action_to_joint(trajectory).to(torch.float32)

        return trajectory
