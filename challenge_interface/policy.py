"""
Participants need to modify this file

This is a sample script about how to adapt a model into a remote evaluation policy for CVPR challenge.

Basically, You need to implement the `step` and `reset` methods.
"""

import numpy as np


class Policy:
    def infer(self, inputs: dict) -> dict:
        """
        The `inputs` is a dict of observations, which includes:
        - task_goal (list[str]): a list of the possible task goals
        - is_first_step (bool): whether the current step is the first step
        - front_rgb_list (list[np.ndarray]): the list of front camera RGB frames
        - wrist_rgb_list (list[np.ndarray]): the list of wrist camera RGB frames
        - joint_state_list (list[np.ndarray]): the list of joint states
        - eef_state_list (list[np.ndarray]): the list of end-effector states
        - gripper_state_list (list[np.ndarray]): the list of gripper states
            
        - (optional) front_depth_list (list[np.ndarray]): the list of front camera depth frames. return only when you select `use_depth` in EvalAI.
        - (optional) wrist_depth_list (list[np.ndarray]): the list of wrist camera depth frames. return only when you select `use_depth` in EvalAI.
        - (optional) front_camera_intrinsic (np.ndarray): the intrinsic matrix of the front camera. return only when you select `use_camera_params` in EvalAI.
        - (optional) wrist_camera_intrinsic (np.ndarray): the intrinsic matrix of the wrist camera. return only when you select `use_camera_params` in EvalAI.
        - (optional) front_camera_extrinsic_list (list[np.ndarray]): the list of extrinsic matrix of the front camera. return only when you select `use_camera_params` in EvalAI.
        - (optional) wrist_camera_extrinsic_list (list[np.ndarray]): the list of extrinsic matrix of the wrist camera. return only when you select `use_camera_params` in EvalAI.
        
        
        The output is a dict of action chunk: {"actions": np.ndarray}
        
        if action space is joint_angle, the action shape is (chunk_size, 8)
        otherwise, the action shape is (chunk_size, 7)
        """
        raise NotImplementedError

    def reset(self) -> None:
        """
        Reset the policy. If your policy is stateful, you need to reset your model state here.
        The organizers will call this at the beginning of each test episode.
        """
        raise NotImplementedError
 
class DummyPolicy(Policy):
    # A random policy that saves video for debugging
    def __init__(self):
        self.chunk_size = 10
        self.base_action = np.array(
            [0.0, 0.0, 0.0, -np.pi / 2, 0.0, np.pi / 2, np.pi / 4, 1.0],
            dtype=np.float32,
        )

    def _add_small_noise(self, action: np.ndarray, noise_level: float = 0.1) -> np.ndarray:
        noise = np.random.normal(0, noise_level, action.shape)
        noise[..., -1:] = 0.0
        return action + noise


    def infer(self, inputs: dict):
        """
        We need to differentiate the first step from the subsequent steps
        For video-conditioned tasks, there would be more than one steps in inputs, the last step is the current step ready for execution, all previous steps are the conditioned video frames.
        For non-video-conditioned tasks, there would be only one step in inputs, which is the current step ready for execution
        """
        if inputs["is_first_step"]:
            self.exec_start_idx = len(inputs["front_rgb_list"]) - 1 # sample id < self.exec_id is the conditioned video frames
        action_chunk = np.concatenate([self.base_action] * self.chunk_size, axis=0).reshape(-1, 8)
        
        return {"actions": self._add_small_noise(action_chunk)}

    def reset(self):
        self.exec_start_idx = 0
    

class SB3Policy(Policy):
    """
    Wraps a Stable-Baselines3 PPO (or PPOWithICM) checkpoint trained via
    `train/train_ppo*.py`. The training policy is memoryless: it consumes
    only the latest (front_rgb, wrist_rgb, joint_state, eef_state, gripper)
    from each step, downsampled to 128x128.

    Usage from deploy.py:

        from challenge_interface.policy import SB3Policy
        policy = SB3Policy(
            model_path=r"runs/ppo_icm_v8/ppo_icm_BinFill_final.zip",
            action_space="joint_angle",
        )

    Notes:
      * Returns one action per infer() call (chunk_size=1). PPO has no
        intrinsic notion of a multi-step action plan, so chunking would
        be open-loop and worse than per-step prediction.
      * Only supports the `joint_angle` action space (8-dim). The model
        we trained outputs joint commands.
      * `reset()` is a no-op — there is no recurrent / memory state to
        clear in the vanilla-PPO + ICM extractor.
    """

    def __init__(
        self,
        model_path: str,
        action_space: str = "joint_angle",
        device: str = "cuda",
        img_h: int = 128,
        img_w: int = 128,
    ):
        if action_space != "joint_angle":
            raise NotImplementedError(
                f"SB3Policy only supports joint_angle (8-dim); got {action_space!r}"
            )
        # Delegate to evaluate_trained's loader so legacy-encoder checkpoints
        # (ppo_v2/v3/v4) deserialize correctly.
        from train.evaluate_trained import _load_ppo_with_encoder_autodetect
        self._model = _load_ppo_with_encoder_autodetect(model_path, device)
        self._model.policy.set_training_mode(False)
        self._img_h = img_h
        self._img_w = img_w

    @staticmethod
    def _to_np(x):
        return x.cpu().numpy() if hasattr(x, "cpu") else np.asarray(x)

    def _extract_latest_obs(self, inputs: dict) -> dict:
        """Build the dict obs that matches what RobommeRLEnv produces at train time."""
        import cv2

        front = self._to_np(inputs["front_rgb_list"][-1]).astype(np.uint8)
        wrist = self._to_np(inputs["wrist_rgb_list"][-1]).astype(np.uint8)
        if front.shape[:2] != (self._img_h, self._img_w):
            front = cv2.resize(front, (self._img_w, self._img_h),
                               interpolation=cv2.INTER_AREA)
        if wrist.shape[:2] != (self._img_h, self._img_w):
            wrist = cv2.resize(wrist, (self._img_w, self._img_h),
                               interpolation=cv2.INTER_AREA)

        joint = self._to_np(inputs["joint_state_list"][-1]).astype(np.float32).flatten()
        eef   = self._to_np(inputs["eef_state_list"][-1]).astype(np.float32).flatten()
        grip  = self._to_np(inputs["gripper_state_list"][-1]).astype(np.float32).flatten()

        return {
            "front_rgb":   np.expand_dims(front, 0),
            "wrist_rgb":   np.expand_dims(wrist, 0),
            "joint_state": np.expand_dims(joint[:7], 0),
            "eef_state":   np.expand_dims(eef[:6],   0),
            "gripper":     np.expand_dims(grip[:2],  0),
        }

    def infer(self, inputs: dict) -> dict:
        obs = self._extract_latest_obs(inputs)
        action, _ = self._model.predict(obs, deterministic=True)
        # action shape from SB3 with a batched dict obs: (1, 8). The protocol
        # wants (chunk_size, 8); chunk_size=1 is fine.
        action = np.asarray(action, dtype=np.float32).reshape(1, 8)
        action = np.clip(action, -1.0, 1.0)
        return {"actions": action}

    def reset(self) -> None:
        # Memoryless policy — nothing to clear.
        pass


class YourPolicy(Policy):
    ...