import glob
import os
import cv2
import numpy as np
import gymnasium as gym
from typing import Optional
from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer


SKY_SEGMENTATION_INDEXES = [-2, 5]
DEFAULT_CAMERA_CONFIG = {
    "distance": 2.2,
    "azimuth": 145,
    "elevation": -25.0,
    "lookat": np.array([0.0, 0.65, 0.0]),
}
# for compatability with Distracting Control Suite
DAVIS17_TRAINING_VIDEOS = [
    "bear",
    "bmx-bumps",
    "boat",
    "boxing-fisheye",
    "breakdance-flare",
    "bus",
    "car-turn",
    "cat-girl",
    "classic-car",
    "color-run",
    "crossing",
    "dance-jump",
    "dancing",
    "disc-jockey",
    "dog-agility",
    "dog-gooses",
    "dogs-scale",
    "drift-turn",
    "drone",
    "elephant",
    "flamingo",
    "hike",
    "hockey",
    "horsejump-low",
    "kid-football",
    "kite-walk",
    "koala",
    "lady-running",
    "lindy-hop",
    "longboard",
    "lucia",
    "mallard-fly",
    "mallard-water",
    "miami-surf",
    "motocross-bumps",
    "motorbike",
    "night-race",
    "paragliding",
    "planes-water",
    "rallye",
    "rhino",
    "rollerblade",
    "schoolgirls",
    "scooter-board",
    "scooter-gray",
    "sheep",
    "skate-park",
    "snowboard",
    "soccerball",
    "stroller",
    "stunt",
    "surf",
    "swing",
    "tennis",
    "tractor-sand",
    "train",
    "tuk-tuk",
    "upside-down",
    "varanus-cage",
    "walking",
]

DAVIS17_VALIDATION_VIDEOS = [
    "bike-packing",
    "blackswan",
    "bmx-trees",
    "breakdance",
    "camel",
    "car-roundabout",
    "car-shadow",
    "cows",
    "dance-twirl",
    "dog",
    "dogs-jump",
    "drift-chicane",
    "drift-straight",
    "goat",
    "gold-fish",
    "horsejump-high",
    "india",
    "judo",
    "kite-surf",
    "lab-coat",
    "libby",
    "loading",
    "mbike-trick",
    "motocross-jump",
    "paragliding-launch",
    "parkour",
    "pigs",
    "scooter-black",
    "shooting",
    "soapbox",
]


def blend(image, background, alpha=0.5):
    return (
        alpha * image.astype(np.float32) + (1.0 - alpha) * background.astype(np.float32)
    ).astype(np.uint8)


def load_mp4(path, height, width):
    cap = cv2.VideoCapture(path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    buf = np.empty((n, height, width, 3), np.uint8)
    i, ret = 0, True
    while i < n and ret:
        ret, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

        buf[i] = frame
        i += 1
    cap.release()
    return buf


class MujocoRendererSegm(MujocoRenderer):
    def render(self, render_mode: Optional[str], segmentation: bool = False):
        viewer = self._get_viewer(render_mode=render_mode)

        if render_mode in ["rgb_array", "depth_array"]:
            return viewer.render(
                render_mode=render_mode,
                camera_id=self.camera_id,
                segmentation=segmentation,
            )
        elif render_mode == "human":
            return viewer.render()


class DistractingMetaworldWrapper(gym.Wrapper):
    def __init__(
        self,
        env,
        video_paths,
        video_split="train",
        video_preload=False,
        blend_alpha=0.0,
        img_hw=256,
        camera_config=None,
        # for vanilla setup & debugging
        disable_distractors=False,
    ):
        gym.utils.RecordConstructorArgs.__init__(
            self,
            video_paths=video_paths,
            video_split=video_split,
            video_preload=video_preload,
            blend_alpha=blend_alpha,
            img_hw=img_hw,
            camera_config=camera_config,
        )
        gym.Wrapper.__init__(self, env)

        self.blend_alpha = blend_alpha
        self.video_preload = video_preload
        self.video_split = video_split
        self.img_hw = img_hw
        self.disable_distractors = disable_distractors

        camera_config = (
            DEFAULT_CAMERA_CONFIG.update(camera_config)
            if camera_config
            else DEFAULT_CAMERA_CONFIG
        )
        self.unwrapped.mujoco_renderer = MujocoRendererSegm(
            env.model, env.data, camera_config, img_hw, img_hw
        )
        # see: https://github.com/Farama-Foundation/Metaworld/pull/370
        # WARN: for some reason this may lead to change of goal on second state after the reset!
        self.unwrapped.seeded_rand_vec = True
        self.unwrapped._freeze_rand_vec = False

        self._curr_video = None
        self._curr_frame = None
        self._direction = None

        if not self.disable_distractors:
            # loading videos
            self._video_paths = sorted(glob.glob(os.path.join(video_paths, "*.mp4")))
            assert video_split in ("train", "val"), "unknown videos split"
            split = (
                DAVIS17_TRAINING_VIDEOS
                if video_split == "train"
                else DAVIS17_VALIDATION_VIDEOS
            )

            self._video_paths = [
                path
                for path in self._video_paths
                if os.path.basename(path)[:-4] in split
            ]
            self.num_videos = len(self._video_paths)
            assert self.num_videos > 0

            if self.video_preload:
                self._videos = [
                    load_mp4(path, height=img_hw, width=img_hw)
                    for path in self._video_paths
                ]

    def __reset_background(self):
        video_idx = self.env.np_random.integers(0, self.num_videos)
        if self.video_preload:
            self._curr_video = self._videos[video_idx]
        else:
            self._curr_video = load_mp4(
                self._video_paths[video_idx], height=self.img_hw, width=self.img_hw
            )

        self._direction = self.env.np_random.choice((-1, 1))
        # self._curr_frame = self.env.np_random.randint(0, len(self._video_frames) - 1)
        self._curr_frame = self.env.np_random.integers(0, len(self._curr_video))

    def __step_background(self):
        if self._curr_frame >= len(self._curr_video) - 1:
            self._direction = -abs(self._direction)

        if self._curr_frame <= 0:
            self._direction = abs(self._direction)

        self._curr_frame = self._curr_frame + self._direction

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.unwrapped.seed(seed)

        if not self.disable_distractors:
            self.__reset_background()
        obs, info = self.env.reset(seed=seed, options=options)
        obs, *_, info = self.env.step(action=self.env.action_space.high)
        return obs, info

    def step(self, action):
        if not self.disable_distractors:
            self.__step_background()
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, reward, terminated, truncated, info

    def _render(self, segmentation=False):
        return self.unwrapped.mujoco_renderer.render(
            render_mode="rgb_array", segmentation=segmentation
        )

    def render(self):
        img_array = self._render(segmentation=False)
        if not self.disable_distractors:
            segm = self._render(segmentation=True).sum(axis=2)

            background_array = self._curr_video[self._curr_frame]
            mask = sum([segm == segm_id for segm_id in SKY_SEGMENTATION_INDEXES])
            mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2).astype(bool)
            img_array[mask] = blend(
                img_array[mask], background_array[mask], alpha=self.blend_alpha
            )
        return img_array


class TerminateOnSuccessWrapper(gym.Wrapper):
    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        terminated = bool(info["success"])
        return obs, reward, terminated, truncated, info


class SelectPixelsObsWrapper(gym.ObservationWrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.observation_space = self.env.observation_space["pixels"]

    def observation(self, obs):
        return obs["pixels"]


class StackedFramesToChannelsFirstWrapper(gym.ObservationWrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        old_shape = self.env.observation_space.shape
        new_shape = (old_shape[0] * old_shape[-1],) + old_shape[1:-1]
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=new_shape, dtype=np.uint8
        )

    def observation(self, obs):
        # should be: [T, C, H, W]
        obs = obs.transpose((0, 3, 1, 2))
        return obs
