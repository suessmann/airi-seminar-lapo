import h5py
import gymnasium as gym
import metaworld.envs as me
import metaworld.policies as mp

from .wrapper import DistractingMetaworldWrapper
from .wrapper import TerminateOnSuccessWrapper
from .wrapper import StackedFramesToChannelsFirstWrapper


def load_expert_policy(task):
    policy_map = {
        "basketball-v2-goal-observable": mp.SawyerBasketballV2Policy,
        "soccer-v2-goal-observable": mp.SawyerSoccerV2Policy,
        "pick-place-v2-goal-observable": mp.SawyerPickPlaceV2Policy,
        "push-v2-goal-observable": mp.SawyerPushV2Policy,
        "pick-place-wall-v2-goal-observable": mp.SawyerPickPlaceWallV2Policy,
        "reach-v2-goal-observable": mp.SawyerReachV2Policy,
        "button-press-v2-goal-observable": mp.SawyerButtonPressV2Policy,
        "peg-insert-side-v2-goal-observable": mp.SawyerPegInsertionSideV2Policy,
        "window-open-v2-goal-observable": mp.SawyerWindowOpenV2Policy,
        "shelf-place-v2-goal-observable": mp.SawyerShelfPlaceV2Policy,
        "lever-pull-v2-goal-observable": mp.SawyerLeverPullV2Policy,
        "hammer-v2-goal-observable": mp.SawyerHammerV2Policy,
        "drawer-open-v2-goal-observable": mp.SawyerDrawerOpenV2Policy,
        "door-unlock-v2-goal-observable": mp.SawyerDoorUnlockV2Policy,
        "plate-slide-v2-goal-observable": mp.SawyerPlateSlideV2Policy,
        "coffee-push-v2-goal-observable": mp.SawyerCoffeePushV2Policy,
        "bin-picking-v2-goal-observable": mp.SawyerBinPickingV2Policy,
    }
    return policy_map[task]()


def make_env_from_dataset(data_path, video_paths, frame_stack):
    with h5py.File(data_path, "r") as df:
        env = me.ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[df.attrs["task_name"]](
            render_mode="rgb_array"
        )
        env = DistractingMetaworldWrapper(
            env,
            video_paths=video_paths,
            video_split=df.attrs["split"],
            img_hw=df.attrs["img_hw"],
            disable_distractors=df.attrs["disable_distractors"],
        )
        env = TerminateOnSuccessWrapper(env)
        env = gym.wrappers.AddRenderObservation(env)
        env = gym.wrappers.FrameStackObservation(env, stack_size=frame_stack)
        env = StackedFramesToChannelsFirstWrapper(env)
    return env
