from isaacgym import gymapi

from rsl_rl.runners import OnPolicyRunner

from utils.helpers import class_to_dict, get_log_dir

from mtss_gym.mtss_gym import MotionTrackingFromSparseSensor
from mtss_cfg.mtss_cfg import MtssCfg, MtssPPOCfg

if __name__ == "__main__":
    mtss_cfg = MtssCfg()
    env = MotionTrackingFromSparseSensor(mtss_cfg, gymapi.SIM_PHYSX, "cuda", True)

    log_dir = get_log_dir("logs", "mtss")
    ppo_runner = OnPolicyRunner(env, class_to_dict(MtssPPOCfg), log_dir, device="cuda")
    mtss_cfg.wirte(log_dir+'/cfg.pkl')
    ppo_runner.learn(num_learning_iterations=MtssPPOCfg.runner.max_iterations, init_at_random_ep_len=True)