from pathlib import Path
import gymnasium as gym
import ale_py
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack

gym.register_envs(ale_py)

# Copy your final saved model here and name the copy my_model.zip.
MODEL_PATH = "/Users/rabbasi074/Documents/GitHub/tensorflowpython/ticsP3/ale-independent-project/models/run_20261119_134349_281515/breakout_initial.zip"
WATCH_STEPS = 2000

agent = DQN.load(MODEL_PATH, device="cpu")

# TODO: Match your training game and wrappers, with render_mode="human".
render_mode="human"
watch_env = make_atari_env(
        "ALE/Bowling-v5",
        n_envs=1,
        env_kwargs={"frameskip": 1, "render_mode": "human"},
        wrapper_kwargs={"terminal_on_life_loss": False}
    )

watch_env = VecFrameStack(watch_env, n_stack=4)

observation = watch_env.reset()

for step in range(2000):
        action, _ = agent.predict(observation, deterministic=True)
        # Perform the action
        observation, rewards, dones, infos = watch_env.step(action)
        watch_env.close()
