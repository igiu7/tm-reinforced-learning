"""
Przykładowy trening PPO (stable-baselines3). Jeśli masz własny algorytm, użyj po prostu
TrackmaniaEnv jak każdego środowiska Gymnasium: obs, info = env.reset(); env.step(action).

    python train.py             # trening
    python train.py play        # jazda wytrenowanym modelem
"""

import sys

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

from tm_env import EnvConfig, TrackmaniaEnv

MODEL_PATH = "models/ppo_tm"


def train():
    env = TrackmaniaEnv(EnvConfig(game_speed=4.0))
    try:
        model = PPO(
            "MlpPolicy", env,
            n_steps=2048, batch_size=256, learning_rate=3e-4,
            gamma=0.995, ent_coef=0.01,
            policy_kwargs=dict(net_arch=[256, 256]),
            tensorboard_log="logs", verbose=1,
        )
        model.learn(
            total_timesteps=2_000_000,
            callback=CheckpointCallback(save_freq=20_000, save_path="models", name_prefix="ppo_tm"),
        )
        model.save(MODEL_PATH)
    finally:
        env.close()


def play():
    env = TrackmaniaEnv(EnvConfig(game_speed=1.0))
    model = PPO.load(MODEL_PATH)
    try:
        while True:
            obs, info = env.reset()
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, r, term, trunc, info = env.step(action)
                done = term or trunc
            print(f"Koniec: {info['reason']}, czas {info['race_time']} ms")
    finally:
        env.close()


if __name__ == "__main__":
    play() if len(sys.argv) > 1 and sys.argv[1] == "play" else train()
