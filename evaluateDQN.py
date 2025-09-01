import os
os.environ['OMP_NUM_THREADS'] = '2'

# os.environ['PYOPENGL_PLATFORM'] = 'egl'  # Limit OpenMP threads to avoid conflicts
import argparse
import sys
import time

from PIL import Image
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cv2
import pyglet
from pyglet.window import key
from gym_duckietown.envs import DuckietownEnv
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
import matplotlib.pyplot as plt
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
# from altro_incrocio import LaneFollower
from altro_incrocio import DuckieRLWrapper
from stable_baselines3.common.evaluation import evaluate_policy  
import imageio
import random

possible_starts = [
    (2, 3),
    (1, 2), 
    (3, 2),  
    (2, 1),  
    (3, 0),
    (1, 0),
    (0, 1),
    (0, 3),
    (4, 3),
    (4, 1),
    (3, 4),
    (1, 4),                    
    ]

tile = random.choice(possible_starts)


def make_env(seed=0):
    env = DuckietownEnv(
        map_name='4way',
        domain_rand=False,
        draw_curve=True,
        draw_bbox=False,
        distortion=False,
        seed=seed,
        frame_skip=1,
        max_steps=2000,
        camera_rand=False,
        dynamics_rand=False,
        randomize_maps_on_reset=False,
        user_tile_start= (2, 1),
    )
    env = DuckieRLWrapper(env)
    env.seed(seed)
    return env

def run_evaluation(model_path, n_eval_episodes=10, render=True, save_video=True):
 
    env = make_env()
    model = DQN.load(model_path)

    env.reset()
    env.render()

    # Aspetta che pyglet crei la finestra (necessario per collegare eventi)
    if hasattr(env.unwrapped, 'window') and env.unwrapped.window is not None:
        @env.unwrapped.window.event
        def on_key_press(symbol, modifiers):
            if symbol == key.ESCAPE:
                env.close()
                sys.exit(0)
    else:
        print("⚠️ Impossibile accedere a env.unwrapped.window. Il rendering è disabilitato o fallito.")


    # env.reset()
    # env.render(mode='human') 

    # (Facoltativo) Gestore ESC
    # @env.env.unwrapped.window.event
    # def on_key_press(symbol, modifiers):
        # if symbol == key.ESCAPE:
           #  env.close()
           # sys.exit(0)


    episode_rewards = []
    step_counter = {'step': 0}
    episode = [0]
    ep_reward = [0.0]
    obs = env.reset()[0] if isinstance(env.reset(), tuple) else env.reset()

    min_steps_per_episode = 300  # Numero minimo di step per episodio

    def update(dt):
        nonlocal obs
        if episode[0] >= n_eval_episodes:
            pyglet.app.exit()
            return
        action, _ = model.predict(obs)
        step_result = env.step(action)
        if len(step_result) == 5:
            obs_, reward, terminated, truncated, info = step_result
            done = terminated or truncated
        else:
            obs_, reward, done, info = step_result
        ep_reward[0] += reward
 
        if render:
            env.render()
        obs = obs_
        step_counter['step'] += 1
        if done or step_counter['step'] >= 2000:
            episode_rewards.append(ep_reward[0])
            print(f"Episodio {episode[0]+1}/{n_eval_episodes} - Reward: {ep_reward[0]:.2f}")
            episode[0] += 1
            if episode[0] < n_eval_episodes:
                obs = env.reset()[0] if isinstance(env.reset(), tuple) else env.reset()
                ep_reward[0] = 0.0
                step_counter['step'] = 0
            else:
                pyglet.app.exit()

    pyglet.clock.schedule_interval(update, 1.0 / env.env.unwrapped.frame_rate)
    pyglet.app.run()
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    print(f"\n✅ Evaluation of model '{model_path}':")
    print(f"   → Mean reward over {n_eval_episodes} episodes: {mean_reward:.2f}")
    print(f"   → Std reward: {std_reward:.2f}")
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Percorso del file .zip del modello (es. dqn_policy_step2.zip)")
    parser.add_argument("--episodes", type=int, default=10, help="Numero di episodi da valutare")
    parser.add_argument("--render", action="store_true", help="Renderizza l'ambiente durante la valutazione")

    args = parser.parse_args()

    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Modello '{args.model}' non trovato.")

    run_evaluation(model_path=args.model, n_eval_episodes=args.episodes, render=args.render)