#!/usr/bin/env python3.10
"""
Lane-following controller for Duckietown simulator.
Enhanced PID lane follower with continuous fallback,
maintaining a minimum speed when the lane is lost.
"""
import os
os.environ['OMP_NUM_THREADS'] = '1'

# os.environ['PYOPENGL_PLATFORM'] = 'egl'  # Limit OpenMP threads to avoid conflicts
import argparse
import sys
import time 

from PIL import Image
from datetime import datetime
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
import random

def moving_average(x, w):

    x = np.array(x).flatten()
    return np.convolve(x, np.ones(w), 'valid') / w

class DuckieRLWrapper(gym.Env):

    def __init__(self, env):
        super(DuckieRLWrapper, self).__init__()
        self.env = env

        possible_starts = [
            ((1, 2), 0),  # da nord verso sud
            ((3, 2), 1), # da sud verso nord
            ((2, 1), 2), # da ovest verso est
            ((2, 3), 3), # da est verso ovest
            ((3, 0), 0),
            ((1, 0), 1),
            ((0, 1), 2),
            ((0, 3), 3),
            ((4, 3), 0),
            ((4, 1), 1),
            ((3, 4), 2), 
            ((1, 4), 3)                   
            ]
        
        self.tile_kinds = ["straight", "curve_left", "curve_right", "3way_left", "3way_right", "4way", "asphalt"]

        
        self.current_road_abs_angle = None

        self.prev_lateral = None
        # Scegli una tile iniziale casuale dalla lista
        start_pos, start_angle = random.choice(possible_starts)
        self.num_steps = 0

        # Imposta la posizione e angolo dell'agente
        self.env.cur_pos = np.array([
            start_pos[0] + 0.5,  # X (centro della tile)
            0.0,                 # Y (altezza fissa, verrà gestita internamente)
            start_pos[1] + 0.5   # Z (centro della tile)
        ])

        self.env.cur_angle = start_angle

        self.oscillation_count = 0
        self.last_action = None

        # Per il tracking del movimento verso la goal
        self.last_pos = None

        # modificato dimensione da (480, 680, 3) per ridurre il carico computazionale

        # hybrid observation: image  + vector features
        # self.observation_space = spaces.Box(
            # low=np.array([-1., -np.pi, 0.,   0.]), 
            # high=np.array([ 1.,  np.pi, 1.,   2.]),  # 2 = due tile di distanza massima
            # dtype=np.float32
        # )

        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32
        )

        self.action_space = spaces.Discrete(7)  # Azioni: sinistra, dritto, destra (per semplificare)

        # possibile implementazione in spazio continuo
        self.good_steps = 0 
        self.goal_tile = None
        self.last_action = None
        self.oscillation_count = 0
        self.lane_pos = None

        self.offroad_counter = 0
        self.offroad_max = 5   # numero di step consecutivi tollerati fuori strada

        self.prev_tile = None
        self.curr_tile = None
        self.left_turn_allowed = False
        self.last_obs = None

        self.current_tile_kind = None  # Inizializza a None o un valore di default appropriato
        self.current_left_turn_result = True # Inizializza a True come default
        self.current_delta_tile_movement = None # Inizializza a None

    def seed(self, seed=None):

        return self.env.seed(seed)
    
    def left_turn_available(self, prev, curr):
        """
        Determina se la svolta a sinistra è disponibile nell'incrocio corrente,
        in base alla tile di provenienza e al tipo di incrocio.
        """

        # prev = getattr(self, "prev_tile", None)
        # curr = getattr(self, "curr_tile", None)
        # if prev is None or curr is None:
            # return False
        
        prev_i, prev_j = prev
        curr_i, curr_j = curr
        # delta = (curr_i - prev_i, curr_j - prev_j)

        # tile_type si può prendere da curr_tile_info se lo salvi, ad esempio:
        # tile_info = self.env.unwrapped._map_tile_dict.get(curr, None)
        # tile_type = tile_info["kind"] if tile_info else None

        # i, j = self.env.unwrapped.get_grid_coords(self.env.unwrapped.curr)
        # 'curr' è già la tupla (i, j) che rappresenta la tile corrente
        curr_i, curr_j = curr
        tile_info = self.env.unwrapped._get_tile(curr_i, curr_j)
        tile_type1 = self.env.unwrapped._get_tile(curr_i, curr_j)
        tile_type = tile_type1["kind"] if tile_info else None


        # Se siamo in una 4-way, la svolta a sinistra è sempre disponibile
        if tile_type == "4way":
            return True, (curr_i - prev_i, curr_j - prev_j)
        elif tile_type == "3way_left" or tile_type == "3way_right":
            # Regole per 3-way sinistra — svolta possibile solo da certe direzioni
            valid_entries_for_left_turn = {
                (2, 0): [(3, 0), (2, 1)],  # da sud
                (2, 4): [(1, 4), (2, 3)],  # da est
                (4, 2): [(3, 2), (4, 3)], # da nord
                (0, 2): [(1, 2), (0, 1)]# da ovest
            }

            if curr in valid_entries_for_left_turn:
                return prev in valid_entries_for_left_turn[curr], (curr_i - prev_i, curr_j - prev_j)
            else:
                # Se non è una direzione valida per la svolta a sinistra, non è permessa
                return False, None
        else:
            # In altri casi la svolta non è permessa
            return False, None
        
    def right_turn_available(self, prev, curr):
        """
        Determina se la svolta a destra è disponibile nell'incrocio corrente,
        in base alla tile di provenienza e al tipo di incrocio.
        """
        
        prev_i, prev_j = prev
        curr_i, curr_j = curr

        # i, j = self.env.unwrapped.get_grid_coords(self.env.unwrapped.curr)
        # 'curr' è già la tupla (i, j) che rappresenta la tile corrente

        tile_info = self.env.unwrapped._get_tile(curr_i, curr_j)
        tile_type1 = self.env.unwrapped._get_tile(curr_i, curr_j)
        tile_type = tile_type1["kind"] if tile_info else None


        # Se siamo in una 4-way, la svolta a destra è sempre disponibile
        if tile_type == "4way":
            return True, (curr_i - prev_i, curr_j - prev_j)
        elif tile_type == "3way_left" or tile_type == "3way_right":
            # Regole per 3-way sinistra — svolta possibile solo da certe direzioni
            valid_entries_for_right_turn = {
                (2, 0): [(2, 1), (1, 0)], 
                (0, 2): [(0, 3), (1, 2)], 
                (4, 2): [(3, 1), (4, 1)],
                (2, 4): [(3, 4), (2, 3)]
            }

            if curr in valid_entries_for_right_turn:
                return prev == valid_entries_for_right_turn[curr], (curr_i - prev_i, curr_j - prev_j)
            else:
                # Se non è una direzione valida per la svolta a sinistra, non è permessa
                return False, None
        else:
            # In altri casi la svolta non è permessa
            return False, None

    def stop_line_detected(self, img):
        hsv   = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        # rosso chiaro in Duckietown
        lower = np.array([0, 100, 100])
        upper = np.array([10, 255, 255])
        mask1 = cv2.inRange(hsv, lower, upper)
        lower = np.array([160, 100, 100])
        upper = np.array([180, 255, 255])
        mask2 = cv2.inRange(hsv, lower, upper)
        mask  = cv2.bitwise_or(mask1, mask2)
        # ROI in basso (davanti al robot)
        h, w = mask.shape
        roi   = mask[int(h*0.6):h, int(w*0.2):int(w*0.8)]
        return cv2.countNonZero(roi) > (roi.size * 0.05)
        
    def reset(self, seed=None, options=None):
        if seed is not None:
            self.env.seed(seed)
            np.random.seed(seed)
            random.seed(seed)
        self.good_steps = 0

        img = self.env.reset()[0]

        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        obs_img = cv2.resize(img, (400, 300))

        self.last_pos = self.env.cur_pos.copy()

        self.prev_tile = self.env.unwrapped.get_grid_coords(self.env.cur_pos)
        self.current_tile = self.prev_tile # Inizialmente prev e current sono uguali

        print(f"Posizione iniziale: {self.env.cur_pos}, Angolo: {self.env.cur_angle}")

        # La goal_region non viene calcolata qui, ma nel metodo step
        self.goal_tile = None
        self.goal_center = None # Verrà calcolata una volta definita la goal_center

        self.in_intersection = False
        self.crossed_intersection = False
        self.stop_detected = False

        try:
            lane_pos = self.env.get_lane_pos2(self.env.cur_pos, self.env.cur_angle)
            lateral = lane_pos.dist
            angle = lane_pos.angle_deg / 180.0
            dot_dir = lane_pos.dot_dir

        except Exception as e:
            print(f"[WARN] Could not compute lane pos: {e}")
            lateral = 0.0
            angle = 0.0
            dot_dir = 0.0
        stop_flag = 0.0

        # dist_to_goal e obs inizializzati senza goal_center
        # Verranno aggiornati dopo il calcolo di goal_center


        if self.goal_tile is not None:
            cur_tile = self.env.get_current_tile()
            dist_to_goal = np.linalg.norm(np.array(cur_tile) - np.array(self.goal_tile))
        else:
            dist_to_goal = -1.0  # Segnala "goal non ancora noto"

        tile_size = self.env.road_tile_size
        initial_dist_norm = dist_to_goal / (2*tile_size)
        i, j = self.env.unwrapped.get_grid_coords(self.env.cur_pos)
        tile_kind = self.env.unwrapped._get_tile(i, j)["kind"]
        tile_one_hot = self.tile_one_hot(tile_kind)

        # Componi l’osservazione finale (senza più stop_flag)
        initial_obs = np.concatenate(
            (np.array([lateral, angle, initial_dist_norm, dot_dir], dtype=np.float32), tile_one_hot),
            axis=0
        )

        self.prev_dist_to_goal = initial_dist_norm

        self.last_obs = initial_obs

        self.post_goal_counter = None

        return initial_obs, {} 

    def tile_one_hot(self, tile_kind):
        tile_types = ['straight', 'curve_left', 'curve_right', '3way_left', '3way_right', '4way']
        one_hot = np.zeros(len(tile_types), dtype=np.float32)
        if tile_kind in tile_types:
            idx = tile_types.index(tile_kind)
            one_hot[idx] = 1.0
        return one_hot

    def step(self, action):
        
        obs_to_return = self.last_obs if self.last_obs is not None else np.zeros(self.env.observation_space.shape, dtype=np.float32)

        if self.post_goal_counter is not None:
            self.post_goal_counter += 1
            if self.post_goal_counter >= 200:
                done = True
                info = {"post_goal_timeout": True}
                reward, done_extra, info_extra = self.compute_reward(action, None, None, None, None) # Adatta gli argomenti
                return obs, reward, done, False, info
            else:
                reward, _, info_extra = self.compute_reward(action, None, None, None, None) # Adatta gli argomenti
                return obs, reward, False, False, info

        action_map = {
            0: (0.0, 0.3), 1: (0.1, 0.3), 2: (0.2, 0.3), 3: (0.3, 0.3),
            4: (0.3, 0.2), 5: (0.3, 0.1), 6: (0.3, 0.0),
        }
        vl, vr = action_map[int(action)]

        raw_obs, _, done_env, info = self.env.step(np.array([vl, vr]))

        self.num_steps += 1

        obs = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32) # Valori default, adatta la dimensione

        img = self.env.render('rgb_array')
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        obs_img = cv2.resize(img, (400, 300))
        # Placeholder per stop_line_detected - adattalo alla tua implementazione
        # def stop_line_detected(img):
            # return False
        # stop_flag = 1.0 if stop_line_detected(obs_img) else 0.0

        try:
            self.lane_pos = self.env.get_lane_pos2(self.env.cur_pos, self.env.cur_angle)
            lateral = self.lane_pos.dist
            angle = self.lane_pos.angle_deg / 180.0
            lane_position = self.lane_pos
            dot_dir = self.lane_pos.dot_dir
        except:
            done = True
            # self.lane_pos = None
            # lane_position = None
            # lateral, angle, dot_dir = 0.0, 0.0, 0.0

        delta_lateral = 0.0
        if self.prev_lateral is not None and not self.in_intersection and lane_position is not None:
            delta_lateral = abs(self.lane_pos.dist - self.prev_lateral)
        
        self.prev_lateral = self.lane_pos.dist if self.lane_pos is not None else None
        self.delta_lateral = delta_lateral  # lo passi implicitamente a reward_straight 

        # Rilevo posizione e tile correnti
        pos = self.env.unwrapped.cur_pos
        curr_i, curr_j = self.env.unwrapped.get_grid_coords(pos)
        self.current_tile = (curr_i, curr_j)

        tile_info = self.env.unwrapped._get_tile(curr_i, curr_j)
        tile_kind = tile_info["kind"] if tile_info else None

        if self.num_steps % 100 == 0 and self.lane_pos is not None:
            print(f"[DEBUG] step={self.num_steps} | lateral={lateral:.2f}, angle={angle:.2f}, dot_dir={self.lane_pos.dot_dir:.2f}, tile_kind={tile_kind}")


        # --- Logica per il calcolo della goal_region ---
        if self.prev_tile != self.current_tile and self.current_tile in ["3way_left", "3way_right", "4way"]:
            # Una variazione nella tile è stata rilevata
            left_turn_result, delta_tile_movement = self.left_turn_available(self.prev_tile, self.current_tile)

            # Mappatura di delta_tile_movement a angoli assoluti in radianti
            # Convenzione angoli assoluti: 0=nord, pi/2=ovest, pi=sud, -pi/2=est
            if delta_tile_movement == (1, 0):    # Movimento nord
                self.current_road_abs_angle = 0.0 
            elif delta_tile_movement == (-1, 0): # Movimento sud
                self.current_road_abs_angle = np.pi 
            elif delta_tile_movement == (0, - 1):  # Movimento ovest
                self.current_road_abs_angle = np.pi / 2 
            elif delta_tile_movement == (0, 1): # Movimento est
                self.current_road_abs_angle = -np.pi / 2 
            else:
                # Se non è un movimento rettilineo (es. transizione in una curva o incrocio con svolta)
                self.current_road_abs_angle = None #

            if not left_turn_result:
                # Se la svolta a sinistra non è permessa, termina e ritorna False
                # info["left_turn_not_available"] = True
                pass
            else:
                
                direzione = {
                    (1, 0): (self.current_tile[0], self.current_tile[1] - 1), # sud-nord
                    (-1, 0): (self.current_tile[0], self.current_tile[1] + 1), # nord-sud
                    (0, 1): (self.current_tile[0] + 1, self.current_tile[1]), # ovest-est
                    (0, -1): (self.current_tile[0] - 1, self.current_tile[1]) # est-ovest                             }
                }                   
                
                if delta_tile_movement in direzione:
                    self.goal_tile = tuple(direzione[delta_tile_movement])
                    tile_size = self.env.road_tile_size
                    gx, gy = self.goal_tile
                    self.goal_center = np.array([gx + 0.5, gy + 0.5]) * tile_size
                    print(f"Goal tile calcolata: {self.goal_tile}, Goal center: {self.goal_center}")

        # --- Fine logica per il calcolo della goal_region ---

        # 7️⃣ Distanza normalizzata dalla goal_center (ora calcolata solo se goal_center esiste)
        dist_to_goal = 0.0
        if self.goal_center is not None:
            pos2d = np.array(self.env.cur_pos)[[0, 2]]
            dist_to_goal = np.linalg.norm(pos2d - self.goal_center)
        else:
            dist_to_goal = 1.0
        tile_size = self.env.road_tile_size
        dist_norm = dist_to_goal / (2 * tile_size)

        # 8️⃣ Composizione osservazione vettoriale
        obs = np.concatenate([
            np.array([lateral, angle, dist_norm, dot_dir], dtype=np.float32),  # 3 valori continui
            self.tile_one_hot(tile_kind)                         # 6 valori one-hot
        ])


        # 4️⃣ Gestione flag intersection (solo se left_turn_allowed è True e goal_tile è stata calcolata)
        if tile_kind in ["3way_left", "4way", "3way_right"] and self.left_turn_allowed and self.goal_tile is not None:
            center = np.array([curr_i + 0.5, curr_j + 0.5]) # Usa curr_i, curr_j
            pos2d = np.array(self.env.cur_pos)[[0, 2]] # Usa la posizione corrente
            d2c = np.linalg.norm(pos2d - center)
            if d2c < 0.25:
                self.in_intersection = True
            if d2c > 0.5 and getattr(self, "in_intersection", False):
                self.in_intersection = False
                self.crossed_intersection = True
        # else:
            # vl, vr = action_map[3]  # forzi dritto

        # 9️⃣ Calcolo reward e done_extra
        reward, done_extra, info_extra = self.compute_reward(action, obs)

        # 🔟 Condizioni di terminazione
        done = done_env

        # 10.1 Off-road prolungato
        if lane_position is None:
            self.offroad_counter += 1
        else:
            self.offroad_counter = 0
        if self.offroad_counter >= self.offroad_max:
            done = True
            info['out_of_road'] = True

        # 10.2 Goal raggiunto via tile
        # Controlla solo se goal_tile è stata impostata
        if self.goal_tile is not None and self.current_tile == self.goal_tile and self.post_goal_counter is None:
            self.post_goal_counter = 0

        # 10.3 Attraversamento incrocio
        if getattr(self, "crossed_intersection", False):
            done = True
            info['crossed_intersection'] = True

        # 10.4 Altre condizioni definite da compute_reward
        done = done or done_extra
        if info_extra:
            info.update(info_extra)

        self.good_steps += 1

        # Aggiorna prev_tile per il prossimo step
        self.prev_tile = self.current_tile
        self.last_obs = obs

        return obs, reward, done, False, info

    def render(self, mode='human'):
        # obs = self.env.render(mode='rgb_array')

        if mode == 'human':
            # Usa la finestra interattiva di Duckietown
            return self.env.render(mode='human')
        else:
            obs = self.env.render(mode='rgb_array')
            # obs = self._highlight_goal_tile(obs)
            return obs

    def reward_straight(self, action, obs):
        lateral, angle, dist_norm, dot_dir = obs[:4]
        tile_one_hot = obs[4:]
        reward = 0.0

        # 1. Penalità per offset laterale dalla corsia
        reward -= abs(lateral) * 3.0

        if dot_dir < 0.0:
            dot_dir *= -1

        # 2. Premia l'allineamento alla corsia

        reward += 10.0 * (dot_dir ** 3)  # più centrato, più alto il reward
        
        # if not self.in_intersection:
            # reward += 5 * dot_dir  # più centrato, più alto il reward
            # if dot_dir > 0.98:
                # reward += 3.0 * dot_dir

        # 5. Bonus extra se perfettamente centrato, orientato e dritto
        # if abs(lateral) < 0.005 and dot_dir > 0.99 and action == 3:
            # reward += 7.0


        # 6. Penalità per oscillazioni laterali (variazione da step precedente)
        if hasattr(self, "delta_lateral") and not self.in_intersection:
            reward -= 3.0 * self.delta_lateral

        return reward

    def reward_curve(self, action, obs):

        lateral, angle, dist_norm, dot_dir = obs[:4]
        tile_one_hot = obs[4:]

        tile_kind_idx = int(np.argmax(tile_one_hot))
        tile_kind = self.tile_kinds[tile_kind_idx]

        reward = 0.0
        reward -= abs(lateral) * 10.0
        # reward -= abs(min(0, lateral)) * 2.0
        #if angle > 0.1: reward -= 3.0
        #elif -0.1 < angle <= 0.1: reward -= 1.0
        #elif angle < -0.2: reward += 1.0  # bonus per buona sterzata

        #elif tile_kind == "curve_right":
        #reward -= min(0, lateral) * 5.0
        #reward -= abs(max(0, lateral)) * 2.0
        #if angle < -0.1: reward -= 3.0
        #elif -0.1 <= angle < 0.1: reward -= 1.0
        #elif angle > 0.2: reward += 1.0

        return reward
    
    def reward_intersection(self, action, obs):
        lateral, angle, dist_norm, dot_dir = obs[:4]
        tile_one_hot = obs[4:]
        reward = 0.0
        reward -= 3.0 * abs(lateral)

        if action in [3, 4, 5, 6]:
            reward -= 6.0
        elif action == 0:  # sinistra
            reward += 6.0
        # else:
           #reward -= abs(angle) * 0.2

        delta = self.prev_dist_to_goal - dist_norm
        reward += delta * 5.0
        self.prev_dist_to_goal = dist_norm

        # if abs(lateral) > 0.4:
            # reward -= 5.0

        return reward

    def compute_reward(self, action, obs):
        import numpy as np

        lateral, angle, dist_norm, dot_dir = obs[:4]
        tile_one_hot = obs[4:]
        done = False
        info_extra = {}
        reward = 0.0

        if lateral > 0.12 or lateral < - 0.35:
            return -20.0, True, {"out_of_lane": True}
        
        start_tile = self.env.unwrapped.get_grid_coords(self.env.cur_pos)
        i, j = start_tile

        tile_kind_idx = int(np.argmax(tile_one_hot))
        tile_kind = self.tile_kinds[tile_kind_idx]
        agent_abs_angle = self.env.unwrapped.cur_angle

        if not hasattr(self, "prev_dist_to_goal"):
            self.prev_dist_to_goal = dist_norm

        if tile_kind == "straight":
            # reward += self.reward_straight(action, lateral, angle, agent_abs_angle, start_tile)
            reward = self.reward_straight(action, obs)
        elif tile_kind in ["curve_left", "curve_right"]:
            reward += self.reward_curve(action, obs)
        elif tile_kind in ["3way_left", "3way_right", "4way"]:
            reward += self.reward_intersection(action, obs)
            self.prev_dist_to_goal = dist_norm

        # Penalità per oscillazione
        if self.last_action in [0, 2] and action in [0, 2] and self.last_action != action:
            self.oscillation_count += 1
        else:
            self.oscillation_count = 0
        self.last_action = action
        if self.oscillation_count >= 3:
            reward -= 1.0

        # Raggiunto obiettivo
        if start_tile == self.goal_tile:
            reward += 50.0
            done = True
            info_extra["goal_reached"] = True

        return reward, done, info_extra

class LaneFollower:
    def __init__(self, kp=0.8, ki=0.0, kd=0.2,
                 target_speed=0.3, min_speed=0.1,
                 max_omega=1.0):
        # PID gains
        self.kp = kp
        # Permette al robot di correggere rapidamente la sua traiettoria quando si allontana dalla linea

        self.ki = ki 
        #  Consente al robot di correggere derive lente o errori sistematici che il solo controllo
        #  proporzionale non riuscirebbe ad eliminare completamente.

        self.kd = kd
        # Aiuta a prevenire un'eccessiva correzione anticipando il comportamento futuro dell'errore. 

        # PID state
        self.integral = 0.0
        self.last_error = 0.0
        # Low-pass filter for derivative term
        self.deriv_filtered = 0.0
        self.alpha = 0.1  # filter weight
        # Speed settings
        self.target_speed = target_speed # Imposta la velocità di crociera del robot
        self.min_speed = min_speed  # fallback speed when lane is lost
        # Steering limit
        self.max_omega = max_omega # Previene sterzate eccessive e potenzialmente instabili

    def process(self, image, dt):
        """
        Main pipeline: image -> (v_linear, omega)
        If lane is detected: PID control.
        If lane lost: maintain straight at min_speed.
        """
        error = self.calculate_lane_error(image)
        if error is None:
            # Lane lost: vai piano e sterza verso l'ultima direzione nota
            steer = np.clip(getattr(self, 'last_error', 0.0), -self.max_omega, self.max_omega)
            return float(self.min_speed), steer

        # Lane detected: normal PID
        speed = self.target_speed
        if abs(error) > 0.5:
            speed *= 0.3  # reduce speed in tight turns
            kp = self.kp * 1.5 # aumenta temporaneamente il guadagno proporzionale
        else:
            kp = self.kp

        speed = max(self.min_speed, speed)

        # PID con guadagno proporzionale adattivo

        # CALCOLA OMEGA USANDO IL PID COMPLETO
        # Aggiorna integral e deriv_filtered all'interno di compute_control o qui
        self.integral += error * dt # Aggiorna il termine integrale
        deriv = (error - self.last_error) / dt if dt > 1e-6 else 0.0
        self.deriv_filtered = self.alpha * deriv + (1 - self.alpha) * self.deriv_filtered # Aggiorna il termine derivativo filtrato

        omega = kp * error + self.ki * self.integral + self.kd * self.deriv_filtered
        omega = float(np.clip(omega, -self.max_omega, self.max_omega))
        self.last_error = error
        return float(self.min_speed), omega

    def calculate_lane_error(self, image):
        """
        Compute lateral deviation error:
         1) ROI mask to remove off-road
         2) HSV threshold for yellow & white lanes
         3) Morphological closing
         4) Bird's-eye warp
         5) Canny edges & centroid
        Returns error in [-1,1] or None if no line.
        """
        h, w = image.shape[:2]
        # ROI: metà destra
        right_half = image[:, int(w*0.5):]

        # HSV threshold per bianco (linea continua destra)
        hsv = cv2.cvtColor(right_half, cv2.COLOR_RGB2HSV)
        white = cv2.inRange(hsv, np.array([0, 0, 200]), np.array([180, 30, 255]))

        # Morphology
        blur = cv2.GaussianBlur(white, (5,5), 0)
        kernel = np.ones((5,5), np.uint8)
        clean = cv2.morphologyEx(blur, cv2.MORPH_CLOSE, kernel)

        # Centroid della linea bianca
        M = cv2.moments(clean)
        if M['m00'] > 0:
            cx = M['m10'] / M['m00']
            # Errore rispetto al bordo destro (non al centro!)
            error = (cx - (right_half.shape[1] - 1)) / (right_half.shape[1] - 1)
            self.last_error = error
            return float(error)
        return None

    def compute_control(self, error, dt):
        """
        PID control with filtered derivative.
        dt: time interval.
        """
        # Integral
        self.integral += error * dt
        # Derivative
        deriv = (error - self.last_error) / dt if dt > 1e-6 else 0.0
        self.deriv_filtered = self.alpha * deriv + (1 - self.alpha) * self.deriv_filtered
        # PID output
        omega = self.kp * error + self.ki * self.integral + self.kd * self.deriv_filtered
        return omega

class RewardHistoryCallback(BaseCallback):
    def __init__(self, reward_history, verbose=0):
        super().__init__(verbose)
        self.reward_history = reward_history

    def _on_step(self) -> bool:
        # self.locals['rewards'] è la reward dell'ultimo step
        self.reward_history.append(self.locals['rewards'][0])
        return True
    
class EpisodeRewardCallback(BaseCallback):
    def __init__(self, episode_rewards, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = episode_rewards
        self.current_reward = 0.0

    def _on_step(self) -> bool:
        self.current_reward += self.locals['rewards'][0]
        if self.locals.get('dones', [False])[0]:
            self.episode_rewards.append(self.current_reward)
            self.current_reward = 0.0
        return True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-name", default="Duckietown-udem1-v0")
    parser.add_argument("--map-name", default="4way") # Udem1
    parser.add_argument("--distortion", action="store_true")
    parser.add_argument("--frame-skip", default=1, type=int)
    parser.add_argument("--seed", default=1, type=int)
    args = parser.parse_args()

    # Crea una cartella con la data nel formato 'DDMM'
    today_folder = datetime.now().strftime('%d%m')
    save_dir = os.path.join(os.getcwd(), today_folder)
    os.makedirs(save_dir, exist_ok=True)

    # tile = random.choice(possible_starts)

    # Setup environment
    if args.env_name and "Duckietown" in args.env_name:
        env = DuckietownEnv(
            map_name="4way",
            domain_rand=False,
            draw_curve=True,
            draw_bbox=False,
            distortion=args.distortion,
            seed=args.seed,
            frame_skip=args.frame_skip,
            max_steps=800,
            camera_rand=False,
            dynamics_rand=False,
            randomize_maps_on_reset=False,
            # user_tile_start=tile,
  
        )

    else:
        env = gym.make(args.env_name)

    env = DuckieRLWrapper(env)  # Wrappare l'ambiente per RL
    env = DummyVecEnv([lambda: env])  # Necessario per stable-baselines3

    raw_env = env.envs[0].env

    start_time = time.time()

    model = DQN(
        policy="MlpPolicy",
        env=env,
        learning_rate=1e-5,
        buffer_size=100_000,               # AUMENTATO
        learning_starts=10000,              # Parte dopo aver riempito un po' di buffer
        batch_size=64, 
        tau=0.005,                           # Soft update, può restare così
        gamma=0.99,
        train_freq=4,
        gradient_steps=2,
        target_update_interval=1000,
        exploration_fraction=0.1,
        exploration_final_eps=0.01,
        verbose=1,
        # tensorboard_log="./dqn_duckie_tensorboard/"
    )

    # 1 milione di step di training

    # Avvia il training vero
    # Training in modalità batch
    num_batches = 5
    timesteps_per_batch = 100000

    # Lista per memorizzare le ricompense episodiche
    episode_rewards = []
    callback = EpisodeRewardCallback(episode_rewards)

    best_avg_reward = -float("inf")  # inizializza la miglior reward
    best_model_path = os.path.join(save_dir, "best_model")  # path temporaneo
    final_model_path = os.path.join(save_dir, "dqn_duckietown_model")  # path finale

    for i in range(num_batches):
        print(f"Avvio del batch {i + 1} di {num_batches}...")
        model.learn(total_timesteps=100000, callback=callback, reset_num_timesteps=False, log_interval=10)

        # Calcolo reward media per episodio (ultimi 10 episodi)
        if len(episode_rewards) > 0:
            mean_ep_reward = np.mean(episode_rewards[-10:])
            print(f"Batch {i + 1} completato. Reward media ultimi 10 episodi: {mean_ep_reward:.2f}")

            # Se è il miglior risultato, salva il modello
            if mean_ep_reward > best_avg_reward:
                best_avg_reward = mean_ep_reward
                model.save(best_model_path)
                print(f"✅ Nuovo best model salvato con reward media: {mean_ep_reward:.2f}")
        else:
            print(f"Batch {i + 1} completato.")

    # Alla fine copia il miglior modello nel file finale
    import shutil
    shutil.copyfile(best_model_path + ".zip", final_model_path + ".zip")
    print("📦 Modello finale salvato come dqn_duckietown_model (corrisponde al best model)")

    # Grafico delle ricompense durante il training
    if len(episode_rewards) > 0:
        plt.figure()
        plt.plot(moving_average(episode_rewards, 10))
        plt.xlabel('Episodi')
        plt.ylabel('Ricompensa media (finestra 10)')
        plt.title('Ricompensa media per episodio durante il training')
        plt.grid()
        # Salva il grafico in automatico
        reward_plot_path = os.path.join(save_dir, "reward_training.png")
        plt.savefig(reward_plot_path)
        print(f"Grafico delle reward salvato in: {reward_plot_path}")
        plt.show()
    else:
        print("Nessuna ricompensa registrata durante il training.")

    end_time = time.time()
    duration = end_time - start_time
    print(f"Training durato {duration:.2f} secondi ({duration/60:.2f} minuti)")

    def run_test_with_render(env, model, raw_env, max_steps=500):
        test_rewards = []
        frames = []
        step_counter = {'step': 0}
        obs = env.reset()
        raw_env.reset()

        def update(dt):
            if step_counter['step'] >= max_steps:
                pyglet.app.exit()
                return
            nonlocal obs
            action, _ = model.predict(obs)
            obs_, reward, done, info = env.step(action)
            test_rewards.append(float(reward))
            frame = raw_env.render('rgb_array')
            frames.append(frame)
            raw_env.render(mode='human')
            obs = obs_
            step_counter['step'] += 1
            if done:
                print("Fine episodio di test.")
                obs = env.reset()
                step_counter['step'] = 0
                # pyglet.app.exit()

        pyglet.clock.schedule_interval(update, 1.0 / raw_env.unwrapped.frame_rate)
        pyglet.app.run()
        return frames, test_rewards

    # Dopo plt.show() del training:
    frames, test_rewards = run_test_with_render(env, model, raw_env, max_steps=500)

    if len(test_rewards) > 0:
        window = min(len(test_rewards), 100)
        plt.plot(moving_average(test_rewards, window))
        plt.xlabel('Passi')
        plt.ylabel('Ricompensa media (finestra 10)')
        plt.title('Performance dell’agente DQN nel test')
        plt.grid()
        plt.show()
    else:
        print("Nessuna ricompensa registrata nel test.")# Salva il modello finale 
    
    env.close()

    try:
        model.save("dqn_duckietown_model")
        print("Modello salvato correttamente come dqn_duckietown_model")
    except Exception as e:
        print(f"Errore durante il salvataggio del modello: {e}")
   
if __name__=='__main__':
    main()
