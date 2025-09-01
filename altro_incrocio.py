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
import yaml
import numpy as np

def moving_average(x, w):

    x = np.array(x).flatten()
    return np.convolve(x, np.ones(w), 'valid') / w

class DuckieRLWrapper(gym.Env):

    def __init__(self, env):
        super(DuckieRLWrapper, self).__init__()
        self.env = env
        
        self.tile_kinds = ["straight", "curve_left", "curve_right", "3way_left", "3way_right", "4way"]
        self.action_types = ['left', 'right', 'straight']

        # initial_pos = [[1.3455,  0.,      0.73125], # ovest-est
                       # [0.73125, 0.,      1.5795 ], # sud-nord
                       # [1.5795,  0.,      2.19375], # est-ovest
                       # [2.19375, 0.,      1.3455 ]] # nord-sud
        
        # self.env.cur_pos = random.choice(initial_pos)

        # self.env.cur_angle = 0.0

        self.tile_counter = 0

        self.direzione = None

        flag = None

        self.best_curve = None

        self.max_offset = 0.35

        self.max_angle = 0.157

        self.lane_width = 0.22

        self.current_target_curve = None  # Inizializza a None o un valore di default appropriato

        self.current_road_abs_angle = None

        self.prev_lateral = None
        # Scegli una tile iniziale casuale dalla lista
        # start_pos = random.choice(initial_pos)
        self.num_steps = 0

        # Imposta la posizione e angolo dell'agente
        # self.env.cur_pos = np.array(start_pos)

        # self.env.cur_angle = 0.0

        self.oscillation_count = 0
        self.last_action = None

        # Per il tracking del movimento verso la goal
        self.last_pos = None

        self.normalized_dist = 0.0

        # modificato dimensione da (480, 680, 3) per ridurre il carico computazionale

        # hybrid observation: image  + vector features
        # self.observation_space = spaces.Box(
            # low=np.array([-1., -np.pi, 0.,   0.]), 
            # high=np.array([ 1.,  np.pi, 1.,   2.]),  # 2 = due tile di distanza massima
            # dtype=np.float32
        # )

        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(11,), dtype=np.float32
        )

        self.action_space = spaces.Discrete(7)  # Azioni: sinistra, dritto, destra (per semplificare)

        # possibile implementazione in spazio continuo
        self.good_steps = 0 
        self.goal_tile = None
        self.last_action = None
        self.oscillation_count = 0
        self.lane_pos = None
        self.curve_type = None
        

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
        tile_type = tile_info["kind"] if tile_info else None


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
        # tile_type1 = self.env.unwrapped._get_tile(curr_i, curr_j)
        tile_type = tile_info["kind"] if tile_info else None


        # Se siamo in una 4-way, la svolta a destra è sempre disponibile
        if tile_type == "4way":
            return True, (curr_i - prev_i, curr_j - prev_j)
        elif tile_type == "3way_left":
            # Regole per 3-way sinistra — svolta possibile solo da certe direzioni
            valid_entries_for_right_turn = {
                (2, 0): [(2, 1), (1, 0)], 
                (0, 2): [(0, 3), (1, 2)], 
                (4, 2): [(3, 2), (4, 1)],
                (2, 4): [(3, 4), (2, 3)]
            }

            if curr in valid_entries_for_right_turn:
                if prev in valid_entries_for_right_turn[curr]:
                    return True, (curr_i - prev_i, curr_j - prev_j)
                else:
                    return False, None
        
    def go_straight(self, prev, curr):
        """
        Determina se la svolta a destra è disponibile nell'incrocio corrente,
        in base alla tile di provenienza e al tipo di incrocio.
        """
        
        prev_i, prev_j = prev
        curr_i, curr_j = curr

        # i, j = self.env.unwrapped.get_grid_coords(self.env.unwrapped.curr)
        # 'curr' è già la tupla (i, j) che rappresenta la tile corrente

        tile_info = self.env.unwrapped._get_tile(curr_i, curr_j)
        # tile_type1 = self.env.unwrapped._get_tile(curr_i, curr_j)
        tile_type = tile_info["kind"] if tile_info else None


        # Se siamo in una 4-way, la svolta a destra è sempre disponibile
        if tile_type == "4way":
            return True, (curr_i - prev_i, curr_j - prev_j)
        elif tile_type == "3way_left":
            # Regole per 3-way sinistra — svolta possibile solo da certe direzioni
            valid_entries_for_straight = {
                (2, 0): [(3, 0), (1, 0)], 
                (0, 2): [(0, 3), (0, 1)], 
                (4, 2): [(4, 3), (4, 1)],
                (2, 4): [(3, 4), (1, 4)]
            }

            if curr in valid_entries_for_straight:
                if prev in valid_entries_for_straight[curr]:
                    return True, (curr_i - prev_i, curr_j - prev_j)
                else:
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

        self.tile_counter = 0

        try:
            lane_pos = self.env.get_lane_pos2(self.env.cur_pos, self.env.cur_angle)
            lateral = lane_pos.dist
            angle = lane_pos.angle_rad # / 180.0
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
        initial_dist_curve = dist_to_goal / (2*tile_size)

        i, j = self.env.unwrapped.get_grid_coords(self.env.cur_pos)
        tile_kind = self.env.unwrapped._get_tile(i, j)["kind"]
        tile_one_hot = self.tile_one_hot(tile_kind)

        # if self.prev_tile != self.current_tile:
            # if tile_kind in ["3way_left", "4way"]:
                # in_intersection = 1
                # print(self.prev_tile, self.current_tile, tile_kind)
                # Una variazione nella tile è stata rilevata
                # left_turn_result, delta_tile_movement_left = self.left_turn_available(self.prev_tile, self.current_tile)
                # right_turn_result, delta_tile_movement_right = self.right_turn_available(self.prev_tile, self.current_tile)
                # straight_result, delta_tile_movement_straight = self.go_straight(self.prev_tile, self.current_tile)

                # chose_direction = []
                # if left_turn_result: chose_direction.append(["left", delta_tile_movement_left])
                # if right_turn_result: chose_direction.append(["right", delta_tile_movement_right])
                # if straight_result: chose_direction.append(["straight", delta_tile_movement_straight])

                # self.direzione, delta_tile_movement = random.choice(chose_direction)
        
        if tile_kind == "straight":
            self.direzione = "straight"

        action_one_hot = self.action_one_hot(self.direzione)

        # if tile_kind in ["3way_left", "3way_right", "4way"]:
            # in_intersection = 1
        # else:
            # in_intersection = 0

        # Componi l’osservazione finale (senza più stop_flag)
        initial_obs = np.concatenate(
            (np.array([lateral, angle, initial_dist_norm, initial_dist_curve], dtype=np.float32), tile_one_hot, action_one_hot),
            axis=0
        )

        self.prev_dist_to_goal = initial_dist_norm
        self.prev_dist_curve = initial_dist_curve

        self.last_obs = initial_obs

        self.post_goal_counter = None

        # initial_pos = [[1.3455,  0.,      0.73125], # ovest-est
                       # [0.73125, 0.,      1.5795 ], # sud-nord
                       # [1.5795,  0.,      2.19375], # est-ovest
                       # [2.19375, 0.,      1.3455 ]] # nord-sud
        
        # self.env.cur_pos = random.choice(initial_pos)

        # self.env.cur_angle = 0.0

        # self.env.unwrapped.step(np.array([0, 0]))
                               
        return initial_obs, {} 

    def tile_one_hot(self, tile_kind):
        tile_types = ['straight', 'curve_left', '3way_left', '4way']
        one_hot = np.zeros(len(tile_types), dtype=np.float32)
        if tile_kind in tile_types:
            idx = tile_types.index(tile_kind)
            one_hot[idx] = 1.0
        return one_hot

    def action_one_hot(self, direzione):
        action_types = ['left', 'right', 'straight']
        one_hot = np.zeros(len(action_types), dtype=np.float32)
        if direzione in action_types:
            idx = action_types.index(direzione)
            one_hot[idx] = 1.0
        return one_hot
    
    def bezier_curve(self, points, n_points=100):
        """
        Genera una curva Bézier cubica da 4 punti di controllo.

        Args:
            points (np.ndarray): Array shape (4, 3) con i punti di controllo (in 3D).
            n_points (int): Numero di punti da generare lungo la curva.

        Returns:
            np.ndarray: Array shape (n_points, 3) con i punti lungo la curva.
        """
        assert points.shape == (4, 3), "Sono richiesti esattamente 4 punti di controllo in 3D"

        t = np.linspace(0, 1, n_points).reshape(-1, 1)  # Shape (n_points, 1)
        
        # Formula Bézier cubica:
        curve = (1 - t)**3 * points[0] + \
                3 * (1 - t)**2 * t * points[1] + \
                3 * (1 - t) * t**2 * points[2] + \
                t**3 * points[3]
        
        return curve
    
    def step(self, action):

        flag = None
        
        obs_to_return = self.last_obs if self.last_obs is not None else np.zeros(self.env.observation_space.shape, dtype=np.float32)
        
        # print("metodo step")

        # commentata per fase test
        # if self.post_goal_counter is not None:
            # self.post_goal_counter += 1
            # if self.post_goal_counter >= 200:
                # done = True
                # info = {"post_goal_timeout": True}
                # reward, done_extra, info_extra = self.compute_reward(action, None, None, None, None) # Adatta gli argomenti
                # return obs, reward, done, False, info
            # else:
                # reward, _, info_extra = self.compute_reward(action, None, None, None, None) # Adatta gli argomenti
                # return obs, reward, False, False, info

        action_map = {
            0: (0.0, 0.3), 1: (0.1, 0.3), 2: (0.2, 0.3), 3: (0.3, 0.3),
            4: (0.3, 0.2), 5: (0.3, 0.1), 6: (0.3, 0.0),
        }
        vl, vr = action_map[int(action)]


        raw_obs, _, done_env, info = self.env.step(np.array([vl, vr]))

        self.num_steps += 1

        # obs = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32) # Valori default, adatta la dimensione

        img = self.env.render('rgb_array')
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        obs_img = cv2.resize(img, (400, 300))

        try:
            self.lane_pos = self.env.get_lane_pos2(self.env.cur_pos, self.env.cur_angle)
            lateral = self.lane_pos.dist
            angle = self.lane_pos.angle_rad # / 180.0
            lane_position = self.lane_pos
            dot_dir = self.lane_pos.dot_dir
            pos = self.env.unwrapped.cur_pos
            curr_i, curr_j = self.env.unwrapped.get_grid_coords(pos)
            self.current_tile = (curr_i, curr_j)
            reward = 0.0

            tile_info = self.env.unwrapped._get_tile(curr_i, curr_j)
            tile_kind = tile_info["kind"] if tile_info else None
            if tile_kind in ["straight", "curve_left"]:
                in_intersection = 0
            else:
                in_intersection = 1
        except Exception as e:
                self.lane_pos = None
                reward = -1.0  # penalità
                done = True
                obs = self.observation_space.sample()  # placeholder o zero array
                terminated = done  # il tuo "done" logico
                truncated = False  # oppure True se vuoi gestire timeout
                return obs, reward, terminated, truncated, {}

        # Rilevo posizione e tile correnti
        pos = self.env.unwrapped.cur_pos
        curr_i, curr_j = self.env.unwrapped.get_grid_coords(pos)
        self.current_tile = (curr_i, curr_j)

        tile_info = self.env.unwrapped._get_tile(curr_i, curr_j)
        tile_kind = tile_info["kind"] if tile_info else None

        # if self.num_steps % 100 == 0 and self.lane_pos is not None:
            # print(f"[DEBUG] step={self.num_steps} | lateral={lateral:.2f}, angle={angle:.2f}, dot_dir={self.lane_pos.dot_dir:.2f}, tile_kind={tile_kind}")

        # --- Logica per il calcolo della goal_region nell'incrocio ---
        if self.prev_tile != self.current_tile and self.prev_tile is not None:

            self.tile_counter += 1
            print (f"tile passed until now: {self.tile_counter}")

            if self.goal_tile is not None:
                if (self.current_tile == self.goal_tile):
                    reward += 2.5
                else:
                    reward -= 5.0
                    flag = True
                    info["wrong_way"] = True
                    print ("Wrong way!")

            if tile_kind in ["3way_left", "4way"]:
                in_intersection = 1
                # print(self.prev_tile, self.current_tile, tile_kind)
                # Una variazione nella tile è stata rilevata
                left_turn_result, delta_tile_movement_left = self.left_turn_available(self.prev_tile, self.current_tile)
                right_turn_result, delta_tile_movement_right = self.right_turn_available(self.prev_tile, self.current_tile)
                straight_result, delta_tile_movement_straight = self.go_straight(self.prev_tile, self.current_tile)

                chose_direction = []
                if left_turn_result: chose_direction.append(["left", delta_tile_movement_left])
                if right_turn_result: chose_direction.append(["right", delta_tile_movement_right])
                if straight_result: chose_direction.append(["straight", delta_tile_movement_straight])

                self.direzione, delta_tile_movement = random.choice(chose_direction)

                print(f"Previous tile: {self.prev_tile}, current tile: {self.current_tile}, moving with delta: {chose_direction[0][1]}")
                for i in chose_direction:
                    print(f"Possible direction {i[0]}")

                print(f"direction chosen: {self.direzione}")

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

                if self.direzione == "left":
                    direzione_left = {
                        (1, 0): (self.current_tile[0], self.current_tile[1] - 1), # sud-nord
                        (-1, 0): (self.current_tile[0], self.current_tile[1] + 1), # nord-sud
                        (0, 1): (self.current_tile[0] + 1, self.current_tile[1]), # ovest-est
                        (0, -1): (self.current_tile[0] - 1, self.current_tile[1]) # est-ovest                             }
                        }                   
                        
                    if delta_tile_movement in direzione_left:
                        self.goal_tile = tuple(direzione_left[delta_tile_movement])
                        tile_size = self.env.road_tile_size
                        # gx, gy = self.goal_tile                   
                        # self.goal_center = np.array([gx + 0.5, gy + 0.5]) * tile_size
                        gx, gy = self.current_tile
                        if delta_tile_movement == (1, 0):    # Movimento nord
                            self.goal_center = np.array([gx + 0.7, gy]) * tile_size
                        elif delta_tile_movement == (-1, 0): # Movimento sud
                            self.goal_center = np.array([gx + 0.3, gy + 1.0]) * tile_size 
                        elif delta_tile_movement == (0, - 1):  # Movimento ovest
                            self.goal_center = np.array([gx, gy + 0.3]) * tile_size
                        elif delta_tile_movement == (0, 1): # Movimento est
                            self.goal_center = np.array([gx + 1.0, gy + 0.7]) * tile_size 

                if self.direzione == "right":
                    direzione_right = {
                        (1, 0): (self.current_tile[0], self.current_tile[1] + 1), # sud-nord
                        (-1, 0): (self.current_tile[0], self.current_tile[1] - 1), # nord-sud
                        (0, 1): (self.current_tile[0] - 1, self.current_tile[1]), # ovest-est
                        (0, -1): (self.current_tile[0] + 1, self.current_tile[1]) # est-ovest 
                        }
                        
                    if delta_tile_movement in direzione_right:
                        self.goal_tile = tuple(direzione_right[delta_tile_movement]) 
                        tile_size = self.env.road_tile_size
                        # gx, gy = self.goal_tile
                        gx, gy = self.current_tile
                        if delta_tile_movement == (1, 0):    # Movimento nord
                            self.goal_center = np.array([gx + 0.3, gy + 1.0]) * tile_size
                        elif delta_tile_movement == (-1, 0): # Movimento sud
                            self.goal_center = np.array([gx + 0.7, gy]) * tile_size 
                        elif delta_tile_movement == (0, - 1):  # Movimento ovest
                            self.goal_center = np.array([gx + 1.0, gy + 0.7]) * tile_size
                        elif delta_tile_movement == (0, 1): # Movimento est
                            self.goal_center = np.array([gx, gy + 0.3]) * tile_size 
                
                if self.direzione == "straight":
                    direzione_straight = {
                        (1, 0): (self.current_tile[0] + 1, self.current_tile[1]), # sud-nord
                        (-1, 0): (self.current_tile[0] - 1, self.current_tile[1]), # nord-sud
                        (0, 1): (self.current_tile[0], self.current_tile[1] + 1), # ovest-est
                        (0, -1): (self.current_tile[0], self.current_tile[1] - 1) # est-ovest                             
                        }                   
                        
                    if delta_tile_movement in direzione_straight:
                        self.goal_tile = tuple(direzione_straight[delta_tile_movement])
                        tile_size = self.env.road_tile_size
                        gx, gy = self.current_tile
                        # self.goal_center = np.array([gx + 0.5, gy + 0.5]) * tile_size
                        if delta_tile_movement == (1, 0):    # Movimento nord
                            self.goal_center = np.array([gx + 1, gy + 0.7]) * tile_size
                        elif delta_tile_movement == (-1, 0): # Movimento sud
                            self.goal_center = np.array([gx, gy + 0.3]) * tile_size 
                        elif delta_tile_movement == (0, - 1):  # Movimento ovest
                            self.goal_center = np.array([gx + 0.7, gy]) * tile_size
                        elif delta_tile_movement == (0, 1): # Movimento est
                            self.goal_center = np.array([gx + 0.3, gy + 1.0]) * tile_size

                print(f"Goal tile calcolata: {self.goal_tile}, Goal center: {self.goal_center}, tyele_tipe: {tile_kind}")
                
                curves = self.env._get_curve(self.current_tile[0], self.current_tile[1])

                if curves is not None and len(curves) > 0:
                    # Estendi goal_center a 3D per confronto (Y=0)
                    goal_center_3d = np.array([self.goal_center[0], 0, self.goal_center[1]])

                    # Trova la curva che finisce più vicina al centro della goal_tile
                    self.best_curve = min(curves, key=lambda curve: (
                            np.linalg.norm(curve[-1] - goal_center_3d) +  # distanza dalla goal
                            0.5 * np.linalg.norm(curve[0] - self.env.cur_pos)  # distanza dall'attuale posizione
                        ))
                    
                    print(f"Curve choosen: {self.best_curve}")

                    # Salva la curva migliore
                    self.current_target_curve = self.best_curve
                else:
                    self.current_target_curve = None
                    print("[WARNING] Nessuna curva Bézier trovata nel tile attuale.")

        # --- Fine logica per il calcolo della goal_region ---

        # --- Logica per il calcolo della goal_region in curva ---
            if tile_kind == "curve_left":
                # print(self.prev_tile, self.current_tile, tile_kind)
                in_intersection = 0
                # Una variazione nella tile è stata rilevata

                delta_tile_movement = (curr_i - self.prev_tile[0], (curr_j - self.prev_tile[1]))

                # Mappatura di delta_tile_movement a angoli assoluti in radianti
                # Convenzione angoli assoluti: 0=nord, pi/2=ovest, pi=sud, -pi/2=est
                if delta_tile_movement == (1, 0): # movimento nord
                    self.current_road_abs_angle = 0.0
                elif delta_tile_movement == (-1, 0): # movimento sud
                    self.current_road_abs_angle = np.pi
                elif delta_tile_movement == (0, -1): # movimento ovest
                    self.current_road_abs_angle = np.pi / 2
                elif delta_tile_movement == (0, 1): # movimento est
                    self.current_road_abs_angle = -np.pi / 2
                else:
                    # Se non è un movimento rettilineo (es. transizione in una curva o incrocio con svolta)
                    self.current_road_abs_angle = None #

                direzione = {
                    (1, (1,0)): (self.current_tile[0], self.current_tile[1] - 1), # curva a sinistra
                    (2, (1,0)): (self.current_tile[0], self.current_tile[1] + 1), # curva a destra
                    (0, (-1,0)): (self.current_tile[0], self.current_tile[1] - 1), # curva a destra
                    (3, (-1,0)): (self.current_tile[0], self.current_tile[1] + 1), # curva a sinistra
                    (1, (0,1)): (self.current_tile[0] - 1, self.current_tile[1]), # curva a destra
                    (0, (0,1)): (self.current_tile[0] + 1, self.current_tile[1]), # curva a sinistra
                    (2, (0,-1)): (self.current_tile[0] - 1, self.current_tile[1]), # curva a sinistra
                    (3, (0,-1)): (self.current_tile[0] + 1, self.current_tile[1]) # curva a destra
                }

                possible_curve = {
                    (1, (1,0)): "left_curve",
                    (2, (1,0)): "right_curve",
                    (0, (-1,0)): "right_curve",
                    (3, (-1,0)): "left_curve",
                    (1, (0,1)): "right_curve",
                    (0, (0,1)): "left_curve",
                    (2, (0,-1)): "left_curve",
                    (3, (0,-1)): "right_curve"
                }             
                                    
                if (tile_info["angle"], (curr_i - self.prev_tile[0], curr_j - self.prev_tile[1])) in direzione:
                    self.goal_tile = tuple(direzione[(tile_info["angle"], (curr_i - self.prev_tile[0], curr_j - self.prev_tile[1]))])
                    tile_size = self.env.road_tile_size
                    gx, gy = self.current_tile

                    if delta_tile_movement == (1, 0) and possible_curve[(tile_info["angle"], (curr_i - self.prev_tile[0], curr_j - self.prev_tile[1]))] == "left_curve":
                        self.goal_center = np.array([gx + 0.7, gy]) * tile_size    # Movimento nord
                    if delta_tile_movement == (-1, 0) and possible_curve[(tile_info["angle"], (curr_i - self.prev_tile[0], curr_j - self.prev_tile[1]))] == "left_curve":
                        self.goal_center = np.array([gx + 0.3, gy + 1.0]) * tile_size    # Movimento sud
                    if delta_tile_movement == (0, -1) and possible_curve[(tile_info["angle"], (curr_i - self.prev_tile[0], curr_j - self.prev_tile[1]))] == "left_curve":
                        self.goal_center = np.array([gx, gy + 0.3]) * tile_size    # Movimento ovest
                    if delta_tile_movement == (0, 1) and possible_curve[(tile_info["angle"], (curr_i - self.prev_tile[0], curr_j - self.prev_tile[1]))] == "left_curve":
                        self.goal_center = np.array([gx + 1.0, gy + 0.7]) * tile_size    # Movimento est
                    if delta_tile_movement == (1, 0) and possible_curve[(tile_info["angle"], (curr_i - self.prev_tile[0], curr_j - self.prev_tile[1]))] == "right_curve":
                        self.goal_center = np.array([gx + 0.3, gy + 1.0]) * tile_size    # Movimento nord
                    if delta_tile_movement == (-1, 0) and possible_curve[(tile_info["angle"], (curr_i - self.prev_tile[0], curr_j - self.prev_tile[1]))] == "right_curve":
                        self.goal_center = np.array([gx + 0.7, gy]) * tile_size    # Movimento sud
                    if delta_tile_movement == (0, -1) and possible_curve[(tile_info["angle"], (curr_i - self.prev_tile[0], curr_j - self.prev_tile[1]))] == "right_curve":
                        self.goal_center = np.array([gx + 1.0, gy + 0.7]) * tile_size    # Movimento ovest
                    if delta_tile_movement == (0, 1) and possible_curve[(tile_info["angle"], (curr_i - self.prev_tile[0], curr_j - self.prev_tile[1]))] == "right_curve":
                        self.goal_center = np.array([gx, gy + 0.3]) * tile_size    # Movimento est
                    print(f"Goal tile calcolata: {self.goal_tile}, Goal center: {self.goal_center}")

                if (tile_info["angle"], (curr_i - self.prev_tile[0], curr_j - self.prev_tile[1])) in possible_curve:
                    self.curve_type = possible_curve[(tile_info["angle"], (curr_i - self.prev_tile[0], curr_j - self.prev_tile[1]))]
                    self.direzione = "left" if self.curve_type == "left_curve" else "right"

                curves = self.env._get_curve(self.current_tile[0], self.current_tile[1])

                if curves is not None and len(curves) > 0:
                    # posizione goal_center (x, 0, z)
                    goal_center_3d = np.array([self.goal_center[0], 0, self.goal_center[1]])

                    self.best_curve = min(curves, key=lambda curve: (
                            np.linalg.norm(curve[-1] - goal_center_3d) +  # distanza dalla goal
                            0.5 * np.linalg.norm(curve[0] - self.env.cur_pos)  # distanza dall'attuale posizione
                        ))
                    
                    print(f"Curve choosen: {self.best_curve}")

                    # Salva la curva migliore
                    self.current_target_curve = self.best_curve
                else:
                    self.current_target_curve = None
                    print("[WARNING] Nessuna curva Bézier trovata nel tile attuale.")

        # --- Fine logica per il calcolo della goal_region ---

        # --- Calcolo della migliore curva da seguire su un rettilineo ---
            if self.tile_counter == 1 and tile_kind == "straight":

                curves = curves = self.env._get_curve(self.current_tile[0], self.current_tile[1])

                if curves is not None and len(curves) > 0:

                    # Trova la curva che finisce più al punto di partenza (parte da rettilineo)
                    self.best_curve = min(curves, key=lambda curves: (
                            np.linalg.norm(curves[0] - self.env.cur_pos)  # distanza dall'attuale posizione
                            ))
                    
                    print(f"Curve choosen: {self.best_curve}")
                
                    self.current_target_curve = self.best_curve
                else:
                    self.current_target_curve = None
                    print("[WARNING] Nessuna curva Bézier trovata nel tile attuale.")

                
            if self.tile_counter > 1 and tile_kind == "straight":
                self.direzione = "straight"

                delta_tile_movement = tuple([self.current_tile[0] - self.prev_tile[0], self.current_tile[1] - self.prev_tile[1]])
                print (self.current_tile, self.prev_tile)

                tile_size = self.env.road_tile_size

                gx = self.current_tile[0]
                gy = self.current_tile[1]

                if delta_tile_movement == (1, 0): # movimento nord
                    self.goal_tile = tuple([self.current_tile[0] + 1, self.current_tile[1]])
                    self.goal_center = np.array([gx + 1.0, gy + 0.7]) * tile_size
                if delta_tile_movement == (-1, 0): # movimento sud
                    self.goal_tile = tuple([self.current_tile[0] - 1 , self.current_tile[1]])
                    self.goal_center = np.array([gx, gy + 0.3]) * tile_size
                if delta_tile_movement == (0, -1): # movimento ovest
                    self.goal_tile = tuple([self.current_tile[0], self.current_tile[1] - 1])
                    self.goal_center = np.array([gx + 0.7, gy]) * tile_size
                if delta_tile_movement == (0, 1): # movimento est
                    self.goal_tile = tuple([self.current_tile[0], self.current_tile[1] + 1])
                    self.goal_center = np.array([gx + 0.3, gy + 1.0]) * tile_size

                print(f"Goal tile calcolata: {self.goal_tile}, Goal point: {self.goal_center}")

                curves = self.env._get_curve(self.current_tile[0], self.current_tile[1])

                print (f"available curves: {curves}")

                if curves is not None and len(curves) > 0:
                    # Estendi goal_center a 3D per confronto (Y=0)
                    goal_center_3d = np.array([self.goal_center[0], 0, self.goal_center[1]])

                    # Trova la curva che finisce più vicina al centro della goal_tile
                    self.best_curve = min(curves, key=lambda curves: (
                                np.linalg.norm(curves[-1] - goal_center_3d) +  # distanza dalla goal
                                0.5 * np.linalg.norm(curves[0] - self.env.cur_pos)  # distanza dall'attuale posizione
                            ))
                    
                    print(f"Curve choosen: {self.best_curve}")

                        
                    self.current_target_curve = self.best_curve
                else:
                    self.current_target_curve = None
                    print("[WARNING] Nessuna curva Bézier trovata nel tile attuale.")

        if self.best_curve is None:
            curves = self.env._get_curve(self.current_tile[0], self.current_tile[1])

            if curves is not None and len(curves) > 0:

                # Trova la curva che finisce più al punto di partenza (parte da rettilineo)
                self.best_curve = min(curves, key=lambda curves: (
                        np.linalg.norm(curves[0] - self.env.cur_pos)  # distanza dall'attuale posizione
                        ))
                
                self.current_target_curve = self.best_curve

        # 7️⃣ Distanza normalizzata dalla goal_center (usata solo se esiste un goal)
        tile_size = self.env.road_tile_size
        dist_to_goal = 0.0

        try:
            # Calcolo dei punti della curva Bézier
            curve_points = self.bezier_curve(self.current_target_curve, n_points=100)  # (100, 3)

            # Prendi solo le coordinate X,Y (ignora Z)
            curve_xy = curve_points[:, [0, 2]]  # shape (100, 2)

            # Posizione attuale dell'agente (X,Z)
            agent_pos = np.array([self.env.cur_pos[0], self.env.cur_pos[2]])

            # Calcola le distanze da tutti i punti della curva
            dists = np.linalg.norm(curve_xy - agent_pos, axis=1)

            # Trova la distanza minima
            min_dist = np.min(dists)

            # Normalizza rispetto alla metà della corsia
            self.normalized_dist = min_dist / (self.lane_width / 2.0)

            # Penalità proporzionale alla distanza
            # reward -= normalized_dist

            # print (f"distance from the curve: {self.normalized_dist}")

        except Exception as e:
            print(f"[ERROR] Errore nel calcolo della reward Bézier: {e}")
            # Eventualmente penalizza

        if self.goal_center is not None:
            pos2d = np.array(self.env.cur_pos)[[0, 2]]
            dist_to_goal = np.linalg.norm(pos2d - self.goal_center)
            dist_norm = dist_to_goal / (np.sqrt(2) * tile_size)  # valore ∈ [0, 1] in teoria
        else:
            dist_norm =  1.0  # corretto perchè ci sia sempre un goal da raggiungere meno all'inizio

        # 8️⃣ Composizione osservazione vettoriale
        obs = np.concatenate([
            np.array([lateral, angle, dist_norm, self.normalized_dist], dtype=np.float32),  # 4 valori continui
            self.tile_one_hot(tile_kind),                       # 4 valori one-hot
            self.action_one_hot(self.direzione)                 # 3 valori one-hot per direzione                         
        ])

        # 9️⃣ Calcolo reward e done_extra
        reward, done_extra, info_extra = self.compute_reward(action, obs)

        if self.num_steps % 100 == 0 and self.lane_pos is not None:
            print(f"[DEBUG] step={self.num_steps} | lateral={lateral:.2f}, angle={angle:.2f}, dot_dir={self.lane_pos.dot_dir:.2f}")
            print(f"tile_kind={tile_kind}, reward={reward:.2f}, distance from the curve choosen: {self.normalized_dist}")
            if self.goal_center is not None:
                print(f" Distance from the goal: {dist_norm}")

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

        if self.tile_counter > 5:
            reward += 10.0  # arrivato alla fine del percorso
            done = True

        # 10.4 Altre condizioni definite da compute_reward
        done = done or done_extra
        if info_extra:
            info.update(info_extra)

        self.good_steps += 1

        # Aggiorna prev_tile per il prossimo step
        self.prev_tile = self.current_tile
        self.last_obs = obs

        if flag is not None and flag == True:
            done = True

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
        lateral, angle, dist_norm, dist_curve = obs[:4]
        tile_one_hot = obs[4:8]
        action_one_hot = obs[8:]
        reward = 0.0

        # if self.current_target_curve is None or self.tile_counter < 2:
            # print("[WARNING] self.current_target_curve è None, salta reward Bézier.")
                    # 1. Penalità per offset laterale dalla corsia
            # if -0.105 <= lateral <= 0.105:
                # reward -= abs(lateral) / 0.105   # normalizzato [-1,0]
            # else:
                # sei fuori corsia → chiudi episodio
                # reward -= 5.0
                # done = True
                # info_extra = {"wrong lane": True}
            # reward -= abs(angle)/self.max_angle
            # if  -0.02 <= angle <= 0.02:
                # reward += 1.0
            # elif angle <= -1 or angle >= 1:
                # reward -= 5.0
                # done = True
                # info_extra = {"wrong angle": True} 
            # return reward  # ritorna il reward calcolato finora

        if self.goal_center is not None:
            if (self.prev_dist_to_goal - dist_norm) <=0:
                reward -= 3.0 # the agent stays still or it is moving away from the goal

            else:
                reward += self.prev_dist_to_goal - dist_norm

            if dist_norm < 0.15:
                reward += 1.5
                info_extra = {"near the goal": True}
        
        self.prev_dist_to_goal = dist_norm

        if (self.prev_dist_curve - dist_curve) <= 0:
            reward = -3.0 # the agent stays still or it is moving away from the correct curve
        else:
            reward += dist_curve - self.prev_dist_curve

        self.prev_dist_curve = dist_curve

        reward += dist_curve - self.prev_dist_curve
        self.prev_dist_curve = dist_curve

        if lateral < -0.105 or lateral > 0.105:
            reward -= 5.0
            # done = True
            # info_extra = {"out_of_lane": True}
            # return reward

        if angle < -1.0 or angle > 1.0:
            reward -= 5.0
            # done = True
            # info_extra = {"wrong_angle": True}
            # return reward
        
        return reward

    def reward_curve(self, action, obs):

        lateral, angle, dist_norm, dist_curve = obs[:4]
        tile_one_hot = obs[4:8]
        action_one_hot = obs[8:]
        reward = 0.0

        tile_kind_idx = int(np.argmax(tile_one_hot))
        tile_kind = self.tile_kinds[tile_kind_idx]

        action_one_hot_idx = int(np.argmax(action_one_hot))
        action = self.action_types[action_one_hot_idx]

        reward = 0.0
        # reward -= abs(lateral)/ self.max_offset

        if (self.prev_dist_to_goal - dist_norm) <=0:
            reward -= 3.0 # the agent stays still or it is moving away from the goal

        else:
            reward += self.prev_dist_to_goal - dist_norm
        
        self.prev_dist_to_goal = dist_norm

        if (self.prev_dist_curve - dist_curve) <= 0:
            reward = -3.0 # the agent stays still or it is moving away from the correct curve
        else:
            reward += dist_curve - self.prev_dist_curve

        self.prev_dist_curve = dist_curve

            # Eventualmente penalizza
            # reward -= 1.0

        if dist_norm < 0.15:
            reward += 1.5
            info_extra = {"near the goal": True}

        return reward
    
    def reward_intersection(self, action, obs):
        lateral, angle, dist_norm, dist_curve = obs[:4]
        tile_one_hot = obs[4:8]
        action_one_hot = obs[8:]
        reward = 0.0

        if (self.prev_dist_to_goal - dist_norm) <=0:
            reward -= 3.0 # the agent stays still or it is moving away from the goal

        else:
            reward += self.prev_dist_to_goal - dist_norm
        
        self.prev_dist_to_goal = dist_norm

        if (self.prev_dist_curve - dist_curve) <= 0:
            reward = -3.0 # the agent stays still or it is moving away from the correct curve
        else:
            reward += dist_curve - self.prev_dist_curve

        self.prev_dist_curve = dist_curve

        # Controlla se la curva è valida
        # if self.current_target_curve is None:
            # print("[WARNING] self.current_target_curve è None, salta reward Bézier.")
            # return reward  # ritorna il reward calcolato finora

        # try:
            # Calcolo dei punti della curva Bézier
            # curve_points = self.bezier_curve(self.current_target_curve, n_points=100)  # (100, 3)

            # Prendi solo le coordinate X,Y (ignora Z)
            # curve_xy = curve_points[:, [0, 2]]  # shape (100, 2)

            # Posizione attuale dell'agente (X,Z)
            # agent_pos = np.array([self.env.cur_pos[0], self.env.cur_pos[2]])

            # Calcola le distanze da tutti i punti della curva
            # dists = np.linalg.norm(curve_xy - agent_pos, axis=1)

            # Trova la distanza minima
            # min_dist = np.min(dists)

            # Normalizza rispetto alla metà della corsia
            # normalized_dist = min_dist / (self.lane_width / 2.0)

            # Penalità proporzionale alla distanza
            # reward -= normalized_dist

            #print (f"distance from the curve: {normalized_dist}")

        # except Exception as e:
            # print(f"[ERROR] Errore nel calcolo della reward Bézier: {e}")
            # Eventualmente penalizza
        
        if dist_norm < 0.15:
            reward += 1.5
            info_extra = {"near the goal": True}
        return reward

    def compute_reward(self, action, obs):

        lateral, angle, dist_norm, dist_curve = obs[:4]
        tile_one_hot = obs[4:8]
        action_one_hot = obs[8:]
        reward = 0.0
        done = False
        info_extra = {}
        reward = 0.0

        start_tile = self.env.unwrapped.get_grid_coords(self.env.cur_pos)
        i, j = start_tile

        tile_kind_idx = int(np.argmax(tile_one_hot))
        tile_kind = self.tile_kinds[tile_kind_idx]
        agent_abs_angle = self.env.unwrapped.cur_angle

        pos = self.env.unwrapped.cur_pos
        curr_i, curr_j = self.env.unwrapped.get_grid_coords(pos)
        self.current_tile = (curr_i, curr_j)

        tile_info = self.env.unwrapped._get_tile(curr_i, curr_j)
 
        if self.lane_pos is None or not tile_info['drivable']:
            reward = -10.0
            done = True
            info_extra["out_of_lane"] = True
            return reward, done, info_extra

        if not hasattr(self, "prev_dist_to_goal"):
            self.prev_dist_to_goal = dist_norm

        if tile_kind == "straight":
            # reward += self.reward_straight(action, lateral, angle, agent_abs_angle, start_tile)
            reward = self.reward_straight(action, obs)
        elif tile_kind in ["curve_left", "curve_right"]:
            reward += self.reward_curve(action, obs)
            # print(obs)
            self.prev_dist_to_goal = dist_norm
        elif tile_kind in ["3way_left", "3way_right", "4way"]:
            reward += self.reward_intersection(action, obs)
            self.prev_dist_to_goal = dist_norm

        # Raggiunto obiettivo
        if start_tile == self.goal_tile:
            reward += 5.0
            # done = True
            done = False # solo per test
            info_extra["goal_reached"] = True

        return reward, done, info_extra

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
            user_tile_start= (1,2),
  
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
        learning_rate=1e-4,
        buffer_size=100_000,               # AUMENTATO
        learning_starts=10000,              # Parte dopo aver riempito un po' di buffer
        batch_size=64, 
        tau=0.005,                 
        gamma=0.99,
        train_freq=4,
        gradient_steps=4,
        target_update_interval=1000,
        exploration_fraction=0.3,
        exploration_final_eps=0.01,
        verbose=1,
        # tensorboard_log="./dqn_duckie_tensorboard/"
    )

    # 1 milione di step di training

    # Avvia il training vero
    # Training in modalità batch
    num_batches = 10
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
