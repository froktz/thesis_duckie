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


# Funzione per media mobile
def moving_average(x, w):
    x = np.array(x).flatten()
    return np.convolve(x, np.ones(w), 'valid') / w

def left_turn_available(env):
    # Coordinate attuali del robot
    x, y, z = env.cur_pos
    i = int(np.floor(x))
    j = int(np.floor(z))

    tile = env.unwrapped._get_tile(i, j)
    if tile is None:
        return False

    tile_kind = tile['kind']
    # Lista dei tipi di incrocio che ammettono svolta a sinistra
    return tile_kind in ['4way', '3way_left', 'curve_left']

class DuckieRLWrapper(gym.Env):

    def __init__(self, env):
        super(DuckieRLWrapper, self).__init__()
        self.env = env

        # Lista delle posizioni iniziali possibili (posizione, angolo)
        self.possible_starts = [
            (np.array([1.85313356, 0., 1.5687472]), -0.06229765469565894),
            (np.array([1.35389949, 0., 1.89896477]), -1.6180560830233601),
            (np.array([0.69820322, 0., 1.5677339]), -0.028414341288794793),
            (np.array([0.2134203, 0., 0.60191441]), -1.5570214445017387),
            (np.array([0.72964605, 0., 1.56071875]), -0.21878014687120712),
            (np.array([1.9739118, 0., 1.60418237]), 0.018816033774968922),
            (np.array([1.61513161, 0., 2.01799389]), 1.5476869500965005),
            (np.array([1.64426201, 0., 0.75767361]), 1.5476869500965005),
            (np.array([1.58126874, 0., 2.19177157]), 1.5289606578796389),
            (np.array([1.84308267, 0., 1.51558419]), 0.04482048222269934),
            (np.array([2.14962907, 0., 0.20062444]), 3.088490894742158)
            ]
        
        self.oscillation_count = 0
        self.last_action = None

        # Per il tracking del movimento verso la goal
        self.last_pos = None

        # Per gestire il centro della goal_region
        self.goal_points = None
        self.goal_region = lambda pos, angle: False  # placeholder

        # modificato dimensione da (480, 680, 3) per ridurre il carico computazionale

        # hybrid observation: image  + vector features
        self.observation_space = spaces.Box(
            low=np.array([-1., -np.pi, 0.]),
            high=np.array([1., np.pi, 1.]),
            dtype=np.float32
            )
        self.action_space = spaces.Discrete(7)  # Azioni: sinistra, dritto, destra (per semplificare)

        # possibile implementazione in spazio continuo
        self.good_steps = 0 
        self.goal_tile = None
        self.last_action = None
        self.oscillation_count = 0

    def seed(self, seed=None):
        return self.env.seed(seed)

    def generate_goal_region_from_start(self, start_pos, start_angle, dist=0.8, width=0.2, height=0.2):
        import numpy as np
        # x, _, z = start_pos
        x, _, z = start_pos[:3]  # Assicurati di usare solo x e z
        forward_dir = np.array([np.cos(start_angle), np.sin(start_angle)])
        goal_center = np.array([x, z]) + dist * forward_dir

        dx = width / 2
        dz = height / 2
        p1 = goal_center + [-dx, -dz]
        p2 = goal_center + [dx, -dz]
        p3 = goal_center + [dx, dz]
        p4 = goal_center + [-dx, dz]

        return [np.array([p[0], 0, p[1]]) for p in [p1, p2, p3, p4]]

    def create_goal_region_function(self, goal_points):
        from shapely.geometry import Point, Polygon

        poly_2d = Polygon([(p[0], p[2]) for p in goal_points])

        def is_in_goal(pos, angle):
            pt = Point(pos[0], pos[2])
            return poly_2d.contains(pt)

        return is_in_goal

    def _get_tile_from_pos(self, pos):
        """
        Calcola la tile (x, y) a partire dalla posizione globale.
        """
        tile_size = self.env.road_tile_size
        return tuple(np.floor(np.array(pos)[:2] / tile_size).astype(int))

    def stop_line_detected(self, img):
        # img: la stessa immagine ridimensionata da passare all'agente

        # Converti in scala di grigi
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # Equalizzazione per condizioni di luce variabili
        gray = cv2.equalizeHist(gray)

        # Canny edge detection
        edges = cv2.Canny(gray, 100, 200)
        
        # cerca linee orizzontali via HoughLinesP
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=50, maxLineGap=10)
        if lines is None: 
            return False
        h, w = img.shape[:2]
        
        for x1, y1, x2, y2 in lines[:, 0]:
            # Cerca linee orizzontali abbastanza lunghe nella parte bassa dell'immagine
            if abs(y1 - y2) < 8 and abs(x2 - x1) > w * 0.5 and min(y1, y2) > h * 0.6:
                return True
        return False
    
    # def get_goal_tile(self, position, angle):
        """
        Calcola la goal_tile finale dopo una svolta a sinistra, basandosi solo su feature vettoriali.
        """
        # tile_coords = self._get_tile_from_pos(position)

        # Normalizza l'angolo tra 0 e 2π
        # angle_norm = angle % (2 * np.pi)

        # Direzione attuale: 0=est, 1=nord, 2=ovest, 3=sud
        #direction = int(np.round(angle_norm / (0.5 * np.pi))) % 4

        # Dopo svolta a sinistra: nuova direzione
        # left_turn = {0: 1, 1: 2, 2: 3, 3: 0}
        # new_direction = left_turn[direction]

        # Delta movimento in base alla nuova direzione
        # deltas = {0: (1, 0), 1: (0, -1), 2: (-1, 0), 3: (0, 1)}
        # dx, dy = deltas[new_direction]

        # Tile di destinazione
        # goal_tile = (tile_coords[0] + dx, tile_coords[1] + dy)

        # return goal_tile

    def get_tile_ahead(env, distance=0.5):
        """Restituisce le coordinate e il tipo della tile davanti al robot, a distanza specificata"""
        pos = env.cur_pos
        angle = env.cur_angle

        # Versore direzione del robot
        direction = np.array([np.cos(angle), 0, np.sin(angle)])
    
        # Punto davanti
        lookahead_pos = pos + direction * distance
        i, j = int(np.floor(lookahead_pos[0])), int(np.floor(lookahead_pos[2]))

        try:
            tile = env.unwrapped._map_tile_dict[(i, j)]
            return (i, j), tile
        except KeyError:
            return None, None
        
    def stop_line_ahead(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        _, th = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        h, w = th.shape
        crop = th[h//2:h//2 + h//4, :]  # striscia centrale
        cnts, _ = cv2.findContours(crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in cnts:
            x, y, cw, ch = cv2.boundingRect(cnt)
            if cw > w * 0.6 and ch > h * 0.05:
                return True
        return False

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.env.seed(seed)
            np.random.seed(seed)
            random.seed(seed)
        self.good_steps = 0
        img = self.env.reset()[0]

        self.last_pos = self.env.cur_pos.copy() # Inizializza qui

        # start_pos, start_angle = random.choice(self.possible_starts)
        # self.env.cur_pos = start_pos.copy()
        # self.env.cur_angle = start_angle
        # self.goal_region = self.generate_goal_region_from_start(start_pos, start_angle)

        start_pos, start_angle = random.choice(self.possible_starts)
        self.goal_region = self.generate_goal_region_from_start(start_pos, start_angle)

        self.env.cur_pos = start_pos.copy()
        self.env.cur_angle = start_angle


        print(f"Posizione iniziale: {self.env.cur_pos}, Angolo: {self.env.cur_angle}")

        self.goal_tile = None
        self.stop_tile = None
        self.stop_detected = False

        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        obs_img = cv2.resize(img, (400, 300))

        # Rileva la linea di stop (opzionale, puoi rimuovere se non serve)
        cur_pos = self.env.cur_pos
        tile_size = self.env.road_tile_size
        cur_tile = tuple(np.floor(cur_pos[:2] / tile_size).astype(int))

        if not self.stop_detected and self.stop_line_detected(obs_img):
            self.stop_tile = cur_tile
            self.stop_detected = True
            print(f"Linea di stop rilevata in tile: {self.stop_tile}")

        # Gestione robusta di lane_position
        try:
            lane_pos = self.env.get_lane_pos2(self.env.cur_pos, self.env.cur_angle)
            lateral = lane_pos.dist
            angle = lane_pos.angle_deg / 180.0  # normalizza l'angolo in [-1, 1]
        except Exception as e:
            print(f"[WARN] Could not compute lane pos: {e}")
            lateral = 0.0
            angle = 0.0
        stop_flag = 0.0
        obs = np.array([lateral, angle, stop_flag], dtype=np.float32)

        # Imposta la goal_region e la lista dei suoi punti
        self.goal_points = self.generate_goal_region_from_start(start_pos, start_angle)
        self.goal_region = self.create_goal_region_function(self.goal_points)

        return obs, {}  

    #def _tile_to_pos(self, tile):
        # Esempio: supponiamo che ogni tile sia larga 1 unità
        # e che la posizione sia il centro della tile
        #x, y = tile
        #return [x + 0.5, 0, y + 0.5]

    def step(self, action):
        # self.action_space = spaces.Discrete(7)

        action_map = {
            0: (0.0, 0.3),    # svolta molto forte a sinistra
            1: (0.1, 0.3),    # svolta forte a sinistra
            2: (0.2, 0.3),    # svolta leggera a sinistra
            3: (0.3, 0.3),    # dritto
            4: (0.3, 0.2),    # svolta leggera a destra
            5: (0.3, 0.1),    # svolta forte a destra
            6: (0.3, 0.0),    # svolta molto forte a destra
        }   

        vl, vr = action_map[int(action)]

        raw_obs, _, done, info = self.env.step(np.array([vl, vr]))

        # ⚠️ Verifica se l'agente ha scelto una svolta a sinistra
        if int(action) in [0, 1, 2]:
            if not left_turn_available(self.env):
                print("Svolta a sinistra non disponibile! Episodio terminato.")
                reward = -2.0  # penalità opzionale
                done = True

        img = self.env.render('rgb_array')
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        obs_img = cv2.resize(img, (400, 300))
        self.good_steps += 1

        # Feature extraction
        try:
            lane_pos = self.env.get_lane_pos2(self.env.cur_pos, self.env.cur_angle)
            lateral = lane_pos.dist
            angle = lane_pos.angle_deg / 180.0  # normalizza l'angolo in [-1, 1]
            lane_position = {'lateral': lateral, 'angle': angle}
        except Exception as e:
            lane_position = None
            lateral = 0.0
            angle = 0.0
        stop_flag = 1.0 if self.stop_line_ahead(obs_img) else 0.0
        obs = np.array([lateral, angle, stop_flag], dtype=np.float32)

        # Calcola la ricompensa usando funzione esterna
        reward, done_extra, info_extra = self.compute_reward(action, lane_position, obs, img)
        done = done or done_extra
        if info_extra is not None:
            info.update(info_extra)

        return obs, reward, done, False, info

    from PIL import Image, ImageDraw

    def _highlight_goal_tile(self, obs_image):
        """
        Disegna un rettangolo rosso sulla goal_tile nella visualizzazione dell'ambiente.
        """
        if not hasattr(self, "goal_tile") or self.goal_tile is None:
            return obs_image  # niente da disegnare

        img = Image.fromarray(obs_image)
        draw = ImageDraw.Draw(img)

        # Coordinate tile (es. [2, 4])
        tile_x, tile_y = self.goal_tile

        # Calcola posizione in pixel
        map_width, map_height = self.env.map_shape
        cam_w, cam_h = self.env.camera_width, self.env.camera_height
        tile_size_px_x = cam_w / map_width
        tile_size_px_y = cam_h / map_height

        px = int((tile_x + 0.5) * tile_size_px_x)
        py = int((tile_y + 0.5) * tile_size_px_y)

        box_size = 10
        draw.rectangle([px - box_size, py - box_size, px + box_size, py + box_size], outline="red", width=3)

        return np.array(img)

    def render(self, mode='human'):
        # obs = self.env.render(mode='rgb_array')

        if mode == 'human':
            # Usa la finestra interattiva di Duckietown
            return self.env.render(mode='human')
        else:
            obs = self.env.render(mode='rgb_array')
            obs = self._highlight_goal_tile(obs)
            return obs

    # def render(self, mode='human'):
        # return self.env.render(mode)

    def compute_reward(self, action, lane_position, obs, img):
        reward = 0.0
        done = False
        info_extra = {}

        if lane_position is None:
            reward -= 20.0  # Uscito completamente dalla corsia
            done = True
            info_extra["out_of_lane"] = True
            return reward, done, info_extra

        # -------------------------------
        # Distanza laterale dalla corsia
        # -------------------------------
        lateral_offset = lane_position.get('lateral', 0.0)
        reward -= abs(lateral_offset) * 2.0  # Penalità crescente

        if abs(lateral_offset) > 0.3:
            reward -= 5.0  # Penalità extra se molto fuori

        # -------------------------------
        # Penalità angolare
        # -------------------------------
        if self.env.step_count > 10:
            angle = lane_position.get('angle', 0.0)
            ref_heading = lane_position.get('ref_heading', 0.0)
            angle_diff = np.abs((angle - ref_heading + np.pi) % (2 * np.pi) - np.pi)
            reward -= angle_diff * 0.5
        else:
            angle_diff = 0.0

        # -------------------------------
        # Penalità per oscillazioni
        # -------------------------------
        if self.last_action is not None and ((self.last_action == 0 and action == 2) or (self.last_action == 2 and action == 0)):
            self.oscillation_count += 1
        else:
            self.oscillation_count = 0
        self.last_action = action
        if self.oscillation_count >= 3:
            reward -= 1.0

        # -------------------------------
        # Penalità se nella corsia sbagliata
        # -------------------------------
        wrong_side = False
        try:
            start = np.array(lane_position.lane_segment['start_node'])
            end = np.array(lane_position.lane_segment['end_node'])
            direction = end - start

            if tuple(direction) in [(0, 1), (1, 0)]:  # Nord o Est → corsia destra = lateral > 0
                if lateral_offset < 0:
                    reward -= 2.0
                    wrong_side = True
            elif tuple(direction) in [(0, -1), (-1, 0)]:  # Sud o Ovest → corsia destra = lateral < 0
                if lateral_offset > 0:
                    reward -= 2.0
                    wrong_side = True
        except Exception:
            pass

        info_extra["wrong_side"] = wrong_side

        # -------------------------------
        # Ricompensa per stare bene in corsia
        # -------------------------------
        if abs(lateral_offset) < 0.05 and angle_diff < 0.1 and not wrong_side:
            reward += 1.0  # piccolo bonus per buon comportamento

        # Penalizza se ruota troppo quando è già ben centrato
        if abs(lateral_offset) < 0.05 and abs(angle_diff) > 0.2:
            reward -= 1.0  # puoi regolare l'intensità a piacere

        # Premia se è centrato e sceglie di andare dritto (solo se azioni discrete)
        if abs(lateral_offset) < 0.05 and abs(angle_diff) < 0.1 and action == 3:
            reward += 0.5  # puoi aumentare leggermente se non si nota miglioramento

        # -------------------------------
        # Goal raggiunta
        # -------------------------------
        if self.goal_region(self.env.cur_pos, self.env.cur_angle):
            reward += 100.0
            done = True
            info_extra["goal_reached"] = True

        # -------------------------------
        # Ricompensa per avvicinamento alla goal
        # -------------------------------
        if hasattr(self, "last_pos"):
            if hasattr(self, "goal_points"):
                goal_center = np.mean(self.goal_points, axis=0)[[0, 2]]
            else:
                forward_dir = np.array([np.cos(self.env.cur_angle), np.sin(self.env.cur_angle)])
                goal_center = np.array(self.env.cur_pos)[[0, 2]] + 0.8 * forward_dir

            old_dist = np.linalg.norm(np.array(self.last_pos)[[0, 2]] - goal_center)
            new_dist = np.linalg.norm(np.array(self.env.cur_pos)[[0, 2]] - goal_center)
            reward += (old_dist - new_dist) * 1.5
        self.last_pos = self.env.cur_pos

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
            max_steps=500,
            camera_rand=False,
            dynamics_rand=False,
            randomize_maps_on_reset=False,
            user_tile_start=(1, 2),
            # user_angle_start=[-np.pi / 2]
        )

    else:
        env = gym.make(args.env_name)

    env = DuckieRLWrapper(env)  # Wrappare l'ambiente per RL
    env = DummyVecEnv([lambda: env])  # Necessario per stable-baselines3

    raw_env = env.envs[0].env

    # raw_env.render(mode='human')  # Crea la finestra se non esiste ancora

    # @raw_env.unwrapped.window.event
    # def on_key_press(symbol, modifiers):
        #if symbol in (key.BACKSPACE, key.SLASH):
            #raw_env.reset() 
            #raw_env.render()
        #elif symbol == key.ESCAPE:
            #raw_env.close()
            #sys.exit(0)

    # Addestrare l'agente RL
    # Crea e addestra il modello PPO

    start_time = time.time()

    model = DQN(
        policy="MlpPolicy",
        env=env,
        learning_rate=1e-4,
        buffer_size=100_000,               # AUMENTATO
        learning_starts=5000,              # Parte dopo aver riempito un po' di buffer
        batch_size=64,
        tau=1.0,                           # Soft update, può restare così
        gamma=0.99,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=1000,
        exploration_fraction=0.1,
        exploration_final_eps=0.01,
        verbose=1,
        # tensorboard_log="./dqn_duckie_tensorboard/"
    )

    # 1 milione di step di training

    # Avvia il training vero
    # Training in modalità batch
    num_batches = 3
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
        plt.title('Performance dell’agente PPO nel test')
        plt.grid()
        plt.show()
    else:
        print("Nessuna ricompensa registrata nel test.")# Salva il modello finale 
    
    env.close()

    try:
        model.save("ppo_duckietown_model")
        print("Modello salvato correttamente come ppo_duckietown_model")
    except Exception as e:
        print(f"Errore durante il salvataggio del modello: {e}")
   
if __name__=='__main__':
    main()
