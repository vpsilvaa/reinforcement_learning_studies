import eventlet
import math
import numpy as np
import threading
from flask import Flask, render_template
from flask_socketio import SocketIO
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

eventlet.monkey_patch()
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')

# Classe do ambiente
class CityEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=1, shape=(7,), dtype=np.float32)
        
        self.obstacles = [
            {"x": 42, "y": 340, "w": 90, "h": 55},
            {"x": 36, "y": 147, "w": 98, "h": 96},
            {"x": 321, "y": 184, "w": 75, "h": 58},
            {"x": 575, "y": 147, "w": 38, "h": 29},
            {"x": 702, "y": 151, "w": 48, "h": 25},
            {"x": 838, "y": 356, "w": 32, "h": 23},
            {"x": 842, "y": 473, "w": 31, "h": 20},
            {"x": 605, "y": 354, "w": 54, "h": 39},
            {"x": 397, "y": 495, "w": 135, "h": 39},
        ]

        self.checkpoints = [
            {"x": 35, "y": 489},
            {"x": 91, "y": 494},
            {"x": 164, "y": 496},
            {"x": 233, "y": 501},
            {"x": 230, "y": 423},
            {"x": 230, "y": 338},
            {"x": 227, "y": 247},
            {"x": 230, "y": 163},
            {"x": 229, "y": 85},
            {"x": 318, "y": 84},
            {"x": 418, "y": 87},
            {"x": 488, "y": 95},
            {"x": 488, "y": 182},
            {"x": 496, "y": 260},
            {"x": 565, "y": 265},
            {"x": 643, "y": 262},
            {"x": 723, "y": 268},
            {"x": 754, "y": 326},
            {"x": 757, "y": 390},
            {"x": 760, "y": 463},
            {"x": 755, "y": 534},
            {"x": 768, "y": 575},
        ]
        self.reset()

    def reset(self, seed=None, options=None):
        """
        Redefine o estado do ambiente para uma configuração inicial.
        
        Args:
            seed (int, optional): Semente aleatória para inicialização.
            options (dict, optional): Opções adicionais para o reset.
            
        Returns:
            tuple: Uma tupla contendo (observação_inicial, info_adicional).
        """
        super().reset(seed=seed)
        
        # pega as coordenadas do primeiro checkpoint (índice 0)
        start_point = self.checkpoints[0]
        
        # define a posição do carro baseada no checkpoint
        self.car = {"x": 37.0, "y": 480.0, "angle": 0.0, "speed": 0.0}
        
        # próximo alvo deve ser o 1
        self.current_ckpt = 1 
        
        self.steps = 0
        return self._get_obs(), {}

    def _get_obs(self):
        """
        Calcula e retorna a observação atual do ambiente para o agente.
        
        A observação consiste em 5 sensores de distância (lidar), a velocidade 
        atual normalizada e o ângulo relativo ao próximo checkpoint.
        
        Returns:
            np.ndarray: Array contendo os 7 valores da observação.
        """
        sensors = [self._cast_ray(self.car["angle"] + a) / 500.0 for a in [-60, -30, 0, 30, 60]]
        idx = min(self.current_ckpt, len(self.checkpoints) - 1)
        target = self.checkpoints[idx]
        dx, dy = target["x"] - self.car["x"], target["y"] - self.car["y"]
        angle_to_target = (math.degrees(math.atan2(-dy, dx)) - self.car["angle"]) / 180.0
        return np.array(sensors + [self.car["speed"]/5.0, angle_to_target], dtype=np.float32)

    def _cast_ray(self, angle):
        """
        Simula o disparo de um feixe (raio) para detectar obstáculos em uma direção.
        
        Args:
            angle (float): O ângulo absoluto em graus para onde o raio será disparado.
            
        Returns:
            float: A distância (0 a 500) até o primeiro obstáculo ou borda do mapa.
        """
        rad = math.radians(angle)
        for dist in range(0, 500, 15):
            rx, ry = self.car["x"] + math.cos(rad) * dist, self.car["y"] - math.sin(rad) * dist
            if rx < 0 or rx > 1000 or ry < 0 or ry > 600: return dist
            for o in self.obstacles:
                if o["x"] < rx < o["x"]+o["w"] and o["y"] < ry < o["y"]+o["h"]: return dist
        return 500.0

    def step(self, action):
        """
        Executa uma ação no ambiente, atualiza a física e retorna o novo estado.
        
        Args:
            action (np.ndarray): Array com [aceleração, direção] no intervalo [-1, 1].
            
        Returns:
            tuple: (observação, recompensa, finalizado, truncado, info).
        """
        self.steps += 1
        throttle, steer = float(action[0]), float(action[1])
        self.car["speed"] = float(np.clip(self.car["speed"] + throttle * 0.5, 0, 4))
        self.car["angle"] += steer * 10.0
        
        target = self.checkpoints[min(self.current_ckpt, len(self.checkpoints)-1)]
        old_dist = math.dist([self.car["x"], self.car["y"]], [target["x"], target["y"]])
        self.car["x"] += math.cos(math.radians(self.car["angle"])) * self.car["speed"]
        self.car["y"] -= math.sin(math.radians(self.car["angle"])) * self.car["speed"]
        new_dist = math.dist([self.car["x"], self.car["y"]], [target["x"], target["y"]])

        reward = (old_dist - new_dist) * 5.0
        done = False
        if self.car["x"] < 0 or self.car["x"] > 1000 or self.car["y"] < 0 or self.car["y"] > 600:
            reward, done = -200, True
        for o in self.obstacles:
            if o["x"] < self.car["x"] < o["x"]+o["w"] and o["y"] < self.car["y"] < o["y"]+o["h"]:
                reward, done = -200, True

        if new_dist < 60:
            reward += 100
            self.current_ckpt += 1
            if self.current_ckpt >= len(self.checkpoints):
                reward += 500
                done = True
        
        if self.steps > 800: done = True
        return self._get_obs(), reward, done, False, {}

# callback para vizualizar o treinamento em real time
class VisualTrainingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(VisualTrainingCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        """
        Método executado pelo Stable Baselines após cada passo do treinamento.
        
        Extrai o estado atual do ambiente e o envia via WebSockets (SocketIO) 
        para a interface frontend.
        
        Returns:
            bool: Se retornar False, o treinamento é interrompido.
        """
        # pega o ambiente que está sendo treinado
        env_instance = self.training_env.envs[0].unwrapped
        
        # envia os dados para o navegador
        socketio.emit('update_car', {
            'x': float(env_instance.car["x"]),
            'y': float(env_instance.car["y"]),
            'angle': float(env_instance.car["angle"]),
            'speed': float(env_instance.car["speed"]),
            'obstacles': env_instance.obstacles,
            'checkpoints': env_instance.checkpoints,
            'goal': env_instance.checkpoints[min(env_instance.current_ckpt, len(env_instance.checkpoints)-1)],
            'sensors': [float(s) for s in env_instance._get_obs()[:5]]
        })
        
        # pausa curta para o navegador conseguir desenhar
        eventlet.sleep(0.02) 
        return True

@app.route('/')
def index():
    """
    Rota principal da aplicação Flask.
    
    Returns:
        str: O conteúdo HTML da página index.html.
    """
    return render_template('index.html')

def start_training():
    """
    Inicializa o ambiente CityEnv e o modelo PPO para iniciar o aprendizado.
    
    Esta função configura os hiperparâmetros do modelo e utiliza o callback 
    de visualização para monitorar o progresso em tempo real.
    """
    # cria o ambiente aqui dentro
    local_env = CityEnv() 
    model = PPO("MlpPolicy", local_env, verbose=1, learning_rate=0.0003)
    
    callback = VisualTrainingCallback()
    # callback vai cuidar de enviar os dados para o navegador durante o aprendizado
    model.learn(total_timesteps=200000, callback=callback)

if __name__ == '__main__':
    # inicia o processo de treinamento em paralelo com o servidor web
    eventlet.spawn(start_training)
    socketio.run(app, port=5000)