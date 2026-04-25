"""Ejercicio 4 — DQN en almacén (MLII MUBD).

Usa stable-baselines3 como librería de RL. El alumno sólo debe
completar ``observacion.py``; este script se encarga de crear el
entorno, envolverlo con los wrappers de observación y recompensa, y
lanzar el entrenamiento/evaluación con DQN de stable-baselines3.

Uso::

    python main.py --entorno 1                    # sólo recogida
    python main.py --entorno 2 --timesteps 50000
    python main.py --entorno 3 --render

Requisitos::

    pip install numpy gymnasium matplotlib stable-baselines3
"""
from __future__ import annotations

import argparse

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy

from env import WarehouseEnv
from observacion import ObservationBuilder

ENV_CONFIG = {
    1: {"random_objects": False, "drop": False},
    2: {"random_objects": False, "drop": True},
    3: {"random_objects": True, "drop": True},
}


# ---------------------------------------------------------------- wrappers
class FullWrapper(gym.Wrapper):
    """Envuelve ``WarehouseEnv`` aplicando las transformaciones del alumno:

    1. La **observación** pasa por ``ObservationBuilder.build()``.
    2. La **recompensa** se reemplaza por ``ObservationBuilder.calculate_reward()``.

    De este modo stable-baselines3 recibe directamente las features y la
    recompensa diseñadas por el alumno, sin que éste tenga que tocar
    este fichero.
    """

    def __init__(self, env: WarehouseEnv, obs_builder: ObservationBuilder):
        super().__init__(env)
        self.obs_builder = obs_builder
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_builder.obs_dim,),
            dtype=np.float32,
        )
        self._prev_raw_obs: np.ndarray | None = None

    def reset(self, **kwargs):
        raw_obs, info = self.env.reset(**kwargs)
        self._prev_raw_obs = raw_obs.copy()
        return self.obs_builder.build(raw_obs), info

    def step(self, action):
        raw_obs, _env_reward, terminated, truncated, info = self.env.step(action)
        reward = self.obs_builder.calculate_reward(
            self._prev_raw_obs, raw_obs, terminated, truncated,
        )
        self._prev_raw_obs = raw_obs.copy()
        return self.obs_builder.build(raw_obs), reward, terminated, truncated, info


# ---------------------------------------------------------------- cli
def build_parser():
    p = argparse.ArgumentParser(
        description="Ejercicio 4 — DQN en almacén (MLII MUBD)",
    )
    p.add_argument("--entorno", type=int, choices=[1, 2, 3], default=1)
    p.add_argument("--timesteps", type=int, default=20_000,
                    help="Pasos totales de entrenamiento.")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--render", action="store_true")
    p.add_argument("--eval-episodes", type=int, default=50,
                    help="Episodios de evaluación tras el entrenamiento.")
    return p


def main():
    args = build_parser().parse_args()
    cfg = ENV_CONFIG[args.entorno]
    obs_builder = ObservationBuilder()

    # --- Entorno de entrenamiento (sin render) ---
    train_env = FullWrapper(WarehouseEnv(**cfg), obs_builder)

    # --- Agente DQN (stable-baselines3) ---
    model = DQN(
        "MlpPolicy",
        train_env,
        learning_rate=args.lr,
        gamma=args.gamma,
        batch_size=64,
        buffer_size=50_000,
        learning_starts=500,
        target_update_interval=500,
        exploration_fraction=0.3,
        exploration_final_eps=0.05,
        verbose=1,
        seed=args.seed,
    )

    print(
        f"Entorno {args.entorno} | DQN (stable-baselines3) "
        f"| {args.timesteps} timesteps | obs_dim={obs_builder.obs_dim}"
    )
    model.learn(total_timesteps=args.timesteps)

    # --- Evaluación ---
    eval_env = FullWrapper(WarehouseEnv(**cfg), obs_builder)
    mean_reward, std_reward = evaluate_policy(
        model, eval_env, n_eval_episodes=args.eval_episodes,
    )
    print(
        f"\nEvaluación ({args.eval_episodes} episodios): "
        f"retorno medio = {mean_reward:.2f} +/- {std_reward:.2f}"
    )

    # --- Visualización opcional ---
    if args.render:
        render_env = FullWrapper(
            WarehouseEnv(render_mode="human", **cfg), obs_builder,
        )
        obs, _ = render_env.reset()
        for _ in range(2000):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = render_env.step(action)
            render_env.render()
            if terminated or truncated:
                obs, _ = render_env.reset()
        render_env.close()

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
