"""train_and_plot.py — Entrenamiento DQN con métricas y gráficas completas.

Uso:
    python train_and_plot.py --entorno 1 --timesteps 30000
    python train_and_plot.py --entorno 2 --timesteps 100000
    python train_and_plot.py --entorno 3 --timesteps 200000
"""
from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from gymnasium import spaces
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback

from env import WarehouseEnv
from observacion import ObservationBuilder

ENV_CONFIG = {
    1: {"random_objects": False, "drop": False},
    2: {"random_objects": False, "drop": True},
    3: {"random_objects": True,  "drop": True},
}

WINDOW = 20  # ventana de media móvil


# ══════════════════════════════════════════════════════════════════════════════
# Utilidad: obtener el WarehouseEnv base recorriendo wrappers
# ══════════════════════════════════════════════════════════════════════════════

def get_warehouse_env(env) -> WarehouseEnv:
    """Desenvuelve la cadena de wrappers hasta encontrar WarehouseEnv."""
    while not isinstance(env, WarehouseEnv):
        env = env.env
    return env


# ══════════════════════════════════════════════════════════════════════════════
# Wrapper
# ══════════════════════════════════════════════════════════════════════════════

class FullWrapper(gym.Wrapper):
    def __init__(self, env: WarehouseEnv, obs_builder: ObservationBuilder):
        super().__init__(env)
        self.obs_builder = obs_builder
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(obs_builder.obs_dim,), dtype=np.float32,
        )
        self._prev_raw_obs: np.ndarray | None = None

    def reset(self, **kwargs):
        raw_obs, info = self.env.reset(**kwargs)
        self._prev_raw_obs = raw_obs.copy()
        return self.obs_builder.build(raw_obs), info

    def step(self, action):
        raw_obs, _, terminated, truncated, info = self.env.step(action)
        reward = self.obs_builder.calculate_reward(
            self._prev_raw_obs, raw_obs, terminated, truncated,
        )
        # Guardar banderas del ultimo estado para callbacks con VecEnv,
        # que resetea automaticamente tras done.
        info = dict(info)
        info["terminal_delivery"] = bool(raw_obs[10])
        info["terminal_collision"] = bool(raw_obs[9])
        info["terminal_has_object"] = bool(raw_obs[8])
        self._prev_raw_obs = raw_obs.copy()
        return self.obs_builder.build(raw_obs), reward, terminated, truncated, info


# ══════════════════════════════════════════════════════════════════════════════
# Callback de métricas
# ══════════════════════════════════════════════════════════════════════════════

class MetricsCallback(BaseCallback):
    """Registra métricas episodio a episodio durante el entrenamiento."""

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self._ep_reward = 0.0
        self._ep_steps  = 0
        self.ep_rewards:    list[float] = []
        self.ep_lengths:    list[int]   = []
        self.ep_successes:  list[float] = []
        self.ep_collisions: list[float] = []

    def _on_step(self) -> bool:
        self._ep_reward += float(self.locals["rewards"][0])
        self._ep_steps  += 1

        if self.locals["dones"][0]:
            info = self.locals["infos"][0]
            delivery = bool(info.get("terminal_delivery", False))
            collision = bool(info.get("terminal_collision", False))
            has_object = bool(info.get("terminal_has_object", False))
            base_env = get_warehouse_env(self.training_env.envs[0])

            # Exito: entorno 1 = recogio objeto, entornos 2/3 = entrego.
            success = float(delivery or (has_object and not base_env.drop))

            self.ep_rewards.append(self._ep_reward)
            self.ep_lengths.append(self._ep_steps)
            self.ep_successes.append(success)
            self.ep_collisions.append(float(collision))

            self._ep_reward = 0.0
            self._ep_steps  = 0

        return True


# ══════════════════════════════════════════════════════════════════════════════
# Evaluación post-entrenamiento
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_with_metrics(
    model: DQN,
    env_cfg: dict,
    obs_builder: ObservationBuilder,
    n_episodes: int,
) -> dict:

    eval_env  = FullWrapper(WarehouseEnv(**env_cfg), obs_builder)
    rewards, lengths, successes, collisions = [], [], [], []

    for _ in range(n_episodes):
        obs, _   = eval_env.reset()
        ep_reward = 0.0
        ep_steps  = 0
        done      = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = eval_env.step(action)
            ep_reward += reward
            ep_steps  += 1
            done = terminated or truncated

        # get_warehouse_env funciona igual aquí: eval_env → WarehouseEnv
        base_env  = get_warehouse_env(eval_env)
        success   = float(base_env.delivery or
                          (base_env.agent_has_object and not base_env.drop))
        collision = float(base_env.collision)

        rewards.append(ep_reward)
        lengths.append(ep_steps)
        successes.append(success)
        collisions.append(collision)

    eval_env.close()

    return {
        "rewards":    np.array(rewards),
        "lengths":    np.array(lengths),
        "successes":  np.array(successes),
        "collisions": np.array(collisions),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Gráficas
# ══════════════════════════════════════════════════════════════════════════════

def moving_average(data, w: int) -> np.ndarray:
    arr = np.array(data, dtype=float)
    if len(arr) < w:
        return arr
    return np.convolve(arr, np.ones(w) / w, mode="valid")


def plot_training(callback: MetricsCallback, entorno: int, timesteps: int) -> plt.Figure:
    rewards    = callback.ep_rewards
    lengths    = callback.ep_lengths
    successes  = callback.ep_successes
    collisions = callback.ep_collisions
    episodes   = np.arange(1, len(rewards) + 1)

    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(
        f"Entorno {entorno} — Métricas de entrenamiento\n"
        f"({timesteps:,} pasos  |  {len(rewards)} episodios)",
        fontsize=14, fontweight="bold",
    )
    gs = gridspec.GridSpec(2, 2, hspace=0.45, wspace=0.35)

    specs = [
        (gs[0, 0], rewards,    "Recompensa por episodio",     "#2196F3", "Recompensa acumulada", False),
        (gs[0, 1], lengths,    "Duración por episodio",        "#FF9800", "Pasos",                False),
        (gs[1, 0], successes,  "Tasa de éxito por episodio",   "#4CAF50", "Éxito",                True),
        (gs[1, 1], collisions, "Tasa de choques por episodio", "#F44336", "Choque",               True),
    ]

    for pos, data, title, color, ylabel, pct in specs:
        ax  = fig.add_subplot(pos)
        ma  = moving_average(data, WINDOW)
        ep_ma = episodes[WINDOW - 1:] if len(episodes) >= WINDOW else episodes

        ax.plot(episodes, data, alpha=0.2, color=color, linewidth=0.8)
        ax.plot(ep_ma, ma, color=color, linewidth=2.0,
                label=f"Media móvil ({WINDOW} ep.)")
        ax.set_xlabel("Episodio", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        if pct:
            ax.set_ylim(-0.05, 1.05)
            ax.yaxis.set_major_formatter(
                plt.FuncFormatter(lambda v, _: f"{v*100:.0f}%"))

    return fig


def plot_evaluation(metrics: dict, entorno: int, n_episodes: int) -> plt.Figure:
    rewards    = metrics["rewards"]
    lengths    = metrics["lengths"]
    successes  = metrics["successes"]
    collisions = metrics["collisions"]
    episodes   = np.arange(1, n_episodes + 1)

    sr = successes.mean() * 100
    cr = collisions.mean() * 100
    mr = rewards.mean()
    ml = lengths.mean()

    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(
        f"Entorno {entorno} — Métricas de evaluación ({n_episodes} episodios)\n"
        f"Éxito: {sr:.1f}%  |  Choques: {cr:.1f}%  |  "
        f"Retorno medio: {mr:.1f}  |  Duración media: {ml:.1f} pasos",
        fontsize=13, fontweight="bold",
    )
    gs = gridspec.GridSpec(2, 2, hspace=0.45, wspace=0.35)

    # Recompensa — barras
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.bar(episodes, rewards, color="#2196F3", alpha=0.6, width=0.8)
    ax1.axhline(mr, color="#0D47A1", linewidth=2, linestyle="--",
                label=f"Media: {mr:.1f}")
    ax1.set_xlabel("Episodio", fontsize=10)
    ax1.set_ylabel("Recompensa acumulada", fontsize=10)
    ax1.set_title("Recompensa por episodio", fontsize=11, fontweight="bold")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3, axis="y")

    # Duración — barras
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.bar(episodes, lengths, color="#FF9800", alpha=0.6, width=0.8)
    ax2.axhline(ml, color="#E65100", linewidth=2, linestyle="--",
                label=f"Media: {ml:.1f}")
    ax2.set_xlabel("Episodio", fontsize=10)
    ax2.set_ylabel("Pasos", fontsize=10)
    ax2.set_title("Duración por episodio", fontsize=11, fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, axis="y")

    # Éxito — pie
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.pie(
        [sr, 100 - sr],
        labels=[f"Éxito\n{sr:.1f}%", f"Fallo\n{100-sr:.1f}%"],
        colors=["#4CAF50", "#BDBDBD"],
        startangle=90,
        wedgeprops={"edgecolor": "white", "linewidth": 2},
        textprops={"fontsize": 11},
    )
    ax3.set_title("Tasa de éxito global", fontsize=11, fontweight="bold")

    # Choques — pie
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.pie(
        [cr, 100 - cr],
        labels=[f"Choque\n{cr:.1f}%", f"Sin choque\n{100-cr:.1f}%"],
        colors=["#F44336", "#BDBDBD"],
        startangle=90,
        wedgeprops={"edgecolor": "white", "linewidth": 2},
        textprops={"fontsize": 11},
    )
    ax4.set_title("Tasa de choques global", fontsize=11, fontweight="bold")

    return fig


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def build_parser():
    p = argparse.ArgumentParser(description="DQN almacén — entrenamiento y métricas")
    p.add_argument("--entorno",       type=int,   choices=[1, 2, 3], default=1)
    p.add_argument("--timesteps",     type=int,   default=30_000)
    p.add_argument("--lr",            type=float, default=1e-3)
    p.add_argument("--gamma",         type=float, default=0.99)
    p.add_argument("--seed",          type=int,   default=0)
    p.add_argument("--eval-episodes", type=int,   default=100)
    p.add_argument("--save-dir",      type=str,   default="resultados")
    return p


def main():
    args   = build_parser().parse_args()
    cfg    = ENV_CONFIG[args.entorno]
    outdir = Path(args.save_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    obs_builder = ObservationBuilder()
    train_env   = FullWrapper(WarehouseEnv(**cfg), obs_builder)

    model = DQN(
        "MlpPolicy", train_env,
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

    callback = MetricsCallback()

    print(f"\nEntrenando Entorno {args.entorno} — {args.timesteps:,} pasos...\n")
    model.learn(total_timesteps=args.timesteps, callback=callback)
    train_env.close()

    # Resumen consola — entrenamiento
    print(f"\n{'═'*55}")
    print(f"  RESUMEN ENTRENAMIENTO — Entorno {args.entorno}")
    print(f"{'═'*55}")
    print(f"  Episodios:         {len(callback.ep_rewards)}")
    print(f"  Recompensa media:  {np.mean(callback.ep_rewards):.2f}")
    print(f"  Duración media:    {np.mean(callback.ep_lengths):.1f} pasos")
    print(f"  Tasa de éxito:     {np.mean(callback.ep_successes)*100:.1f}%")
    print(f"  Tasa de choques:   {np.mean(callback.ep_collisions)*100:.1f}%")
    print(f"{'═'*55}\n")

    # Gráfica de entrenamiento
    fig_train = plot_training(callback, args.entorno, args.timesteps)
    path_train = outdir / f"entorno{args.entorno}_entrenamiento.png"
    fig_train.savefig(path_train, dpi=150, bbox_inches="tight")
    print(f"Gráfica entrenamiento → {path_train}")

    # Evaluación
    print(f"\nEvaluando ({args.eval_episodes} episodios, determinista)...\n")
    eval_metrics = evaluate_with_metrics(model, cfg, obs_builder, args.eval_episodes)

    # Resumen consola — evaluación
    print(f"{'═'*55}")
    print(f"  RESUMEN EVALUACIÓN — Entorno {args.entorno}")
    print(f"{'═'*55}")
    print(f"  Episodios:         {args.eval_episodes}")
    print(f"  Recompensa media:  {eval_metrics['rewards'].mean():.2f} "
          f"+/- {eval_metrics['rewards'].std():.2f}")
    print(f"  Duración media:    {eval_metrics['lengths'].mean():.1f} "
          f"+/- {eval_metrics['lengths'].std():.1f} pasos")
    print(f"  Tasa de éxito:     {eval_metrics['successes'].mean()*100:.1f}%")
    print(f"  Tasa de choques:   {eval_metrics['collisions'].mean()*100:.1f}%")
    print(f"{'═'*55}\n")

    # Gráfica de evaluación
    fig_eval = plot_evaluation(eval_metrics, args.entorno, args.eval_episodes)
    path_eval = outdir / f"entorno{args.entorno}_evaluacion.png"
    fig_eval.savefig(path_eval, dpi=150, bbox_inches="tight")
    print(f"Gráfica evaluación  → {path_eval}")

    plt.show()


if __name__ == "__main__":
    main()