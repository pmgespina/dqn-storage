"""Entorno de almacén (Ejercicio 4) — versión PROFESOR.

Recinto cuadrado 10 x 10 con tres estanterías, una zona de entrega y
hasta tres objetos. El agente tiene seis acciones (mover en 4
direcciones, coger, soltar) y debe resolver tres variantes cada vez
más complejas:

  * Entorno 1: ``random_objects=False, drop=False`` (sólo recoger).
  * Entorno 2: ``random_objects=False, drop=True`` (recoger y entregar).
  * Entorno 3: ``random_objects=True,  drop=True`` (igual que el 2 pero
    con posiciones de objetos aleatorias en las estanterías).

API Gymnasium (``reset``, ``step``, ``render``, ``close``).
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

# Acciones
UP, DOWN, LEFT, RIGHT, PICK, DROP = 0, 1, 2, 3, 4, 5


class WarehouseEnv(gym.Env):
    """Entorno de almacén con recogida y entrega de objetos.

    Args:
        random_objects: si ``True``, las posiciones de los objetos se
            sortean al inicio de cada episodio.
        drop: si ``True``, el agente puede soltar objetos; el episodio
            termina con éxito si suelta dentro de la zona de entrega.
            Si ``False``, el episodio termina al recoger un objeto.
        max_steps: límite de pasos por episodio (truncamiento).
        render_mode: ``"human"``, ``"rgb_array"`` o ``None``.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    # --- Geometría del entorno ---
    WIDTH = 10.0
    HEIGHT = 10.0
    SHELVES = (
        (1.9, 1.0, 0.2, 5.0),
        (4.9, 1.0, 0.2, 5.0),
        (7.9, 1.0, 0.2, 5.0),
    )
    DELIVERY_AREA = (2.5, 9.0, 5.0, 1.0)
    AGENT_RADIUS = 0.2
    AGENT_VELOCITY = 0.25
    PICKUP_DISTANCE = 0.6
    FIXED_OBJECTS = ((2.0, 3.0), (5.0, 2.0), (8.0, 4.0))

    # --- Recompensa ---
    # El entorno devuelve reward=0 en todos los casos. El diseño de la
    # señal de recompensa es responsabilidad del alumno: implementar
    # calculate_reward() en observacion.py.

    def __init__(
        self,
        random_objects: bool = False,
        drop: bool = False,
        max_steps: int = 2000,
        render_mode: str | None = None,
    ):
        super().__init__()
        self.random_objects = random_objects
        self.drop = drop
        self.max_steps = max_steps
        self.render_mode = render_mode

        # Acciones disponibles
        n_actions = 6 if self.drop else 5
        self.action_space = spaces.Discrete(n_actions)
        self.observation_space = spaces.Box(
            low=0.0, high=10.0, shape=(11,), dtype=np.float32
        )

        # Estado
        self.agent_pos: tuple[float, float] = (0.0, 0.0)
        self.object_positions: list[tuple[float, float] | None] = []
        self.agent_has_object = False
        self.collision = False
        self.delivery = False
        self._steps = 0

        self._fig = None
        self._ax = None

    # ---------------------------------------------------------------- api
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.agent_pos = self._sample_empty_position()
        if self.random_objects:
            self.object_positions = [self._sample_on_shelf(s) for s in self.SHELVES]
        else:
            self.object_positions = list(self.FIXED_OBJECTS)
        self.agent_has_object = False
        self.collision = False
        self.delivery = False
        self._steps = 0
        return self._get_obs(), {}

    def step(self, action: int):
        self._steps += 1
        terminated = False
        truncated = False

        if action < 4:
            new_pos = self._move(action)
            if self._is_collision(new_pos):
                self.collision = True
                terminated = True
            else:
                self.agent_pos = new_pos
        elif action == PICK:
            self._try_pick()
            if self.agent_has_object and not self.drop:
                terminated = True
        elif action == DROP and self.drop:
            self._try_drop()
            terminated = True

        if self._steps >= self.max_steps:
            truncated = True

        # reward=0 siempre: el alumno diseña la recompensa en observacion.py
        return self._get_obs(), 0.0, terminated, truncated, {}

    # ---------------------------------------------------------------- helpers
    def _try_pick(self) -> None:
        if self.agent_has_object:
            return
        for i, pos in enumerate(self.object_positions):
            if pos is None:
                continue
            if self._distance(self.agent_pos, pos) <= self.PICKUP_DISTANCE:
                self.agent_has_object = True
                self.object_positions[i] = None
                return

    def _try_drop(self) -> None:
        if not self.agent_has_object:
            return
        self.agent_has_object = False
        if self._in_area(self.agent_pos, self.DELIVERY_AREA):
            self.delivery = True
        else:
            self.object_positions.append(self.agent_pos)

    def _get_obs(self) -> np.ndarray:
        obs = np.zeros(11, dtype=np.float32)
        obs[0:2] = self.agent_pos
        for i in range(3):
            pos = self.object_positions[i] if i < len(self.object_positions) else None
            if pos is not None:
                obs[2 + 2 * i: 4 + 2 * i] = pos
            else:
                obs[2 + 2 * i: 4 + 2 * i] = self.agent_pos
        obs[8] = float(self.agent_has_object)
        obs[9] = float(self.collision)
        obs[10] = float(self.delivery)
        return obs

    def _move(self, action: int) -> tuple[float, float]:
        x, y = self.agent_pos
        if action == UP:
            y = min(self.HEIGHT - self.AGENT_RADIUS, y + self.AGENT_VELOCITY)
        elif action == DOWN:
            y = max(self.AGENT_RADIUS, y - self.AGENT_VELOCITY)
        elif action == LEFT:
            x = max(self.AGENT_RADIUS, x - self.AGENT_VELOCITY)
        elif action == RIGHT:
            x = min(self.WIDTH - self.AGENT_RADIUS, x + self.AGENT_VELOCITY)
        return x, y

    def _is_collision(self, pos: tuple[float, float]) -> bool:
        x, y = pos
        if (
            x <= self.AGENT_RADIUS
            or x >= self.WIDTH - self.AGENT_RADIUS
            or y <= self.AGENT_RADIUS
            or y >= self.HEIGHT - self.AGENT_RADIUS
        ):
            return True
        for sx, sy, sw, sh in self.SHELVES:
            if (
                sx - self.AGENT_RADIUS <= x <= sx + sw + self.AGENT_RADIUS
                and sy - self.AGENT_RADIUS <= y <= sy + sh + self.AGENT_RADIUS
            ):
                return True
        return False

    def _sample_empty_position(self) -> tuple[float, float]:
        rng = self.np_random
        while True:
            x = float(rng.uniform(self.AGENT_RADIUS, self.WIDTH - self.AGENT_RADIUS))
            y = float(rng.uniform(self.AGENT_RADIUS, self.HEIGHT - self.AGENT_RADIUS))
            if not self._is_collision((x, y)):
                return x, y

    def _sample_on_shelf(self, shelf) -> tuple[float, float]:
        rng = self.np_random
        sx, sy, sw, sh = shelf
        x = sx + (0.25 if rng.uniform() < 0.5 else 0.75) * sw
        y = float(rng.uniform(sy + 0.5, sy + sh - 0.5))
        return x, y

    @staticmethod
    def _distance(a, b) -> float:
        return float(np.hypot(a[0] - b[0], a[1] - b[1]))

    @staticmethod
    def _in_area(pos, area, margin: float = 0.0) -> bool:
        ax, ay, aw, ah = area
        return (
            ax - margin <= pos[0] <= ax + aw + margin
            and ay - margin <= pos[1] <= ay + ah + margin
        )

    # ---------------------------------------------------------------- render
    def render(self):
        if self.render_mode is None:
            return None

        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle, Rectangle

        if self._fig is None:
            self._fig, self._ax = plt.subplots(figsize=(6, 6))
            plt.ion()

        self._ax.clear()
        self._ax.set_xlim(0, self.WIDTH)
        self._ax.set_ylim(0, self.HEIGHT)
        self._ax.set_aspect("equal")

        for sx, sy, sw, sh in self.SHELVES:
            self._ax.add_patch(Rectangle((sx, sy), sw, sh, fill=False, edgecolor="red"))

        dx, dy, dw, dh = self.DELIVERY_AREA
        self._ax.add_patch(
            Rectangle((dx, dy), dw, dh, fill=True, facecolor="lightgreen", edgecolor="green", alpha=0.5)
        )

        for pos in self.object_positions:
            if pos is not None:
                self._ax.add_patch(Circle(pos, radius=0.2, fill=True, facecolor="blue"))

        color = "red" if self.agent_has_object else "orange"
        self._ax.add_patch(Circle(self.agent_pos, radius=self.AGENT_RADIUS, fill=True, facecolor=color))

        self._ax.set_title(f"Warehouse (drop={self.drop}, random={self.random_objects})")
        plt.draw()
        plt.pause(0.01)

        if self.render_mode == "rgb_array":
            self._fig.canvas.draw()
            image = np.frombuffer(self._fig.canvas.tostring_rgb(), dtype=np.uint8)
            return image.reshape(self._fig.canvas.get_width_height()[::-1] + (3,))
        return None

    def close(self):
        if self._fig is not None:
            import matplotlib.pyplot as plt

            plt.close(self._fig)
            self._fig = None
            self._ax = None


if __name__ == "__main__":
    env = WarehouseEnv(random_objects=False, drop=False, render_mode="human")
    obs, _ = env.reset(seed=0)
    for _ in range(50):
        action = env.action_space.sample()
        obs, reward, term, trunc, _ = env.step(action)
        env.render()
        if term or trunc:
            obs, _ = env.reset()
    env.close()
