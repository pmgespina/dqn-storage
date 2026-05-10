"""Diseño de la observación y la recompensa para el agente DQN — versión ALUMNO."""
from __future__ import annotations

import numpy as np

DELIVERY_X_MIN, DELIVERY_X_MAX = 2.5, 7.5
DELIVERY_Y_MIN, DELIVERY_Y_MAX = 9.0, 10.0
DELIVERY_CENTER = np.array([5.0, 9.5])
PICKUP_DISTANCE = 0.6

GAMMA          = 0.99   # debe coincidir con --gamma de main.py
POTENTIAL_SCALE = 5.0   # escala del potencial de distancia


class ObservationBuilder:
    """Construye el vector de features y la recompensa.

    Observación — 7 features:
        [0] agent_x / 10            — posición absoluta x (geometría del mapa)
        [1] agent_y / 10            — posición absoluta y (geometría del mapa)
        [2] has_object              — fase actual de la tarea
        [3] distancia al objetivo   — cuánto falta para llegar
        [4] dx al objetivo          — dirección x hacia el objetivo
        [5] dy al objetivo          — dirección y hacia el objetivo
        [6] in_pickup_range / in_delivery_zone
            — vale 1 cuando el agente está en posición de ejecutar
              la acción clave de la fase actual (PICK o DROP)

    Por qué 7 y no 9:
        - dist_to_obstacles eliminada: es ambigua (mezcla paredes y
          estanterías en un solo número) y redundante con dx/dy al
          objetivo, que ya implica dirección relativa a obstáculos.
        - La feature [6] unifica in_pickup_range e in_delivery_zone
          porque nunca son 1 simultáneamente (dependen de has_object).
          Reduce una dimensión sin perder información.

    Recompensa — potential-based shaping (Ng et al., 1999):
        F(s,s') = gamma * Phi(s') - Phi(s)

    El potencial Phi usa una discontinuidad en el borde de la zona de
    entrega (Phi=0 dentro, Phi=-k*d fuera) que:
        a) crea un gradiente fuerte para entrar,
        b) elimina cualquier incentivo a oscilar dentro o en el borde,
        c) garantiza matemáticamente que ningún ciclo acumula recompensa
           positiva (propiedad del shaping basado en potencial).
    """

    def __init__(self):
        self._obs_dim = 7
        self.delivery_zone_center = DELIVERY_CENTER

    def build(self, raw_obs: np.ndarray) -> np.ndarray:
        agent_pos  = raw_obs[0:2]
        has_object = raw_obs[8]

        # Objetivo según fase de la tarea
        if has_object == 0:
            target_pos = self._nearest_available_object(raw_obs)
        else:
            target_pos = self.delivery_zone_center

        relative_vec       = (target_pos - agent_pos) / 10.0
        distance_to_target = float(np.linalg.norm(relative_vec))

        # Feature de "acción clave disponible":
        # - Sin objeto: 1 si estamos en rango de PICK
        # - Con objeto: 1 si estamos dentro de la zona de DROP
        if has_object == 0:
            dist_to_obj    = self._dist_nearest_available(raw_obs)
            action_ready   = 1.0 if dist_to_obj <= PICKUP_DISTANCE else 0.0
        else:
            action_ready   = 1.0 if self._in_delivery_zone(agent_pos) else 0.0

        return np.array([
            agent_pos[0] / 10.0,
            agent_pos[1] / 10.0,
            has_object,
            distance_to_target,
            relative_vec[0],
            relative_vec[1],
            action_ready,
        ], dtype=np.float32)

    @property
    def obs_dim(self) -> int:
        return self._obs_dim

    def calculate_reward(
        self,
        prev_obs: np.ndarray,
        obs: np.ndarray,
        terminated: bool,
        truncated: bool,
    ) -> float:

        picked_now    = (prev_obs[8] == 0.0 and obs[8] == 1.0)
        delivered_now = (obs[10] == 1.0)
        dropped_wrong = (prev_obs[8] == 1.0 and obs[8] == 0.0 and obs[10] == 0.0)
        collided      = (obs[9] == 1.0)

        reward = 0.0

        # ── 1. Penalización por paso ──────────────────────────────────────
        # -0.1 es suficiente para incentivar eficiencia sin hacer que
        # explorar sea prohibitivo (2000 pasos = -200 en el peor caso,
        # muy por debajo del +120 que vale resolver el entorno 2).
        reward -= 0.1

        # ── 2. Colisión ───────────────────────────────────────────────────
        # Alta para superar el shaping máximo por paso (5 * 0.25 = 1.25),
        # de forma que ir recto hacia el objeto atravesando una estantería
        # nunca sea más rentable que rodearlo.
        if collided:
            reward -= 15.0

        # ── 3. Eventos de la tarea ────────────────────────────────────────
        if picked_now:
            reward += 20.0      # hito intermedio necesario

        if delivered_now:
            reward += 100.0     # objetivo principal

        if dropped_wrong:
            reward -= 30.0      # soltar fuera = peor que no hacer nada

        # ── 4. Potential-based shaping ────────────────────────────────────
        # No se aplica en pasos de evento para no mezclar señales
        # cuando el objetivo acaba de cambiar (picked/dropped/delivered).
        if not (picked_now or delivered_now or dropped_wrong):
            phi_prev = self._phi(prev_obs)
            phi_curr = self._phi(obs)
            reward  += GAMMA * phi_curr - phi_prev

        return float(reward)

    # ── Potencial ─────────────────────────────────────────────────────────

    def _phi(self, raw_obs: np.ndarray) -> float:
        """Función de potencial Phi(s).

        Fase 1 (sin objeto):
            Phi = -POTENTIAL_SCALE * distancia_al_objeto_más_cercano

        Fase 2 (con objeto):
            Fuera de la zona: Phi = -POTENTIAL_SCALE * distancia_al_centro
            Dentro de la zona: Phi = 0

        La discontinuidad en el borde de la zona de entrega es la clave:
        dentro Phi=0 significa que moverse dentro no aporta shaping,
        así que la única acción rentable es ejecutar DROP (+100).
        """
        agent_p = raw_obs[0:2]
        if raw_obs[8] == 0:
            return -POTENTIAL_SCALE * self._dist_nearest_available(raw_obs)
        else:
            if self._in_delivery_zone(agent_p):
                return 0.0
            return -POTENTIAL_SCALE * float(np.linalg.norm(self.delivery_zone_center - agent_p))

    # ── Helpers ───────────────────────────────────────────────────────────

    def _in_delivery_zone(self, pos: np.ndarray) -> bool:
        return (DELIVERY_X_MIN <= pos[0] <= DELIVERY_X_MAX
                and DELIVERY_Y_MIN <= pos[1] <= DELIVERY_Y_MAX)

    def _is_available(self, obj_pos: np.ndarray, agent_pos: np.ndarray) -> bool:
        """Objeto disponible = no ha sido recogido (posición ≠ agente)."""
        return float(np.linalg.norm(obj_pos - agent_pos)) > PICKUP_DISTANCE / 2

    def _nearest_available_object(self, raw_obs: np.ndarray) -> np.ndarray:
        agent_p = raw_obs[0:2]
        objs    = raw_obs[2:8].reshape(3, 2)
        best_dist, best_pos = np.inf, agent_p
        for obj in objs:
            if self._is_available(obj, agent_p):
                d = float(np.linalg.norm(obj - agent_p))
                if d < best_dist:
                    best_dist, best_pos = d, obj
        return best_pos

    def _dist_nearest_available(self, raw_obs: np.ndarray) -> float:
        if raw_obs[8] == 1:
            return 0.0
        agent_p = raw_obs[0:2]
        objs    = raw_obs[2:8].reshape(3, 2)
        best_dist = np.inf
        for obj in objs:
            if self._is_available(obj, agent_p):
                d = float(np.linalg.norm(obj - agent_p))
                if d < best_dist:
                    best_dist = d
        return best_dist if best_dist < np.inf else 0.0