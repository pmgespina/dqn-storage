"""Diseño de la observación y la recompensa para el agente DQN — versión ALUMNO.

Este es el fichero que debes completar. Tienes dos tareas:

1. **Observación** — Transformar la observación cruda del entorno
   (11 valores) en un vector de features compacto para la red neuronal.
2. **Recompensa** — Diseñar la señal de recompensa que guía el
   aprendizaje del agente. El entorno devuelve ``reward=0`` siempre;
   tú defines qué recompensa recibe el agente en cada transición.

Observación cruda del entorno (``raw_obs``):
    obs[0:2]   — posición del agente (x, y) en metros, rango [0, 10].
    obs[2:8]   — posiciones de los 3 objetos (x1,y1, x2,y2, x3,y3).
                 Si un objeto ha sido recogido, su posición se iguala a
                 la del agente.
    obs[8]     — agent_has_object (0 o 1).
    obs[9]     — collision (0 o 1).
    obs[10]    — delivery (0 o 1).

Zona de entrega:
    Rectángulo con vértice inferior izquierdo en (2.5, 9.0), ancho 5.0,
    alto 1.0.

Pistas para la observación:
  * Normaliza las posiciones al rango [0, 1] (dividiendo por 10).
  * Piensa qué información necesita el agente para decidir su próxima
    acción. No le des más de la necesaria.
  * Incluye ``has_object`` como feature — el comportamiento óptimo es
    radicalmente diferente según lleve objeto o no.
  * La distancia y dirección al objetivo (objeto más cercano o zona de
    entrega) es más útil que las coordenadas absolutas de todos los
    objetos.
  * Cuanto menor sea ``obs_dim``, más rápido aprende la red (si las
    features son buenas).

Pistas para la recompensa:
  * Usa los flags de la observación cruda (collision, delivery,
    has_object) para detectar los eventos relevantes.
  * Compara ``prev_obs`` y ``obs`` para detectar transiciones: por
    ejemplo, si ``prev_obs[8]=0`` y ``obs[8]=1``, el agente acaba de
    coger un objeto.
  * Piensa en qué comportamiento quieres fomentar y cuál penalizar.
  * Una recompensa densa (que dé señal en cada paso) suele ayudar más
    que una dispersa (sólo al final del episodio).
"""
from __future__ import annotations

import numpy as np


class ObservationBuilder:
    """Construye el vector de features y la recompensa."""

    def __init__(self):
        # Definimos la dimensión del vector de features
        # 1. Posición agente x (norm)
        # 2. Posición agente y (norm)
        # 3. has_object (0 o 1)
        # 4. Distancia al objetivo (norm)
        # 5. Delta x al objetivo (norm)
        # 6. Delta y al objetivo (norm)
        self._obs_dim = 6
        self.delivery_zone_center = np.array([5.0, 9.5])

    def build(self, raw_obs: np.ndarray) -> np.ndarray:
        """Transforma ``raw_obs`` (11 floats) en el feature vector.

        Returns:
            np.ndarray de longitud ``self.obs_dim`` con dtype float32.
        """
        # 1. Extraer datos básicos y normalizar a [0, 1]
        agent_pos = raw_obs[0:2] / 10.0
        has_object = raw_obs[8]
        
        # 2. Identificar el objetivo (Target)
        if has_object == 0:
            # El objetivo es el objeto más cercano que no haya sido recogido
            obj_positions = raw_obs[2:8].reshape(3, 2)
            # Calculamos distancias al agente
            distances = np.linalg.norm(obj_positions - raw_obs[0:2], axis=1)
            # Seleccionamos el más cercano (que no sea la posición del agente)
            target_pos = obj_positions[np.argmin(distances)]
        else:
            # El objetivo es el centro de la zona de entrega
            target_pos = self.delivery_zone_center

        # 3. Calcular vector relativo al objetivo
        relative_vec = (target_pos - raw_obs[0:2]) / 10.0
        distance_to_target = np.linalg.norm(relative_vec)

        # 4. Construir el vector final (dtype float32 para la red neuronal)
        features = np.array([
            agent_pos[0],
            agent_pos[1],
            has_object,
            distance_to_target,
            relative_vec[0],
            relative_vec[1]
        ], dtype=np.float32)

        return features

    @property
    def obs_dim(self) -> int:
        """Número de features que devuelve ``build()``."""
        return self._obs_dim

    def calculate_reward(
        self,
        prev_obs: np.ndarray,
        obs: np.ndarray,
        terminated: bool,
        truncated: bool,
    ) -> float:
        """Calcula la recompensa de una transición.

        Se llama después de cada ``env.step()``. Recibe la observación
        cruda **antes** y **después** de la acción, y los flags de
        finalización del episodio.

        Args:
            prev_obs: observación cruda (11 floats) antes de la acción.
            obs: observación cruda (11 floats) después de la acción.
            terminated: True si el episodio terminó (colisión, recogida
                en entorno 1, o drop en entornos 2/3).
            truncated: True si se alcanzó el límite de pasos.

        Returns:
            float con la recompensa de esta transición.
        """
        reward = 0.0

        # 1. Penalización base suave por paso (fomenta no perder el tiempo)
        reward -= 0.05

        # ==========================================================
        # 2. EVENTOS DEL ENTORNO
        # ==========================================================
        
        if obs[9] == 1:
            # ¡CLAVE! Reducimos el castigo drásticamente de -10.0 a -0.5.
            # Rozar una pared es malo, pero no es el fin del mundo.
            # Esto permite al agente atreverse a navegar entre estanterías.
            reward -= 0.5
            
        if prev_obs[8] == 0 and obs[8] == 1:
            # Premio grande por cumplir su trabajo (recoger)
            reward += 20.0
            
        if obs[10] == 1:
            # Premio masivo por entrega final (Entornos 2 y 3)
            reward += 50.0
            
        if prev_obs[8] == 1 and obs[8] == 0 and obs[10] == 0:
            # Penalización por soltar el objeto al suelo por error
            reward -= 5.0

        # ==========================================================
        # 3. REWARD SHAPING (Gradiente de distancia)
        # ==========================================================
        d_prev = self._get_dist_to_target(prev_obs)
        d_curr = self._get_dist_to_target(obs)
        
        # Mantenemos el empujoncito positivo si se acerca al objetivo
        reward += (d_prev - d_curr) * 1.0  

        return float(reward)
    
    def _get_dist_to_target(self, raw_obs):
        """Función auxiliar para calcular la distancia al objetivo actual."""
        agent_p = raw_obs[0:2]
        if raw_obs[8] == 0:
            objs = raw_obs[2:8].reshape(3, 2)
            dists = np.linalg.norm(objs - agent_p, axis=1)
            return np.min(dists)
        else:
            return np.linalg.norm(self.delivery_zone_center - agent_p)
