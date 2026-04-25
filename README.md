# Ejercicio 4 — DQN en almacén

Ejercicio evaluable del bloque de Aprendizaje por Refuerzo de la asignatura MLII (MUBD).

## Objetivo

Entrenar un agente con **Deep Q-Network (DQN)** para que opere en un entorno de almacén: recoger objetos de estanterías y entregarlos en una zona de entrega. La clave del ejercicio es **diseñar la observación y la recompensa** — ingeniería de variables y reward shaping — que determinan cómo aprende la red neuronal.

## Estructura del código

```
alumno/
├── env.py            # Entorno del almacén (Gymnasium). NO MODIFICAR.
├── observacion.py    # FICHERO A COMPLETAR — observación y recompensa.
├── main.py           # Entrenamiento y evaluación con stable-baselines3.
└── README.md
```

### `env.py` — El entorno

Implementa `WarehouseEnv`, un entorno Gymnasium con un recinto de 10x10 m, tres estanterías, objetos y una zona de entrega. Devuelve una observación cruda de **11 valores** (posición del agente, posiciones de los objetos, flags de estado) y **`reward=0` siempre** — el diseño de la recompensa es responsabilidad tuya. Consulta el enunciado para los detalles.

Se instancia con dos flags que definen la variante:

```python
from env import WarehouseEnv

env = WarehouseEnv(random_objects=False, drop=False)   # Entorno 1
env = WarehouseEnv(random_objects=False, drop=True)    # Entorno 2
env = WarehouseEnv(random_objects=True,  drop=True)    # Entorno 3
```

### `observacion.py` — Tu tarea

Contiene la clase `ObservationBuilder` con tres métodos a completar:

- **`build(raw_obs) -> np.ndarray`** — Transforma la observación cruda de 11 floats en un vector de features compacto (`dtype=float32`) que la red neuronal usará como entrada.
- **`obs_dim -> int`** — Devuelve la dimensión del vector de features.
- **`calculate_reward(prev_obs, obs, terminated, truncated) -> float`** — Calcula la recompensa de cada transición. Recibe la observación cruda antes y después de la acción, lo que te permite detectar eventos (colisión, recogida, entrega) comparando los flags entre ambos instantes.

### `main.py` — Entrenamiento y evaluación

Script listo para usar que:

1. Crea el entorno (`WarehouseEnv`).
2. Lo envuelve con un `FullWrapper` que aplica tu `ObservationBuilder` automáticamente: transforma la observación con `build()` y reemplaza la recompensa con `calculate_reward()`.
3. Entrena un agente DQN de **stable-baselines3**.
4. Evalúa el agente entrenado e imprime métricas.
5. Opcionalmente visualiza el agente en acción.

No necesitas modificar este fichero salvo que quieras ajustar hiperparámetros.

## Instalación

```bash
pip install numpy gymnasium matplotlib stable-baselines3
```

> `stable-baselines3` instala PyTorch como dependencia. Si prefieres controlar la versión de PyTorch, instálalo antes ([instrucciones](https://pytorch.org/get-started/locally/)).

## Ejecución

```bash
# Entorno 1 — sólo recoger un objeto (20 000 pasos)
python main.py --entorno 1 --timesteps 20000

# Entorno 2 — recoger y entregar (más pasos)
python main.py --entorno 2 --timesteps 50000

# Entorno 3 — objetos en posición aleatoria
python main.py --entorno 3 --timesteps 100000

# Visualizar el agente tras entrenar
python main.py --entorno 1 --render

# Ajustar hiperparámetros
python main.py --entorno 1 --lr 5e-4 --gamma 0.95 --timesteps 30000
```

### Opciones disponibles

| Argumento | Descripción | Por defecto |
|-----------|-------------|-------------|
| `--entorno` | Variante del entorno (1, 2 o 3) | 1 |
| `--timesteps` | Pasos totales de entrenamiento | 20 000 |
| `--lr` | Tasa de aprendizaje | 1e-3 |
| `--gamma` | Factor de descuento | 0.99 |
| `--seed` | Semilla aleatoria | 0 |
| `--render` | Visualizar el agente tras entrenar | desactivado |
| `--eval-episodes` | Episodios de evaluación | 50 |

## Cómo funciona la integración

El flujo de datos en cada paso es:

```
WarehouseEnv.step(action)
       │
       ▼
  raw_obs (11 floats),  reward=0
       │
       ▼
  FullWrapper
       │  ObservationBuilder.build(raw_obs)      → features
       │  ObservationBuilder.calculate_reward()   → reward
       ▼
  features (obs_dim floats),  reward (tu diseño)
       │
       ▼
  DQN (stable-baselines3)
       │  red neuronal → Q(s, a)
       ▼
  action
```

Tu `ObservationBuilder` actúa como **pasarela** entre el entorno y el agente. El agente nunca ve la observación cruda de 11 valores ni la recompensa nula del entorno — sólo ve las features y la recompensa que tú diseñes. Por eso el diseño de ambas es determinante para el aprendizaje.

## Consejos

### Observación
- Empieza por el **Entorno 1** (el más sencillo) y verifica que tu observación permite al agente aprender antes de pasar al 2 y al 3.
- **Normaliza** las posiciones (dividir por 10) para que todas las features estén en rangos similares.
- Piensa en qué información es **realmente útil** para decidir la siguiente acción. Menos features bien elegidas funcionan mejor que muchas redundantes.
- Incluye `has_object` como feature: el comportamiento óptimo cambia radicalmente según el agente lleve objeto o no.

### Recompensa
- Compara `prev_obs` y `obs` para detectar **transiciones**: `prev_obs[8]=0` y `obs[8]=1` significa que el agente acaba de coger un objeto.
- Una recompensa **densa** (señal en cada paso) suele funcionar mejor que una dispersa (sólo al final del episodio).
- El diseño de la recompensa probablemente deba cambiar entre entornos: la tarea del Entorno 1 (llegar y coger) no es la misma que la del 2 (coger y entregar).

### Referencia
- Con un buen diseño de observación (5–8 features) y recompensa, el Entorno 1 se resuelve en ~20 000 pasos.
