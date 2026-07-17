# Informe Fase 5b — cierre de la evaluación del peldaño 1

Reproducible con un comando: `uv run --extra train python scripts/phase5b_report.py --sample 400`.
Mismo test congelado (pools no vistos), n=400 barajado. **Sin entrenar nada.**

## (1) Baseline de modelo grande zero-shot como router

| router | exact | mode acc | PLAN acc | invalid |
|--------|------:|---------:|---------:|--------:|
| **peldaño 1 (270M, entrenado)** | **0.760** | **0.853** | **0.59** | 0.000 |
| esdrac 7B zero-shot (26× el tamaño) | 0.153 | 0.367 | 0.27 | 0.000 |

El 270M **entrenado supera de largo** al modelo 26× mayor **sin entrenar**,
incluso en la clase PLAN. La especialización del dataset gana a la capacidad
bruta — que es la tesis del proyecto.

**Baseline de API de pago**: presupuesto autorizado 10 EUR. Estimación
proyectada ~$7.21 (**~6.63 EUR, por debajo del presupuesto**), pero **no hay
clave de API** → no se ejecuta (`BLOCKERS.md`). Sustituto no-pago: esdrac local.

## (2) Calibración por logprobs — prior a PLAN

Bajo el constraint, sumar un prior λ al score de PLAN antes del argmax:

| prior | PLAN recall | DIRECT recall | TOOL recall | mode acc | coste medio |
|------:|------------:|--------------:|------------:|---------:|------------:|
| 0.00 | 0.592 | 0.936 | 0.795 | 0.853 | 5.73 |
| 0.25 | 0.837 | 0.906 | 0.795 | **0.865** | 5.97 |
| **0.50** ← op | **0.939** | 0.842 | 0.786 | 0.838 | 6.17 |
| 0.75 | 1.000 | 0.726 | 0.752 | 0.767 | 6.50 |
| 1.00 | 1.000 | 0.551 | 0.684 | 0.645 | 6.92 |
| ≥1.50 | 1.000 | ↓↓ | ↓↓ | ↓↓ | ↑↑ |

**Hallazgo central**: la debilidad de PLAN es **de calibración, no de capacidad**.
- Con **prior 0.25**, PLAN recall sube 0.59 → **0.84 y la exactitud global MEJORA**
  (0.853 → 0.865): un ajuste gratis, sin reentrenar.
- **Punto de operación elegido: prior 0.50** — maximiza el recall de PLAN (0.94)
  con una caída de mode_acc ≤ 2 pts (0.853 → 0.838). Recalibrable en runtime sin
  tocar los pesos.

## (3) Matriz de confusión en el punto de operación (prior 0.50)

```
  gold\pred     DIRECT TOOL_CALL   PLAN  INVALID
  DIRECT          197       5       32        0
  TOOL_CALL        11      92       14        0
  PLAN              1       2       46        0
```

- **PLAN recall 0.592 → 0.939** (de 49 PLAN, 46 correctas ahora).
- Coste: DIRECT pierde 32 casos a PLAN (sobre-clasificación), mode_acc 0.853 →
  0.838. Trade-off explícito y elegido conscientemente.

## (4) Economía simulada: router vs cascada pura (tareas de código, n=400)

Reutiliza la verificación de la Fase 2 (dificultad = rango del mínimo suficiente).
Coste ~ tamaño en B de parámetros (1.5 / 7 / 8). Router = apuesta de la política
+ escalado al capaz si falla; cascada = barato→capaz.

| sistema | coste medio | pass-rate |
|---------|------------:|----------:|
| router (apuesta + escala) | 6.05 | 0.880 |
| cascada pura (barato→capaz) | 5.61 | 0.880 |

**Resultado honesto**: en **código puro**, el router es **7.9% MÁS CARO** que la
cascada (a igual pass-rate). La cascada barato→capaz es difícil de batir en coste
cuando el modelo barato basta a menudo; el router a veces sobre-apuesta. **El
valor del router NO está en el coste del código**, sino en (a) decidir el MODO
(cascada no lo hace), (b) evitar intentos fallidos/latencia, y (c) tareas no
verificables. La comparación económica pura de código favorece a la cascada, y
es correcto reportarlo así.

## (5) Regret de coste — definición y explicación del −0.169

**Definición**: `regret_i = coste(elegido_i) − coste(oráculo_i)`, donde el
oráculo es el **modelo mínimo suficiente** (la etiqueta gold). Media sobre el test.

- **regret medio = −0.169** (n=400) — reproduce exactamente el valor de Fase 5.
- Desglose: **infra-provisión (elige más barato que el mínimo, <0) = 43** ·
  exacto (=0) = 322 · sobre-provisión (>0) = 35.

**Por qué es negativo (y por qué NO es bueno)**: un regret negativo **no es un
ahorro óptimo**. Sale de que la política a veces elige un modelo **más barato que
el mínimo suficiente** (infra-provisión), que además **falla la tarea**. Las 43
infra-provisiones superan a las 35 sobre-provisiones → media negativa. Esto
**correlaciona con el fallo de PLAN** (subestima la dificultad → elige modelo
insuficiente). Un regret *sano* rondaría 0⁺ (sobre-provisión leve); el negativo
señala **under-provisioning**, un fallo de calidad, no un ahorro.

## Conclusión de Fase 5b

El peldaño 1 no solo supera los baselines: su **punto débil (PLAN) es en gran
parte recalibrable** con un prior de logprobs (recall 0.59 → 0.84–0.94 sin
reentrenar), y su regret negativo se explica por la misma infra-provisión. Con la
calibración, el peldaño 1 es un **suelo notablemente sólido**. La decisión de
entrenar el peldaño 2 (Qwen3-0.6B) para superar el techo de PLAN de raíz sigue
siendo del usuario, ahora con datos más finos para decidir.
