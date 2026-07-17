# Informe Fase 5 — comparativo del peldaño 1 (Gemma 3 270M)

Reproducible con un comando: `uv run --extra train python scripts/phase5_report.py --sample 400`.
Todo sobre el **mismo test congelado** (pools nunca vistos). Muestra n=400 (barajada, semilla fija).

## Peldaño 1 — métricas descompuestas (pools NUNCA vistos)

| | exact | mode acc | model_id acc | invalid |
|---|------:|---------:|-------------:|--------:|
| sin constraint | 0.757 | 0.853 | 0.792 | 0.013 |
| **con constraint** | **0.760** | **0.853** | **0.802** | **0.000** |

- Exactitud de modo por clase: **DIRECT 0.94 · TOOL_CALL 0.79 · PLAN 0.59**.
- **regret de coste vs oráculo: −0.169** (elige algo más barato que el óptimo).
- El constraint elimina inválidos (0.013 → 0.000) y sube model_id acc.

### Matriz de confusión de modo (con constraint)

```
  gold\pred     DIRECT TOOL_CALL   PLAN  INVALID
  DIRECT          219       5       10        0
  TOOL_CALL        20      93        4        0
  PLAN             16       4       29        0
```

**A dónde van los errores de PLAN**: de 49 tareas PLAN, 29 correctas; **16 se
clasifican como DIRECT** (el 270M subestima la complejidad y no eleva a plan), 4
como TOOL_CALL. Ese es el punto débil del peldaño.

## Global vs pools nunca vistos (generalización)

| | exact | mode acc |
|---|------:|---------:|
| pools vistos (train) | 0.930 | 1.000 |
| pools NO vistos (test) | 0.760 | 0.853 |
| **gap** | **+0.170** | +0.147 |

Hay algo de sobreajuste, pero **no se hunde** en composiciones nunca vistas: la
augmentación de Fase 3 cumple su función de portabilidad.

## Suficiencia real (pool de generación, re-usando la verificación de Fase 2)

Sobre el pool REAL (qwen2.5-coder 1.5b/7b + esdrac), ¿la elección de la política
tiene capacidad suficiente para la tarea (capacidad ≥ dificultad de la cascada)?

| | tasa de suficiencia |
|---|------:|
| **política peldaño 1** | **0.825** |
| cascada pura (siempre el más barato) | 0.590 |
| oráculo (techo: alguno del pool basta) | 0.880 |

La política acierta el modelo suficiente el 82.5% (vs 59% de "siempre barato"),
cerca del techo del oráculo (88%).

## Baselines obligatorios (mismo test)

| baseline | exact | mode acc | model_id acc |
|----------|------:|---------:|-------------:|
| **peldaño 1 (con constraint)** | **0.760** | **0.853** | **0.802** |
| cascada pura (siempre el más barato) | 0.212 | 0.585 | 0.307 |
| Gemma 3 270M BASE zero-shot | 0.075 | 0.350 | 0.215 |
| API zero-shot como router | (no ejecutado — ver abajo) | | |

El peldaño 1 **supera con holgura** los dos baselines ejecutables.

**Baseline de API zero-shot**: NO ejecutado. No hay clave de API y el umbral [X]
del goal quedó indefinido. Estimación proyectada desde 100 tareas al test
completo: **~$7.21 (~6.63 EUR)** con precios ilustrativos (3/15 USD/Mtok).
Documentado en `BLOCKERS.md`.

## Latencia con llama.cpp (runtime local real)

Decisión = ~320 tokens de prompt + 24 de salida. `llama-bench`.

| cuant. | tamaño | CPU (8 hilos) | GPU (CUDA) | presupuesto |
|--------|-------:|--------------:|-----------:|:-----------:|
| Q8_0 | 286 MB | 537 ms | 37 ms | CPU<1s ✓ · GPU<50ms ✓ |
| Q4_K_M | 249 MB | 532 ms | 37 ms | CPU<1s ✓ · GPU<50ms ✓ |

Ambos presupuestos de la sección 2 cumplidos.

## Regla de selección aplicada

Elegir el peldaño más pequeño que: (a) supere todos los baselines, (b) quede a
<2 pts del mejor peldaño, (c) no se hunda en pools no vistos, (d) cumpla la
latencia CPU.

- (a) supera baselines: **sí** (0.76 vs 0.21 cascada, 0.075 base).
- (b) único peldaño probado → es el mejor por defecto.
- (c) generalización: gap +0.17, no se hunde.
- (d) latencia CPU < 1 s: **sí** (Q4_K_M ~0.53 s).

## Recomendación

El peldaño 1 **cumple la regla formal** (supera baselines y latencia) y sería el
modelo final si solo mirásemos eso. **Pero PLAN es débil (0.59)** y es la clase
donde importa no infra-clasificar. Recomendación: es un **suelo válido y
desplegable**, pero conviene **entrenar el peldaño 2 (Qwen3-0.6B)** y comparar
antes de fijar el final — la escalera existe justo para esto. Decisión del
usuario.
