---
license: apache-2.0
base_model: google/gemma-3-270m
library_name: transformers
pipeline_tag: text-generation
tags:
  - routing
  - model-routing
  - on-device
  - gguf
  - llama.cpp
  - ollama
language:
  - en
---

# Shiori (栞) — router de modelos on-device (peldaño 1, Gemma 3 270M)

Política de routing **más pequeña posible**: dado un registro de modelos
renderizado en el prompt (formato fijo y compacto) y una tarea, predice

```json
{"mode": "DIRECT | TOOL_CALL | PLAN", "model_id": "<el mínimo suficiente del pool>"}
```

Corre **100% en local** (llama.cpp / Ollama / LM Studio) con **decoding
constreñido** al enum de modos y a los `model_id` del pool presente en el prompt.
Full fine-tune de `google/gemma-3-270m` (268M parámetros). Este es el **peldaño 1**
de una escalera (270M → 0.6B → 1.7B); se publica porque cumple el criterio de
selección, con los límites honestos de abajo.

Código y pipeline: https://github.com/marcmayol/shiori

## Qué hace (y qué NO)

- **Predice**, no ejecuta ni verifica. La detección de fallo y el escalado son
  código alrededor del modelo (try → verify → next).
- Elige el **modelo mínimo suficiente** del pool que se le presenta, leyendo el
  registro (tags, coste, ctx). Generaliza a pools **nunca vistos** (marcas/costes
  aleatorizados en el entrenamiento).

## Métricas (test congelado, pools NUNCA vistos, n=400)

Con y sin la **calibración** de modo (prior a PLAN, ver abajo):

| | exact | mode acc | model_id acc | DIRECT | TOOL_CALL | PLAN |
|---|------:|---------:|-------------:|-------:|----------:|-----:|
| **sin calibración** (prior 0) | 0.760 | 0.853 | 0.802 | 0.94 | 0.79 | **0.59** |
| **prior 0.25** | ~0.77 | **0.865** | — | 0.91 | 0.80 | **0.84** |
| **prior 0.50** (por defecto) | ~0.76 | 0.838 | — | 0.84 | 0.79 | **0.94** |

- **Tasa de salida inválida**: 0.013 sin constraint → **0.000 con constraint**.
- Generalización: pools vistos (train) 0.93 vs no vistos (test) 0.76 (gap +0.17,
  no se hunde).

### Curva del prior a PLAN (calibración por logprobs)

| prior | PLAN recall | DIRECT recall | TOOL recall | mode acc | coste medio |
|------:|------------:|--------------:|------------:|---------:|------------:|
| 0.00 | 0.592 | 0.936 | 0.795 | 0.853 | 5.73 |
| 0.25 | 0.837 | 0.906 | 0.795 | **0.865** | 5.97 |
| 0.50 | 0.939 | 0.842 | 0.786 | 0.838 | 6.17 |
| ≥1.00 | 1.000 | ↓↓ | ↓↓ | ↓↓ | ↑↑ |

**La debilidad de PLAN es de calibración, no de capacidad**: un prior de logprobs
(parámetro público `plan_prior`, 0.50 por defecto) recupera el recall de PLAN
**sin reentrenar**. `prior=0.25` es un ajuste casi gratis (mejora la exactitud
global); `0.50` maximiza el recall de PLAN.

## Comparación con baselines (mismo test)

| router | exact | mode acc | PLAN acc |
|--------|------:|---------:|---------:|
| **Shiori 270M (entrenado)** | **0.760** | **0.853** | 0.59 / **0.94** cal |
| gpt-4o zero-shot (API, ~frontera) | 0.228 | 0.718 | **0.00** |
| esdrac 7B zero-shot (26× tamaño) | 0.153 | 0.367 | 0.27 |
| cascada pura (siempre el más barato) | 0.212 | 0.585 | 0.00 |

El 270M **entrenado supera a gpt-4o zero-shot** en esta tarea: exige aprender la
semántica de "mínimo suficiente" y el formato fijo, no capacidad bruta.
(gpt-4o zero-shot: coste real ~0.49 EUR sobre 400.)

## Comparación económica (honesta)

Simulación con la verificación real del pool (qwen2.5-coder 1.5b/7b + un 7B
capaz), coste ~ tamaño en B de parámetros, router = apuesta + escalado al capaz:

| sistema (tareas de CÓDIGO) | coste medio | pass-rate |
|----------------------------|------------:|----------:|
| router (Shiori) | 6.05 | 0.880 |
| **cascada pura (barato→capaz)** | **5.61** | 0.880 |

En **código verificable la cascada pura es ~8% más barata** a igual pass-rate: si
el modelo barato basta a menudo, empezar por él y escalar es difícil de batir.
**El valor del router NO está ahí**, sino en: (a) decidir el **MODO** (la cascada
no lo hace), (b) evitar intentos fallidos y **latencia**, y (c) dominios **sin
verificador mecánico** (donde no puedes "probar y escalar"). Regret de coste
−0.169 = leve **infra-provisión** (elige algo más barato que el mínimo), no ahorro
óptimo; el parámetro `capability_prior` lo corrige a cambio de coste.

## Latencia (runtime local, llama.cpp)

Decisión ≈ 320 tokens de prompt + 24 de salida, `llama-bench`:

| cuant. | tamaño | CPU (8 hilos) | GPU (CUDA) | presupuesto |
|--------|-------:|--------------:|-----------:|:-----------:|
| Q8_0 | 286 MB | 512 ms | 35 ms | CPU<1s ✓ · GPU<50ms ✓ |
| Q4_K_M | 249 MB | 526 ms | 35 ms | CPU<1s ✓ · GPU<50ms ✓ |

CPU-only es funcional (< 1 s por decisión en 8 hilos).

## Uso

### Ollama (recomendado)

```bash
ollama create shiori-router -f Modelfile   # con el GGUF Q4_K_M
```

```python
from routerpolicy.inference.route import route   # repo shiori
decision = route("Design and implement a rate limiter with tests.", registry, plan_prior=0.5)
```

`route()` habla con Ollama local, genera el JSON schema (o GBNF) desde el
registro, aplica el prior de modo (0.50) y cae al **modelo más capaz del pool**
si algo falla. Sin ninguna llamada de red externa.

### Formato del registro (contrato FIJO)

```
Available models:
- <id> | tags:<t1,t2> | ctx:<n> | cost:<c> | loc:<local|api> | tools:<yes|no>
...
Task:
<la tarea>
```

## Datos y licencias

- Suficiencia (código): **HumanEval+ / MBPP+** (evalplus), verificación mecánica.
- Modo: **Hermes function-calling** (Apache-2.0, TOOL_CALL) y **Dolly-15k**
  (CC-BY-SA-3.0, DIRECT/PLAN) — no-gated. Etiquetas de suficiencia por ejecución
  real de un pool local. Ver `DATA_LICENSES.md` en el repo. Modelo Apache-2.0.

## Límites conocidos

- **PLAN cruda 0.59** (mitigable con el prior a 0.84–0.94, ver arriba).
- **Dependencia del formato de registro fijo**: fuera de ese contrato no funciona.
- **Dominios cubiertos**: código, tool-calling y chat general (EN). Otros dominios
  no están representados en el dataset.
- **No verifica**: la sufiencia real depende del código que la envuelve.
- Es el **peldaño 1** (270M); un peldaño mayor (Qwen3-0.6B) podría superar el
  techo de PLAN de raíz.
