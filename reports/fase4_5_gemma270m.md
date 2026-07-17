# Informe Fase 4/5 — Peldaño 1: Gemma 3 270M (full fine-tune)

Primer peldaño de la escalera. Full fine-tune de `google/gemma-3-270m` en la
RTX 5070 Ti (bf16, transformers+trl), sobre el dataset de Fase 3.

## Entrenamiento (recursos, sección 3)

| Métrica | Valor |
|---------|------:|
| Parámetros | 268.1 M |
| Método | full fine-tune (bf16) |
| batch / seq / epochs | 16 / 768 (fijo) / 3 |
| VRAM pico | **14.31 GB** (< 16 GB, sin OOM) |
| tokens/s (train) | 11 233 |
| train_runtime | 4340 s (~72 min) |
| loss final | 0.055 |

Claves técnicas:
- **Pérdida y scoring selectivos**: se proyecta `lm_head` (vocab 262k de Gemma)
  solo en las posiciones del assistant. Sin esto: 19 s/paso; con ello 0.48 s/paso.
- Plantilla de chat propia (el base no trae) + máscara de pérdida solo-assistant.
- Shape fija (768) para no recompilar kernels por longitud; batch 32 saturaba VRAM.
- Reanudación robusta (salta checkpoints incompletos) verificada con el ciclo
  lanzar→interrumpir→relanzar.

Smoke test (overfit 200): **exact match 1.000** (valida el pipeline).

## Evaluación Fase 5 (test congelado, 500 barajados, pools NUNCA vistos)

Métricas CON y SIN decoding constreñido:

| Métrica | SIN constraint | CON constraint |
|---------|---------------:|---------------:|
| exact match | 0.770 | 0.774 |
| mode accuracy | 0.866 | 0.866 |
| model_id acc | 0.806 | 0.816 |
| **invalid rate** | 0.010 | **0.000** |

Exactitud de modo por clase: **DIRECT 0.943 · TOOL_CALL 0.821 · PLAN 0.571**.

- **regret de coste medio**: −0.19 (tiende a elegir modelos algo más baratos que
  el oráculo; cost-efficient, con riesgo de infra-provisión puntual).
- El constraint **elimina las salidas inválidas** (0.010 → 0.000) y sube algo la
  precisión de model_id, con la exactitud de modo intacta.

## Latencia (runtime local real, llama.cpp — sección 2)

GGUF exportado y cuantizado; latencia por decisión en **CPU, 8 hilos**
(decisión = ~320 tokens de prompt + 24 de salida), medida con `llama-bench`:

| Cuant. | Tamaño | Latencia/decisión | Presupuesto <1 s |
|--------|-------:|------------------:|:----------------:|
| Q8_0 | 286 MB | ~790 ms | ✅ |
| Q4_K_M | 249 MB | ~756 ms | ✅ |

(Latencia de scoring en transformers/GPU: p50 86 ms — no es la del artefacto,
que es llama.cpp.)

## Lectura

- El peldaño 1 **generaliza a pools no vistos** con 0% de salidas inválidas bajo
  constraint y cumple el presupuesto de latencia CPU (< 1 s).
- **PLAN (0.57)** es el punto débil: un 270M flojea en la clase rara y más
  difícil. Es exactamente la señal que la escalera debe revelar; la decisión de
  si este peldaño basta (o hay que subir a Qwen3-0.6B) es la **regla de selección
  de la Fase 5 completa**, que compara peldaños y baselines — fuera del alcance
  de este entrenamiento de un solo peldaño.

## Notas / desviaciones

- No se hizo el sweep de lr sobre el 10%; se usó lr 2e-5 (estándar) que dio buen
  resultado (loss 0.055, 77% exact en test). El sweep queda como mejora si este
  peldaño se elige.
- Modelo redimensionado a 262145 embeddings (token de imagen de Gemma 3) solo
  para la exportación GGUF; no afecta al comportamiento de texto.
