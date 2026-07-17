# Informe Fase 3 — Dataset final (augmentación de registros)

Dataset de entrenamiento ensamblado con `scripts/build_dataset.py` (determinista,
semilla 20260717). Regenerable desde los labels versionados de la Fase 2.

## Pipeline

labels (6274) + prompts → BaseTask (join) → dedup near-dup → split estratificado
(por modo × dificultad) → augmentación por pool sintético → train/test + leakage.

## Números

| Etapa | Valor |
|-------|------:|
| Tareas base con prompt | 6274 |
| Dedup near-dup (threshold 0.8) | 5571 mantenidas / 703 eliminadas |
| Split (test_frac 0.1) | 5015 train / 556 test |
| Factor de augmentación | 9 |
| **Filas train** | **45 135** |
| **Filas test (pools no vistos)** | **4 975** |
| **TOTAL** | **50 110** |

> Nota de factor: el plan sugiere 5-8x; con 5571 tareas base tras el dedup, el
> factor mínimo para alcanzar el mínimo de la DoD (50k filas) es 9. Desviación
> mínima y justificada; trivialmente reversible (parámetro `--factor`).

## Balance (train)

- **modo**: DIRECT 27315 · TOOL_CALL 12996 · PLAN 4824
- **dificultad**: 1:38475 · 2:1161 · 3:4968 · 4:531
- **nº de modelos por pool (2-8)**: repartido ~uniforme (6157–6769 por tamaño)

## Balance (test — composiciones de pool nunca vistas en train)

- **modo**: DIRECT 3009 · TOOL_CALL 1428 · PLAN 538
- **dificultad**: 1:4241 · 2:126 · 3:554 · 4:54

## Leakage (verde)

- solape de task_id entre train y test: **0**
- solape de firmas de pool entre train y test: **0** (el test usa firmas
  estructurales reservadas por hash, nunca presentes en train)
- `tests/test_dataset_leakage.py` verde sobre el dataset real.

## Test set congelado

- `data/dataset/test.jsonl`, 4975 filas.
- sha256 (16 hex): `270070d2c41258e1` — pin de reproducibilidad; el test set es
  intocable y se regenera idéntico desde (labels + semilla + parámetros).

## Notas

- La dificultad 1 domina (tareas DIRECT/TOOL_CALL sencillas + código fácil); es
  esperable y el router debe aprender a NO sobre-escalar. La estratificación
  garantiza que las clases minoritarias (dificultad 4, PLAN) están en ambos
  splits.
- Formato de cada fila: `{task_id, source, split, mode, model_id, difficulty,
  n_models, messages}`; `messages` es el chat de Fase 1 (system/user/assistant).
