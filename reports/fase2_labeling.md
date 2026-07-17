# Informe Fase 2 — Etiquetado (cascada offline)

Resultado del run de etiquetado con el pool **100% local** (coste $0). Números
reproducibles con `scripts/run_labeling.py` (reanudable) sobre el mismo pool.

## Pool de generación (real, ejecutado offline)

Ordenado de más barato a más capaz (`configs/pool.local.yaml`):

1. `qwen2.5-coder:1.5b` — pequeño de código
2. `qwen2.5-coder:7b` — medio
3. `esdrac:latest` — capaz (Qwen 7B propio); también actúa de juez de modo

## Dataset etiquetado: 6274 tareas

| Eje | Tareas |
|-----|-------:|
| Código (suficiencia, verificación mecánica) | 542 |
| Modo (TOOL_CALL / DIRECT / PLAN) | 5732 |
| **Total** | **6274** |

### Suficiencia (código) — modelo mínimo que pasa los tests

| Modelo mínimo suficiente | Tareas | % |
|--------------------------|-------:|--:|
| `qwen2.5-coder:1.5b` | 316 | 58% |
| `qwen2.5-coder:7b` | 143 | 26% |
| `esdrac:latest` | 18 | 3% |
| ninguno suficiente | 65 | 12% |

Fuentes: HumanEval+ y MBPP+ (evalplus), verificación por ejecución real.

### Modo

| Modo | Tareas |
|------|-------:|
| TOOL_CALL | 2279 |
| DIRECT | 2854 |
| PLAN | 599 |

| Procedencia | Tareas |
|-------------|-------:|
| regla (obvios) | 3766 |
| juez (LLM) | 1966 |

Fuentes: Hermes function-calling (TOOL_CALL) y Dolly-15k (DIRECT/PLAN), ambas
no-gated (sustituyen a xLAM/WildChat gated; ver `DATA_LICENSES.md`).

## Control de ruido (doble anotación del juez)

Juez anotado dos veces (temp 0 vs 0.8) sobre 200 tareas de chat (99 al juez):

| Métrica | Valor |
|---------|------:|
| n anotadas | 93 |
| Acuerdo bruto | 0.903 |
| **κ de Cohen** | **0.850** |
| por clase | DIRECT 0.925 · TOOL_CALL 0.826 · PLAN 0.933 |

κ ≈ 0.85 (acuerdo casi perfecto): la rúbrica es fiable, no requiere revisión
antes de escalar.

## Notas

- **52-80 tareas de clasificación de Dolly** hacen que el juez devuelva la
  categorización en vez del modo y se saltan (~4,5%); no se etiquetan.
- **BigCodeBench**: pendiente (gap menor del eje de código).
- Coste monetario: **$0** (pool local); el gate de coste de pago no aplicó.
