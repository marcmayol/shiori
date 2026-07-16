# Shiori (栞)

Política de routing de modelos **más pequeña posible**. Dado un registro de
modelos renderizado en el prompt (formato fijo y compacto) y una tarea, Shiori
predice:

```json
{"mode": "DIRECT | TOOL_CALL | PLAN", "model_id": "<el mínimo suficiente del pool>"}
```

El nombre (栞, *shiori*) significa marcador de libros / guía: el modelo marca qué
modelo usar para cada tarea.

## Idea central

El tamaño del modelo **se descubre, no se elige**: se sube una escalera de bases
(Gemma 3 270M → Qwen3-0.6B → Qwen3-1.7B) y se para en el primer peldaño que
cumple el criterio de selección. La inferencia corre **100% en local**
(llama.cpp / Ollama / LM Studio) con **decoding constreñido** al enum de modos y
a los ids del pool presente en el prompt. Las APIs solo se usan en el etiquetado
offline de construcción del dataset; no viajan con el artefacto.

Fuente de verdad del proyecto: [`PLAN.md`](./PLAN.md).

## Desarrollo

```bash
uv sync                              # instala deps + grupo dev
uv run check-env                     # reporta GPU/VRAM (código 0 si cumple)
uv run ruff check .                  # lint
uv run mypy                          # tipos (strict)
uv run pytest                        # tests
```

Requiere Python ≥ 3.11 y (para entrenar) una GPU de 16 GB. Licencia Apache-2.0.
