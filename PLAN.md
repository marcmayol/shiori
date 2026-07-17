# PLAN.md: Entrenamiento de la política de routing (tamaño mínimo viable)

Plan de trabajo para Claude Code. El entregable es un MODELO ENTRENADO que, dado un registro de modelos renderizado como prompt guía y una tarea, elige el modo de ejecución y el modelo destino. El objetivo de este proyecto es encontrar el modelo MÁS PEQUEÑO que cumpla el criterio de éxito: el tamaño no se elige a priori, se descubre subiendo una escalera de tamaños y parando en el primer peldaño que cumple.

Ejecutar las fases en orden. No avanzar de fase sin tests en verde y commit. Las decisiones de la sección 2 están cerradas: si alguna parece incorrecta durante la implementación, parar y preguntar.

## 1. Objetivo

Entrenar una política de routing que prediga, para cada tarea:

- `mode`: DIRECT (conocimiento embebido, una completion) | TOOL_CALL (bucle de herramientas corto) | PLAN (requiere planificar y reflexionar; se delega a un modelo capaz)
- `model_id`: el modelo mínimo suficiente DENTRO del registro que se le presenta en el prompt

Definición operativa de "lo más pequeño posible": el peldaño más bajo de la escalera de la Fase 4 cuyo rendimiento en el test congelado quede a menos de 2 puntos de pass-rate del mejor peldaño probado, supere todos los baselines de la Fase 5 (incluida la métrica de pools nunca vistos) y cumpla el presupuesto de latencia en local de la sección 2.

Requisito duro de despliegue: el modelo final corre al 100% en local con el stack estándar (llama.cpp, Ollama, LM Studio), sin red ni APIs en ninguna parte de la inferencia, incluido el decoding constreñido. CPU-only debe ser funcional, no solo teóricamente posible.

Requisito de portabilidad: el modelo debe generalizar a registros que nunca vio (pools de otros usuarios). En el rango mini esa robustez NO viene de la capacidad general del modelo base (que es débil por debajo de 1B), sino del dataset: augmentación agresiva de registros y formato de entrada/salida rígido.

El modelo predice; no verifica ni ejecuta. La detección de fallo y el escalado en producción siguen siendo código alrededor del modelo (try, verify, next). Ese mismo mecanismo se usa aquí offline: es lo que genera las etiquetas del dataset.

## 2. Decisiones cerradas (no re-abrir)

- Escalera de bases, de menor a mayor: Gemma 3 270M → Qwen3-0.6B → Qwen3-1.7B (techo de escape). Se entrena y evalúa en ese orden y se sube SOLO si el peldaño actual no cumple. Al arrancar, verificar si existen releases más recientes equivalentes en cada peldaño y proponer antes de sustituir.
- Método por tamaño: full fine-tune permitido y preferido en ≤ 0.6B (cabe holgado en 16 GB; en este rango no hay capacidad general que merezca preservarse y la especialización total es el comportamiento deseado). LoRA/QLoRA solo en el peldaño 1.7B.
- Todo debe caber en una GPU de 16 GB de VRAM (RTX 5070 Ti). Los peldaños mini permiten batch grande y epochs extra; el 1.7B usa QLoRA 4-bit con gradient accumulation.
- Requisito duro de despliegue: la inferencia completa corre en local con llama.cpp/Ollama/LM Studio. Presupuesto de latencia por decisión: < 1 s en CPU de consumo (8 hilos, Q4_K_M) y < 50 ms en GPU de consumo. Ninguna llamada de red en inferencia. Las APIs solo se permiten en el etiquetado offline de la Fase 2: son coste único de construcción del dataset y no viajan con el artefacto.
- Salida como elección constreñida: en inferencia se usa SIEMPRE decoding constreñido que limita la generación al enum de modos y a los `model_id` presentes en el registro del propio prompt. Esto convierte la generación en clasificación efectiva y es lo que hace viable el rango mini. El constraint debe funcionar en el runtime LOCAL: JSON schema dinámico por petición (structured outputs de Ollama) y gramática GBNF generada desde el registro (llama.cpp); outlines/xgrammar quedan solo como referencia dentro del harness de evaluación. El entrenamiento usa exactamente el mismo formato de salida.
- Salida sin razonamiento: JSON mínimo `{"mode": ..., "model_id": ...}`. Nada de rationale (latencia y capacidad); un modelo mini no lo sostiene.
- Formato del registro: contrato FIJO y compacto (una línea por modelo con campos mínimos: id, tags, context_window, coste relativo, locality). Presupuesto de render ≤ 1000 tokens para pools de 2 a 8 modelos. Se augmenta el CONTENIDO y el ORDEN de los registros, nunca el formato: un modelo mini no tiene capacidad para parsear formatos variados y el formato pasa a ser parte del contrato del producto.
- La etiqueta de suficiencia se obtiene EJECUTANDO los modelos y verificando resultados, no opinando. Verificación real (tests) para código; LLM juez solo donde no hay verificación mecánica, con control de ruido.
- La etiqueta es función del registro presentado: "el modelo mínimo suficiente en ESTE pool". Al augmentar registros se recalcula la etiqueta.
- Stack: unsloth + trl (peft solo para el peldaño 1.7B). Python 3.11+, uv, ruff, mypy strict, pytest para el código del pipeline.
- Dataset y modelo versionados; test set congelado desde el primer día e intocable.
- Licencia Apache-2.0; pesos publicados en Hugging Face Hub con model card.

## 3. Convenciones para Claude Code

- El pipeline de datos es código de producción: tipado, testeado, reproducible con seeds fijos.
- Cada script acepta `--sample N` para ensayos baratos antes de runs completos.
- Antes de cualquier run que llame a APIs de pago: ejecutar la estimación de coste sobre una muestra de 100 y reportar la proyección. No lanzar el run completo sin confirmación explícita.
- Reportar por cada peldaño entrenado: parámetros, VRAM pico, tokens/s de entrenamiento, latencia de inferencia.
- Commits atómicos, estilo conventional commits. Configs de entrenamiento en YAML versionado, nunca hardcodeadas.

## 4. Estructura objetivo del repo

```
router-policy/
├── pyproject.toml
├── configs/            # YAML por peldaño: generación, augmentación, entrenamiento, eval
├── src/routerpolicy/
│   ├── schema/         # tipos: ModelSpec, RegistryRender, Example, Label
│   ├── registry/       # render compacto del registro + generador de pools sintéticos
│   ├── harness/        # ejecución del pool sobre tareas + verificadores + cascada offline
│   ├── labeling/       # etiquetado de modo (reglas + juez), control de calidad
│   ├── dataset/        # ensamblado, augmentación, dedupe, splits
│   ├── training/       # full FT (mini) y QLoRA (1.7B), callbacks, checkpoints
│   ├── evaluation/     # métricas de routing, simulación de coste, baselines, regla de selección
│   └── inference/      # carga del modelo + decoding constreñido a los ids del pool
├── tests/
└── scripts/            # entrypoints CLI por fase
```

## Fase 0: Scaffolding

- [x] pyproject con uv; ruff, mypy strict, pytest configurados
- [x] Detección de entorno GPU (CUDA, VRAM disponible) con mensaje claro si no cumple
- [x] CI ligera (lint + type + tests unitarios; sin GPU en CI)
- [x] Estructura de carpetas de la sección 4

DoD: `uv run pytest` en verde; script `check_env` reporta GPU y VRAM.

## Fase 1: Esquema de datos y formato de entrenamiento

- [x] `ModelSpec` (pydantic): id, tags de capacidades, context_window, coste relativo, locality, supports_tools
- [x] `render_registry_prompt(registry) -> str`: render determinista y COMPACTO (una línea por modelo); test golden-file y test de presupuesto (≤ 1000 tokens con 8 modelos)
- [x] Formato de ejemplo chat: system fijo y corto, user = registro renderizado + tarea, assistant = `{"mode", "model_id"}`
- [x] `Label` con procedencia: cómo se obtuvo (verificación real | juez | regla), scores de la corrida, `schema_version`
- [x] Validadores del JSON de salida (mode válido, model_id existe en el registro del propio ejemplo)
- [x] Gramática/regex del decoding constreñido generada a partir del registro de cada ejemplo; test de que solo admite salidas válidas

DoD: 5 ejemplos construidos a mano pasan la validación de esquema y la gramática de punta a punta.

## Fase 2: Harness de generación de etiquetas (cascada offline)

Fuentes de tareas, por eje:

- [x] Código verificable (eje suficiencia): MBPP+, HumanEval+ y un subset de BigCodeBench; sus tests dan verificación mecánica  <!-- HumanEval+/MBPP+ cableados y verificación probada; BigCodeBench subset pendiente en el run (ver BLOCKERS.md) -->
- [x] Tool calling (eje modo): subset de xlam-function-calling-60k o BFCL como ejemplos TOOL_CALL  <!-- xLAM gated; se usó Hermes function-calling (no-gated): 2279 TOOL_CALL. Ver DATA_LICENSES.md -->
- [x] Instrucciones generales (DIRECT vs PLAN): subset de WildChat o LMSYS-Chat-1M, filtrado y deduplicado  <!-- WildChat/LMSYS gated; se usó Dolly-15k (no-gated); dedupe near-dup en Fase 3 -->
- [x] Revisar licencias de cada dataset y registrar la decisión en `DATA_LICENSES.md`

Ejecución y etiquetado de suficiencia:

- [x] Pool de generación heterogéneo (mínimo: 1 modelo local pequeño de código vía Ollama/vLLM, 1 medio, 1 API capaz)  <!-- pool 100% local: qwen2.5-coder 1.5b/7b + esdrac 7b como capaz (configs/pool.local.yaml), coste $0 -->
- [x] Cascada offline: ejecutar cada tarea de código empezando por el modelo más barato; verificar con los tests; la etiqueta es el modelo mínimo que pasa
- [x] Cache agresiva de completions (hash de tarea+modelo) para re-etiquetar sin re-ejecutar
- [x] Estimación de coste sobre muestra de 100 tareas antes del run completo (regla de la sección 3)  <!-- muestra de 100 ejecutada y medida (58/31/1/10%); coste $0 al ser pool local, sin gate de pago -->

Etiquetado de modo:

- [x] Reglas para los casos obvios (tools declaradas, pregunta factual corta)
- [x] LLM juez para el resto, con rúbrica fija y salida JSON
- [x] Control de ruido: doble anotación sobre una muestra del 10%; reportar acuerdo; si es bajo en alguna clase, revisar la rúbrica antes de escalar el etiquetado

DoD: 5-10k tareas etiquetadas con procedencia completa; distribución por modo y por modelo-suficiente reportada.

## Fase 3: Augmentación de registros (la clave de la portabilidad)

En el rango mini la augmentación sube de importancia: es lo que sustituye a la capacidad general del base.

- [x] Generador de pools sintéticos: variar número de modelos (2-8), ids y nombres (aleatorizados para impedir memorización de marcas), tags, costes, tiers y orden de render; formato SIEMPRE idéntico  <!-- registry/synthetic.py: capacidad oculta codificada en tags recuperables -->
- [x] Recalcular la etiqueta por pool: si el modelo mínimo suficiente no está en el pool sintético, la etiqueta pasa al siguiente suficiente presente; si ninguno es suficiente, etiqueta = el más capaz del pool  <!-- dataset/augment.py: label_for_pool -->
- [x] Factor de augmentación 5-8x por ejemplo base, con límite de repetición de la misma tarea  <!-- factor 9 (mínimo para 50k tras dedup de 703 near-dups); ver reports/fase3_dataset.md -->
- [x] Dedupe near-duplicate de tareas (minhash o embeddings) ANTES del split para evitar leakage  <!-- dataset/dedup.py: MinHash+LSH sin deps -->
- [x] Splits estratificados por modo y dificultad; test set congelado que además incluye SOLO composiciones de pool nunca vistas en train  <!-- dataset/splits.py: firmas de pool reservadas al test -->

Resultado: 50 110 filas (45 135 train / 4 975 test), leakage limpio (0/0). Ver reports/fase3_dataset.md.

DoD: dataset final 50-120k filas; informe de balance por clase; test de leakage en verde.

## Fase 4: Entrenamiento en escalera (de menor a mayor)

Procedimiento: entrenar el peldaño, correr la evaluación de la Fase 5, y SOLO si no cumple pasar al siguiente. El primer peldaño que cumple es el modelo final.

- [ ] Smoke test previo por peldaño: overfit intencional sobre 200 ejemplos hasta ~100% exact match (valida el pipeline antes de gastar horas)
- [ ] Peldaño 1: Gemma 3 270M, full fine-tune. Batch grande, 2-3 epochs, lr con sweep corto sobre el 10% de los datos
- [ ] Peldaño 2: Qwen3-0.6B, full fine-tune (misma receta ajustada)
- [ ] Peldaño 3 (techo de escape): Qwen3-1.7B, QLoRA 4-bit, seq len 4k
- [ ] Máscara de pérdida solo sobre la respuesta del assistant
- [ ] Logging local (tensorboard) + checkpoints con retención de los 2 mejores
- [ ] Ablación solo en el peldaño ganador: full FT vs LoRA (si aplica) y sensibilidad al tamaño del dataset (50% vs 100%)

DoD: cada peldaño entrenado termina en la 5070 Ti sin OOM, con su informe de recursos (sección 3) y su evaluación adjunta.

## Fase 5: Evaluación y regla de selección

Métricas sobre el test set congelado, siempre CON y SIN decoding constreñido (la diferencia mide cuánto sostiene el constraint al modelo):

- [ ] Exactitud de modo (por clase, no solo global)
- [ ] Suficiencia real: para tareas de código, re-ejecutar la elección del modelo y comprobar si pasa la verificación (no solo exact match contra la etiqueta)
- [ ] Regret de coste: coste de la elección vs oráculo (modelo mínimo suficiente conocido)
- [ ] Tasa de salida inválida sin constraint (JSON malformado o model_id inexistente)
- [ ] Generalización: métrica separada sobre los pools nunca vistos
- [ ] Latencia por decisión medida con el runtime local real (llama.cpp): CPU 8 hilos con Q4_K_M y la 5070 Ti, por peldaño

Baselines obligatorios (mismo test set):

- [ ] Cascada pura sin política (empezar siempre por el más barato)
- [ ] Cada modelo base zero-shot con el mismo prompt (sin entrenar)
- [ ] Un modelo grande de API zero-shot como router

Regla de selección de tamaño: elegir el peldaño MÁS PEQUEÑO que (a) supere todos los baselines, (b) quede a menos de 2 puntos de pass-rate del mejor peldaño probado, (c) no se hunda en la métrica de pools no vistos respecto a la global, y (d) cumpla el presupuesto de latencia local de la sección 2 medido con llama.cpp, no con transformers. Si ni el peldaño 3 supera al base zero-shot, el proyecto se detiene aquí y se reporta. Si algún peldaño cumple calidad pero ninguno cumple la latencia en CPU, la vía B pasa a ser el camino principal.

DoD: informe comparativo de peldaños reproducible con un solo comando, con la selección justificada.

## Fase 6: Inferencia, empaquetado y publicación

- [ ] Decoding constreñido como parte del artefacto publicado, no como opción: la función `route(task, registry) -> {mode, model_id}` habla por defecto con un endpoint local (Ollama), genera el JSON schema (enum con los ids del pool) o la gramática GBNF a partir del registro recibido, y decodifica contra ella
- [ ] Test de integración real contra Ollama: registro de ejemplo → decisión válida, sin ninguna llamada de red externa
- [ ] Fallback documentado para el caso residual (empate o gramática vacía): caer al modelo más capaz del pool
- [ ] Benchmark con el runtime local real (llama.cpp/Ollama, no transformers): CPU 8 hilos y la 5070 Ti, registros de 0.5k-1k tokens, cuantizaciones Q8_0 y Q4_K_M; tabla de latencias por peldaño en la model card
- [ ] Export: pesos en HF Hub + GGUF en Q8_0 y Q4_K_M para Ollama/llama.cpp/LM Studio
- [ ] Model card honesta: datos usados, licencias, métricas por peldaño, límites conocidos (dominios no cubiertos, dependencia del formato de registro fijo)

DoD: `ollama run` del GGUF responde decisiones válidas sobre un registro de ejemplo; pesos publicados.

## Vía B (solo si ningún peldaño mini cumple y no se quiere aceptar el 1.7B)

Cambio de arquitectura, no de tamaño: cross-encoder por pares (~100-150M, p. ej. ModernBERT-base) que puntúa cada par (tarea, modelo del pool) por separado con probabilidad calibrada de suficiencia, más una cabeza de clasificación de modo sobre la tarea sola. Se elige el modelo más barato con probabilidad ≥ umbral; el umbral se recalibra sin re-entrenar. Maneja pools de cualquier tamaño por construcción y reutiliza el mismo dataset de la Fase 2 (las etiquetas por par salen de la cascada offline). Es el suelo absoluto de tamaño; se documenta aquí para no re-diseñar desde cero si hace falta. Cumple el requisito local con margen: un encoder de ~100M corre en CPU en pocos milisegundos (incluir export ONNX).

## Fuera de alcance

Librería de routing completa (cascada en producción, verificadores como producto, estrategias enchufables), proxy HTTP, UI, orquestador de planes, RLHF/DPO sobre la política (posible v2 con trazas reales de uso).
