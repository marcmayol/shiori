# BLOCKERS

## Fase 2 — el run de etiquetado está bloqueado por decisiones del usuario

Todo el **harness de etiquetado está construido y testeado** (93 tests en verde:
cascada, verificación, cache, coste, reglas, juez, acuerdo, reanudación,
ingesta). Lo que falta para cerrar la DoD de la Fase 2 (*5-10k tareas
etiquetadas + distribución reportada*) es **ejecutar el run**, y eso está
bloqueado por dos cosas que dependen de ti y que además son la **parada
obligatoria de la sección 3** (estimación de coste + confirmación explícita
antes de cualquier run de pago).

### Estado del entorno (verificado)

- **Ollama** 0.23.2 instalado, pero el único modelo local es `esdrac:latest`
  (tu Qwen 7B). Falta un modelo **pequeño de código** y uno **medio** para los
  tiers baratos de la cascada.
- **Sin ninguna clave de API** (Anthropic/OpenAI/OpenRouter/Gemini ausentes) →
  el tier de API capaz no puede ejecutarse.
- Libs de ingesta (`datasets`, `evalplus`) no instaladas (extra `label`, se
  instalan para el run).

### Decisiones necesarias para desbloquear

1. **Tier de API capaz** (no fijado en la sección 2): proveedor + modelo +
   **clave**. El adaptador `OpenAICompatRunner` ya sirve OpenAI/OpenRouter/
   compat; solo falta elegir y aportar la clave (por variable de entorno).
2. **Tiers locales**: qué modelos bajar con `ollama pull` para el pequeño de
   código y el medio (p. ej. `qwen2.5-coder:1.5b` y `qwen2.5-coder:7b`, o los
   que prefieras).
3. **Presupuesto**: confirmar el gasto tras ver la estimación real sobre 100
   tareas (con datos descargados y pool cableado). La proyección con precios
   placeholder da **$4-20** para 8k tareas (`scripts/estimate_cost.py`), pero
   el número real depende del modelo de API elegido y de la tasa de escalado
   medida.

### Gaps menores a cerrar durante el run (no bloqueantes)

- **BigCodeBench**: falta el mapeador/ingest del subset (su modelo de ejecución
  es más pesado); se valida el esquema al descargar. HumanEval+ y MBPP+ ya
  están cableados y con verificación probada.
- **Datasets gated** (xLAM, WildChat, LMSYS): requieren aceptar términos en HF
  antes de la descarga.
- El **esquema real** de los ingest_* (evalplus/xLAM/WildChat) se valida en la
  primera descarga; los mapeadores puros están testeados con fixtures.

### Cómo se procede al desbloquear

1. `ollama pull` de los dos modelos locales elegidos.
2. Exportar la clave de API del proveedor elegido.
3. `uv sync --extra label` e ingesta de una muestra.
4. Estimación de coste **real sobre 100 tareas** → reporte → **tu confirmación
   explícita**.
5. Run completo con checkpointing incremental (reanudable) → distribución →
   cierre de la DoD.
