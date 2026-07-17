# BLOCKERS

## Fase 5 — baseline de API zero-shot NO ejecutado

El goal permitía una única llamada a APIs de pago (el baseline de API zero-shot
como router), con estimación de coste previa y un umbral `[X]` euros que quedó
**indefinido** en el goal. Además, **no hay clave de API** configurada (constatado
en Fase 2: ANTHROPIC/OPENAI/... ausentes).

- **Estimación de coste** (proyección desde 100 tareas del test al test completo,
  precios ilustrativos 3/15 USD/Mtok): **~$7.21 (~6.63 EUR)**.
- Decisión: **no se lanza** (sin clave + umbral `[X]` indefinido). Per el goal,
  la Fase 5 se completa con los otros dos baselines (cascada pura y Gemma 270M
  base zero-shot) más esta estimación. El peldaño 1 ya supera esos dos con holgura
  (0.76 vs 0.21 y 0.075 de exact match), así que el baseline de API no cambia la
  conclusión de la regla de selección.
- Para ejecutarlo en el futuro: definir el umbral, configurar la clave del
  proveedor (el adaptador `OpenAICompatRunner` ya existe) y re-lanzar el informe.

## (RESUELTO) Fase 2 — run de etiquetado

El bloqueo del run de etiquetado quedó **resuelto** con la decisión de usar un
**pool 100% local y gratis** (sin claves de API ni presupuesto):

- Pool: `qwen2.5-coder:1.5b` (barato) → `qwen2.5-coder:7b` (medio) →
  `esdrac:latest` (capaz). Ver `configs/pool.local.yaml`. Coste $0.
- Fuentes de modo gated (xLAM/WildChat/LMSYS) sustituidas por **no-gated**:
  Hermes function-calling (Apache-2.0) y Dolly-15k (CC-BY-SA-3.0). Ver
  `DATA_LICENSES.md`. Los ingest de las gated se conservan por si se obtiene
  acceso más adelante.

Como el pool es local, **no hubo run de pago** y el gate de coste de la sección 3
no aplicó (la muestra de 100 se ejecutó y midió con coste $0).

### Gaps menores que siguen abiertos (no bloqueantes)

- **BigCodeBench**: sin mapeador/ingest del subset (ejecución más pesada). El eje
  de código queda cubierto por HumanEval+ y MBPP+ con verificación probada.
- **Tier de API**: si en el futuro se quieren etiquetas de suficiencia más
  fuertes, se añade un tier `openai_compat` al pool y se re-etiqueta solo lo
  bloqueado (la cache lo hace barato). El adaptador ya existe.
- **Juez y clasificación**: ~4,5% de tareas de clasificación de Dolly hacen que
  esdrac devuelva la categorización en vez del modo; se saltan (no se etiquetan).
  Endurecer la rúbrica del juez es mejora futura.
