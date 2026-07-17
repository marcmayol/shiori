# Licencias de las fuentes de datos

Revisión de licencias de cada dataset usado para **etiquetar** (Fase 2). El
artefacto publicado es el MODELO, no los datasets; aun así se documenta el
origen y la licencia de cada fuente, y su compatibilidad con el uso previsto
(construcción offline del dataset de entrenamiento de una política de routing).

> Nota: las licencias deben re-verificarse en la ficha de cada dataset en el
> momento de la descarga (pueden cambiar). Esta tabla refleja la revisión de la
> fase y debe actualizarse si al descargar difiere.

| Eje | Dataset | Fuente (HF) | Licencia declarada | Uso aquí | Notas |
|-----|---------|-------------|--------------------|----------|-------|
| Suficiencia (código) | HumanEval / HumanEval+ | `openai/openai_humaneval`, `evalplus/humanevalplus` | MIT | Ejecución de tests para etiqueta de suficiencia | HumanEval original MIT; evalplus añade tests (Apache-2.0) |
| Suficiencia (código) | MBPP / MBPP+ | `google-research-datasets/mbpp`, `evalplus/mbppplus` | CC-BY-4.0 (MBPP) | Ejecución de tests para etiqueta de suficiencia | Atribución requerida; evalplus tests Apache-2.0 |
| Suficiencia (código) | BigCodeBench (subset) | `bigcode/bigcodebench` | Apache-2.0 | Subset para verificación mecánica | Revisar términos de dependencias que ejecuta |
| Modo (TOOL_CALL) | **Hermes function-calling v1** ⭐ EN USO | `NousResearch/hermes-function-calling-v1` | Apache-2.0 | Enunciado (turno human) + nombres de tools para etiqueta de MODO | **No-gated**; sustituye a xLAM |
| Modo (DIRECT/PLAN) | **Dolly-15k** ⭐ EN USO | `databricks/databricks-dolly-15k` | CC-BY-SA-3.0 | `instruction` como enunciado para etiqueta de MODO | **No-gated**; sustituye a WildChat/LMSYS |
| Modo (TOOL_CALL) | xLAM function-calling-60k | `Salesforce/xlam-function-calling-60k` | CC-BY-4.0 (con cláusulas) | (alternativa gated, no usada) | Gated: requiere solicitar acceso |
| Modo (DIRECT/PLAN) | WildChat-1M | `allenai/WildChat-1M` | ODC-BY + AI2 ImpACT | (alternativa gated, no usada) | Gated |
| Modo (DIRECT/PLAN) | LMSYS-Chat-1M | `lmsys/lmsys-chat-1m` | LMSYS-Chat-1M License | (alternativa gated, no usada) | Gated |

## Decisiones registradas

- **Sustitución de fuentes gated por no-gated (eje de modo).** El plan listaba
  xLAM/BFCL (tool) y WildChat/LMSYS (chat), todas gated en HF. Para no bloquear
  el run en aprobaciones manuales, se sustituyen por **Hermes function-calling
  v1** (Apache-2.0) y **Dolly-15k** (CC-BY-SA-3.0), ambas no-gated y de calidad
  equivalente para el eje de modo. Reversible: los ingest de las gated siguen en
  el código por si se obtiene acceso.
- **No se redistribuyen los datasets.** Solo se descargan localmente para
  etiquetar; el repositorio versiona el CÓDIGO de ingesta, no los datos.
- **Solo se extraen los campos mínimos** necesarios por eje: para código, el
  enunciado + tests; para modo, el enunciado (+ nombres de tools). No se
  reproducen completions ajenas en el artefacto.
- **Atribución:** MBPP (CC-BY-4.0) y las fuentes CC-BY se atribuyen en la model
  card final.
- **Datasets gated** (xLAM, WildChat, LMSYS): requieren aceptar términos en HF
  antes de la descarga. Se aceptarán en el momento del run de etiquetado.
- **El modelo publicado es Apache-2.0**; no incorpora los datos, solo el
  comportamiento aprendido de las etiquetas derivadas de la verificación.
