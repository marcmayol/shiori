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
  - es
---

# Shiori 270M — model-routing policy

A 270M-parameter model that maps a task and a set of available models (a
*registry*) to a routing decision. Full fine-tune of `google/gemma-3-270m`.
Input and output are a fixed contract; the output is a single JSON object:

```json
{"mode": "DIRECT|TOOL_CALL|PLAN", "model_id": "<one id from the registry>"}
```

- `mode` — execution mode: `DIRECT` (single completion), `TOOL_CALL` (tool loop),
  `PLAN` (multi-step planning).
- `model_id` — the minimal-cost model in the registry whose capability is
  sufficient for the task.

Inference is designed for local runtimes (llama.cpp, Ollama, LM Studio) with
constrained decoding restricting the output to valid modes and to the ids present
in the registry. Source and training pipeline:
https://github.com/marcmayol/shiori

---

## Model details

- Architecture: Gemma 3 (decoder-only), 268.1M parameters, vocab 262k.
- Method: full fine-tune, bf16.
- Objective: cross-entropy on the assistant tokens only (prompt masked).
- Hyperparameters: 3 epochs, lr 2e-5 (cosine, warmup 0.03), effective batch 16,
  max sequence length 768.
- Chat format (the model has no default template; this one is used for train and
  inference, with `system` merged into the user turn):

  ```
  <start_of_turn>user
  {SYSTEM}

  {registry}

  Task:
  {task}<end_of_turn>
  <start_of_turn>model
  {"mode": "...", "model_id": "..."}<end_of_turn>
  ```

## Registry format (fixed contract)

One line per model. The model reads tags, cost, context window and tool support.

```
Available models:
- fast-local | tags:code,general | ctx:8192 | cost:1 | loc:local | tools:no
- mid-local  | tags:code,reasoning | ctx:32768 | cost:3 | loc:local | tools:yes
- big-api    | tags:code,reasoning,planning | ctx:200000 | cost:20 | loc:api | tools:yes

Task:
<the task>
```

Capability is encoded in the tags: `reasoning` ⟺ capability ≥ 2, `planning` ⟺ ≥ 3,
`expert` ⟺ 4. Registries are not part of the model weights; they are supplied at
inference time.

## Usage

Ollama:

```bash
# from this repo: shiori-270m-Q4_K_M.gguf + Modelfile
ollama create shiori-router -f Modelfile
```

Call `/api/generate` with `prompt = SYSTEM + registry + task` and a JSON-schema
`format` built from the registry (enum of modes, enum of ids) so the output is
always valid.

Transformers:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained("natzx94/shiori-router-270m")
model = AutoModelForCausalLM.from_pretrained("natzx94/shiori-router-270m")
# build the chat format above, generate, stop at <end_of_turn>, parse the JSON.
```

The GitHub repo provides `route(task, registry, plan_prior=0.5)` which talks to a
local Ollama endpoint, generates the JSON schema / GBNF grammar from the registry,
applies the calibration prior, and falls back to the most capable model in the pool
on invalid output.

## Training data

- Sufficiency labels (code axis): HumanEval+ and MBPP+ (evalplus). The label is the
  lowest-cost model that passes the task's tests, obtained by executing a pool of
  models and verifying their output.
- Mode labels: Hermes function-calling v1 (Apache-2.0) for `TOOL_CALL`; Dolly-15k
  (CC-BY-SA-3.0) for `DIRECT`/`PLAN`, via rules plus an LLM judge.

Task registries are synthetic: model names, tags, costs and pool sizes (2–8) are
randomized per example, and the label is recomputed for each pool. The final
dataset is ~50k rows; the test split contains only pool compositions absent from
training. Datasets are not redistributed. Model weights: Apache-2.0.

## Evaluation

Protocol: held-out test set, n = 400 (fixed seed), model pools with structural
compositions not present in training. Constrained decoding via per-candidate
log-probability scoring. "Sufficiency" is defined as chosen-model capability ≥ task
difficulty, where difficulty is the rank of the minimal sufficient model from the
label cascade.

Metrics at the default prior (constrained):

| metric | value |
|--------|------:|
| exact match ({mode, model_id}) | 0.760 |
| mode accuracy | 0.853 |
| model_id accuracy | 0.802 |
| invalid-output rate (constrained / free) | 0.000 / 0.013 |
| per-class mode accuracy (DIRECT / TOOL_CALL / PLAN) | 0.94 / 0.79 / 0.59 |

Generalization gap: exact match 0.930 on training pool compositions vs 0.760 on
unseen ones.

`plan_prior` adds a constant to the `PLAN` log-probability before the argmax over
modes (inference-time, no retraining). Measured sweep:

| plan_prior | PLAN recall | DIRECT recall | TOOL recall | mode acc | mean cost |
|-----------:|------------:|--------------:|------------:|---------:|----------:|
| 0.00 | 0.592 | 0.936 | 0.795 | 0.853 | 5.73 |
| 0.25 | 0.837 | 0.906 | 0.795 | 0.865 | 5.97 |
| 0.50 | 0.939 | 0.842 | 0.786 | 0.838 | 6.17 |

It trades `PLAN` recall against `DIRECT`/`TOOL_CALL` recall and mean cost.

Baselines on the same test set (zero-shot, constrained):

| system | exact | mode acc | PLAN acc |
|--------|------:|---------:|---------:|
| Shiori 270M (fine-tuned) | 0.760 | 0.853 | 0.59 (0.94 at prior 0.50) |
| GPT-4o zero-shot | 0.228 | 0.718 | 0.00 |
| Qwen2.5-7B zero-shot | 0.153 | 0.367 | 0.27 |
| always-cheapest heuristic | 0.212 | 0.585 | 0.00 |

The zero-shot models score lower on `model_id` selection and on `PLAN`; the task
depends on the minimal-sufficient labeling and the fixed I/O contract rather than
general capability.

Cost regret (chosen-model cost − oracle/minimal-sufficient cost, mean) = −0.169.
The negative value reflects under-provisioning (choosing a cheaper-than-sufficient
model), correlated with `PLAN` errors; it is not a cost saving. A `capability_prior`
parameter shifts the model_id choice toward more capable models and reduces
under-provisioning at higher cost.

Economic simulation (code tasks; router = first bet + escalate to the capable model
on failure; cost proportional to model size):

| system | mean cost | pass rate |
|--------|----------:|----------:|
| router | 6.05 | 0.880 |
| cheapest-first cascade | 5.61 | 0.880 |

On verifiable code, the cheapest-first cascade is ~8% cheaper at equal pass rate.
Routing is advantageous for mode selection and for domains without a mechanical
verifier, not for cost on code.

## Latency

Measured with `llama-bench` (registry ≈ 320 tokens, output 24 tokens):

| quant | size | CPU (8 threads) | GPU |
|-------|-----:|----------------:|----:|
| Q8_0 | 286 MB | 512 ms | 35 ms |
| Q4_K_M | 249 MB | 526 ms | 35 ms |

## Limitations

- `PLAN` recall is 0.59 at the default prior; the `plan_prior` parameter raises it
  (0.84 at 0.25, 0.94 at 0.50) at inference time.
- The model requires the fixed registry format; it does not handle other formats.
- Covered domains: code, tool-calling, general instruction-following (English).
- The model predicts a route; it does not execute or verify. Real sufficiency
  depends on the surrounding system.
- 270M is the smallest step of a size ladder; a larger model may raise the `PLAN`
  ceiling.

---

## Español

Modelo de 270M que, dada una tarea y un conjunto de modelos disponibles (un
*registro*), predice una decisión de routing `{"mode", "model_id"}`. Fine-tune
completo de `google/gemma-3-270m`. Inferencia local (llama.cpp/Ollama/LM Studio)
con decoding constreñido a los modos y a los ids del registro.

- `mode`: `DIRECT` (una completion), `TOOL_CALL` (bucle de herramientas), `PLAN`
  (planificación multi-paso).
- `model_id`: el modelo de menor coste del registro cuya capacidad basta.

**Detalles**: 268.1M parámetros, full fine-tune bf16, pérdida solo sobre la
respuesta del assistant, 3 epochs, lr 2e-5, batch 16, seq 768.

**Datos**: suficiencia de HumanEval+/MBPP+ (etiqueta = modelo mínimo que pasa los
tests, por ejecución real); modo de Hermes function-calling (Apache-2.0) y Dolly-15k
(CC-BY-SA-3.0). Registros sintéticos con nombres/costes/tamaños aleatorizados;
~50k filas; el test solo contiene composiciones de pool ausentes en train. Modelo
Apache-2.0.

**Evaluación** (test congelado, n=400, semilla fija, pools no vistos; decoding
constreñido):

Por defecto: exact 0.760, acc. modo 0.853, acc. model_id 0.802, inválidos 0.000
(0.013 sin constraint), por clase DIRECT 0.94 / TOOL_CALL 0.79 / PLAN 0.59.

Barrido de `plan_prior` (recall de PLAN / DIRECT / TOOL, acc. modo, coste):

| plan_prior | PLAN | DIRECT | TOOL | acc. modo | coste |
|-----------:|-----:|-------:|-----:|----------:|------:|
| 0.00 | 0.592 | 0.936 | 0.795 | 0.853 | 5.73 |
| 0.25 | 0.837 | 0.906 | 0.795 | 0.865 | 5.97 |
| 0.50 | 0.939 | 0.842 | 0.786 | 0.838 | 6.17 |

`plan_prior` suma una constante al log-prob de `PLAN` antes del argmax (parámetro
de inferencia, sin reentrenar).

Baselines zero-shot (mismo test): GPT-4o 0.228 exact / PLAN 0.00; Qwen2.5-7B 0.153;
cascada "siempre el más barato" 0.212. El coste-regret medio es −0.169 (infra-
provisión, no ahorro). En código verificable, la cascada pura es ~8% más barata a
igual pass-rate; el router aporta en la decisión de modo y en dominios sin
verificador.

**Latencia** (llama.cpp): Q8_0 512 ms CPU / 35 ms GPU; Q4_K_M 526 ms / 35 ms.

**Límites**: recall de `PLAN` 0.59 por defecto (subible con `plan_prior`);
dependencia del formato de registro fijo; dominios code/tool/chat en inglés;
predice, no verifica; es el peldaño más pequeño de una escalera.

---

```
@software{shiori_router_2026,
  title  = {Shiori: an on-device model-routing policy},
  author = {Mayol, Marc},
  year   = {2026},
  url    = {https://github.com/marcmayol/shiori}
}
```
