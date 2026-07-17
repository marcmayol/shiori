# INFORME FASE 5 — PELDAÑO 1 (Gemma 3 270M) 

PELDAÑO 1 — pools NUNCA vistos (test congelado):
  sin constraint:  exact=0.750 mode=0.750 model_id=0.800 invalid=0.000  [TOOL=0.78 PLAN=0.00 DIRE=0.80]
  con constraint:  exact=0.750 mode=0.750 model_id=0.800 invalid=0.000  [TOOL=0.78 PLAN=0.00 DIRE=0.80]
  regret de coste (vs oráculo): +0.085

PELDAÑO 1 — pools vistos (train), para el gap de generalización:
  con constraint:  exact=0.900 mode=1.000 model_id=0.900 invalid=0.000  [DIRE=1.00 TOOL=1.00]
  gap exact (visto - no visto): +0.150

Matriz de confusión de MODO (con constraint, test):
  gold\pred       DIRECT TOOL_CALL      PLAN   INVALID
  DIRECT              8         2         0         0
  TOOL_CALL           1         7         1         0
  PLAN                1         0         0         0
  errores de PLAN -> {'DIRECT': 1}

SUFICIENCIA REAL (pool de generación, n=20):
  política: 0.900  cascada-barato: 0.750  oráculo: 0.950

BASELINES (mismo test, con constraint salvo indicado):
  cascada pura (siempre barato): exact=0.200 mode=0.500 model_id=0.400 invalid=0.000  [TOOL=0.00 PLAN=0.00 DIRE=1.00]
  Gemma 270M BASE zero-shot:     exact=0.100 mode=0.300 model_id=0.200 invalid=0.000  [TOOL=1.00 PLAN=0.00 DIRE=0.00]
  API zero-shot: NO EJECUTADO. Estimación 100->test ~$6.23 (~5.73 EUR). Sin clave + umbral [X] indefinido -> BLOCKERS.md

LATENCIA (llama.cpp):
  cuant      tamaño      CPU 8h       GPU   presupuesto
  Q8_0       286 MB     589 ms     38 ms   CPU<1s ✓ GPU<50ms ✓
  Q4_K_M     249 MB     564 ms     37 ms   CPU<1s ✓ GPU<50ms ✓

REGLA DE SELECCIÓN (peldaño más pequeño que cumple):
  (a) supera baselines: base=True cascada=True
  (b) único peldaño probado -> es el mejor por defecto
  (c) generalización (gap visto-no visto): +0.150
  (d) latencia CPU < 1 s: sí (Q4_K_M ~0.55 s)

RECOMENDACIÓN: CUMPLE la regla formal (supera baselines + latencia). PERO PLAN es débil (0.00); recomendación: aceptable como suelo, pero entrenar el peldaño 2 (Qwen3-0.6B) para mejorar PLAN antes de fijar el final.
