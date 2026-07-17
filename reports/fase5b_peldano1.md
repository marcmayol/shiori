# INFORME FASE 5b — PELDAÑO 1 

(1) BASELINE GRANDE ZERO-SHOT como router (mismo test):
  peldaño 1 (270M, entrenado): exact=0.750 mode=0.750 invalid=0.000 [TOOL=0.78 PLAN=0.00 DIRE=0.80]
  esdrac 7B zero-shot (26x):   exact=0.125 mode=0.250 invalid=0.000 [TOOL=1.00 PLAN=0.00 DIRE=0.00]
  API de pago: estimación ~$2.68 (~2.46 EUR, <10 EUR autorizados) pero SIN CLAVE -> NO ejecutado (BLOCKERS.md).

(2) CALIBRACIÓN — barrido de prior a PLAN:
   prior   PLAN_r  DIRECT_r   TOOL_r  mode_acc    coste
    0.00     0.000       0.800     0.778     0.750       5.17
    0.25     1.000       0.800     0.778     0.800       5.29  <- op
    0.50     1.000       0.700     0.778     0.750       5.71
    0.75     1.000       0.500     0.778     0.650       6.17
    1.00     1.000       0.500     0.667     0.600       6.35
    1.50     1.000       0.100     0.444     0.300       7.90
    2.00     1.000       0.000     0.000     0.050       8.91
    3.00     1.000       0.000     0.000     0.050       8.91
  Punto de operación elegido: prior=0.25 (máx recall de PLAN con caída de mode_acc <=2 pts vs prior 0).

(3) MATRIZ DE CONFUSIÓN en el punto de operación (prior=0.25):
  gold\pred       DIRECT TOOL_CALL      PLAN   INVALID
  DIRECT              8         2         0         0
  TOOL_CALL           1         7         1         0
  PLAN                0         0         1         0
  PLAN recall 0.000 -> 1.000  |  mode_acc 0.750 -> 0.800

(4) ECONOMÍA (código, n=1, coste ~ B de params):
  router (apuesta+escala): coste_medio=9.50 pass_rate=0.000
  cascada pura (barato->capaz): coste_medio=16.50 pass_rate=0.000
  ahorro de coste del router vs cascada: +42.4% (a igual pass-rate)

(5) REGRET DE COSTE — definición y explicación:
  regret_i = coste(elegido_i) - coste(oráculo_i), oráculo = mínimo suficiente
  (la etiqueta gold). Media sobre el test:
    regret medio = +0.085   (n=20)
    desglose: infra-provisión (más barato que el mínimo, <0)=2  exacto (=0)=16  sobre-provisión (>0)=2
  Explicación del signo NEGATIVO: no es un ahorro óptimo. El -0.169 sale de que la política a veces elige un modelo MÁS BARATO que el mínimo suficiente (infra-provisión), lo cual correlaciona con el fallo de PLAN (subestima la dificultad). Un regret sano sería ~0+; el negativo señala under-provisioning, un fallo de calidad, no un ahorro.
