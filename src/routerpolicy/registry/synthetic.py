"""Generador de pools sintéticos (Fase 3): la clave de la portabilidad.

En el rango mini la robustez a registros nunca vistos viene del DATASET, no del
base. Se generan pools variando nº de modelos (2-8), ids/nombres (aleatorios,
sin marcas reales), tags, costes, capacidad y orden. El FORMATO de render es
siempre idéntico (contrato de Fase 1).

La capacidad de cada modelo es un entero oculto 0..4 que NO se renderiza; se
codifica de forma recuperable en los tags observables (reasoning/planning/
expert) y correlaciona con coste/ctx. El router debe inferir la capacidad de
esos campos para elegir el mínimo suficiente.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from routerpolicy.schema.core import Locality, ModelSpec

MAX_CAPABILITY = 4  # niveles 0..4
MIN_POOL = 2
MAX_POOL = 8

# Nombres falsos (adjetivo-sustantivo-nº): sin marcas reales, para impedir
# memorización de proveedores.
_ADJECTIVES = (
    "aster",
    "nimbus",
    "vortex",
    "lumen",
    "quartz",
    "ember",
    "cobalt",
    "onyx",
    "flint",
    "zephyr",
    "cirrus",
    "halcyon",
    "umbra",
    "solace",
    "verdant",
    "pyro",
)
_NOUNS = (
    "spark",
    "forge",
    "quill",
    "atlas",
    "delta",
    "prism",
    "cortex",
    "relay",
    "beacon",
    "kernel",
    "cipher",
    "lattice",
    "vertex",
    "shard",
    "monolith",
    "echo",
)


@dataclass(frozen=True)
class SyntheticModel:
    """Modelo sintético: la spec renderizable + su capacidad oculta."""

    spec: ModelSpec
    capability: int  # 0..MAX_CAPABILITY (no se renderiza)


def _skill_tags(capability: int, rng: random.Random) -> tuple[str, ...]:
    """Tags que codifican la capacidad de forma recuperable (orden aleatorio).

    reasoning ⟺ cap>=2, planning ⟺ cap>=3, expert ⟺ cap==4. Los tags clave no
    se omiten (si no, la capacidad sería inobservable).
    """
    tags = ["code" if capability >= 1 else "general"]
    if capability >= 2:
        tags.append("reasoning")
    if capability >= 3:
        tags.append("planning")
    if capability >= 4:
        tags.append("expert")
    # ruido no informativo: a veces añade "general" (no rompe la inferencia)
    if capability >= 1 and rng.random() < 0.35:
        tags.append("general")
    rng.shuffle(tags)
    return tuple(tags)


def _fake_id(rng: random.Random, used: set[str]) -> str:
    for _ in range(100):
        candidate = f"{rng.choice(_ADJECTIVES)}-{rng.choice(_NOUNS)}-{rng.randint(0, 99)}"
        if candidate not in used:
            used.add(candidate)
            return candidate
    # fallback improbable: sufijo incremental
    candidate = f"model-{len(used)}"
    used.add(candidate)
    return candidate


def _make_model(capability: int, rng: random.Random, used: set[str]) -> SyntheticModel:
    model_id = _fake_id(rng, used)
    tags = _skill_tags(capability, rng)
    # coste creciente con la capacidad + ruido (más capaz suele costar más)
    cost = round((capability + 1) * 2 * rng.uniform(0.8, 1.3), 1)
    ctx = (capability + 1) * 4096 * rng.choice((1, 2))
    supports_tools = rng.random() < (0.35 + 0.10 * capability)
    locality = Locality.API if rng.random() < (0.25 + 0.12 * capability) else Locality.LOCAL
    spec = ModelSpec(
        id=model_id,
        tags=tags,
        context_window=ctx,
        cost=cost,
        locality=locality,
        supports_tools=supports_tools,
    )
    return SyntheticModel(spec=spec, capability=capability)


def generate_pool(
    rng: random.Random,
    min_models: int = MIN_POOL,
    max_models: int = MAX_POOL,
    capabilities: list[int] | None = None,
) -> list[SyntheticModel]:
    """Genera un pool sintético con orden aleatorizado.

    Si `capabilities` se pasa, usa esas capacidades (para controlar la firma del
    pool en los splits); si no, las muestrea con variedad.
    """
    if capabilities is None:
        n = rng.randint(min_models, max_models)
        capabilities = [rng.randint(0, MAX_CAPABILITY) for _ in range(n)]
    used: set[str] = set()
    models = [_make_model(c, rng, used) for c in capabilities]
    rng.shuffle(models)
    return models


def capability_from_tags(tags: tuple[str, ...]) -> int:
    """Recupera la capacidad oculta (0..4) desde los tags observables del render.

    Inverso de _skill_tags: expert->4, planning->3, reasoning->2, code->1, resto->0.
    """
    t = set(tags)
    if "expert" in t:
        return 4
    if "planning" in t:
        return 3
    if "reasoning" in t:
        return 2
    if "code" in t:
        return 1
    return 0


def pool_signature(pool: list[SyntheticModel]) -> tuple[int, tuple[int, ...], tuple[int, ...]]:
    """Firma ESTRUCTURAL del pool (ignora nombres/costes aleatorios).

    (tamaño, capacidades ordenadas, patrón de soporte de tools ordenado). Se usa
    para garantizar que el test solo tiene composiciones nunca vistas en train.
    """
    caps = tuple(sorted(sm.capability for sm in pool))
    tools = tuple(sorted(int(sm.spec.supports_tools) for sm in pool))
    return (len(pool), caps, tools)
