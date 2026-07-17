"""Render determinista y compacto del registro (contrato de entrada FIJO).

Una línea por modelo, formato idéntico siempre. Se augmenta el CONTENIDO y el
ORDEN (Fase 3), nunca el formato: el formato es parte del contrato del producto
y un modelo mini no puede parsear formatos variados.
"""

from __future__ import annotations

from routerpolicy.schema.core import Locality, ModelSpec, Registry

# Presupuesto de render (sección 2): pools de 2 a 8 modelos deben caber holgados.
RENDER_TOKEN_BUDGET = 1000

_HEADER = "Available models:"


def _format_cost(cost: float) -> str:
    """Coste compacto: entero sin decimales, si no float mínimo."""
    if cost == int(cost):
        return str(int(cost))
    return repr(cost)


def _render_model_line(spec: ModelSpec) -> str:
    tags = ",".join(spec.tags)
    tools = "yes" if spec.supports_tools else "no"
    return (
        f"- {spec.id} | tags:{tags} | ctx:{spec.context_window} "
        f"| cost:{_format_cost(spec.cost)} | loc:{spec.locality.value} | tools:{tools}"
    )


def render_registry_prompt(registry: Registry) -> str:
    """Renderiza el registro como bloque compacto, una línea por modelo.

    Determinista respecto al orden de `registry.models`. Sin salto final.
    """
    lines = [_HEADER]
    lines.extend(_render_model_line(m) for m in registry.models)
    return "\n".join(lines)


def _parse_model_line(line: str) -> ModelSpec:
    body = line[2:] if line.startswith("- ") else line
    fields = {}
    parts = body.split(" | ")
    model_id = parts[0].strip()
    for part in parts[1:]:
        key, _, value = part.partition(":")
        fields[key.strip()] = value.strip()
    return ModelSpec(
        id=model_id,
        tags=tuple(fields["tags"].split(",")),
        context_window=int(fields["ctx"]),
        cost=float(fields["cost"]),
        locality=Locality(fields["loc"]),
        supports_tools=fields["tools"] == "yes",
    )


def parse_registry_prompt(rendered: str) -> Registry:
    """Inverso de `render_registry_prompt`: reconstruye el Registry.

    Necesario en evaluación/inferencia para recuperar ids y costes del registro
    embebido en el prompt (decoding constreñido y regret de coste). Acepta tanto
    el render puro como el mensaje de usuario completo (registro + tarea): solo
    parsea el bloque del registro, ignorando la tarea (que puede tener viñetas).
    """
    head = rendered.split("\nTask:", 1)[0]  # descarta la sección de la tarea
    lines = [ln for ln in head.splitlines() if ln.startswith("- ") and " | tags:" in ln]
    return Registry(models=tuple(_parse_model_line(ln) for ln in lines))
