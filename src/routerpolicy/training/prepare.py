"""Preprocesado de ejemplos para entrenamiento e inferencia.

- merge_system_into_user: Gemma no admite rol `system` en su plantilla de chat;
  se fusiona el system al inicio del primer turno de usuario. El MISMO formato
  se usa en inferencia (contrato consistente train/inference).
- tokenize_with_completion_mask: máscara de pérdida SOLO sobre la respuesta del
  assistant (el prefijo prompt va a -100).
"""

from __future__ import annotations

from typing import Any

IGNORE_INDEX = -100

Message = dict[str, str]

# Plantilla de chat estilo Gemma. El base `gemma-3-270m` no trae ninguna; como
# hacemos full fine-tune y el formato es NUESTRO contrato, lo fijamos aquí y lo
# usamos idéntico en entrenamiento e inferencia. Espera mensajes ya sin `system`
# (fusionado en user). Usa los tokens especiales de Gemma.
GEMMA_CHAT_TEMPLATE = (
    "{{ bos_token }}"
    "{% for message in messages %}"
    "<start_of_turn>{{ 'user' if message['role'] == 'user' else 'model' }}\n"
    "{{ message['content'] | trim }}<end_of_turn>\n"
    "{% endfor %}"
    "{% if add_generation_prompt %}<start_of_turn>model\n{% endif %}"
)


def ensure_chat_template(tokenizer: Any) -> None:
    """Fija nuestra plantilla si el tokenizer no trae ninguna (base sin template)."""
    if not getattr(tokenizer, "chat_template", None):
        tokenizer.chat_template = GEMMA_CHAT_TEMPLATE


def merge_system_into_user(messages: list[Message]) -> list[Message]:
    """Fusiona el turno system en el primer user (compatibilidad Gemma)."""
    system_text = ""
    rest: list[Message] = []
    for msg in messages:
        if msg["role"] == "system":
            system_text = msg["content"]
        else:
            rest.append(dict(msg))
    if system_text and rest and rest[0]["role"] == "user":
        rest[0]["content"] = f"{system_text}\n\n{rest[0]['content']}"
    return rest


def _prompt_only(messages: list[Message]) -> list[Message]:
    return [m for m in messages if m["role"] != "assistant"]


def tokenize_with_completion_mask(
    tokenizer: Any, messages: list[Message], max_len: int
) -> dict[str, list[int]]:
    """Tokeniza el chat completo enmascarando el prefijo (prompt) a -100.

    Solo la respuesta del assistant contribuye a la pérdida.
    """
    prepared = merge_system_into_user(messages)
    full = tokenizer.apply_chat_template(
        prepared, tokenize=True, add_generation_prompt=False, return_dict=True
    )["input_ids"]
    prompt = tokenizer.apply_chat_template(
        _prompt_only(prepared), tokenize=True, add_generation_prompt=True, return_dict=True
    )["input_ids"]
    labels = list(full)
    n_prompt = min(len(prompt), len(labels))
    for i in range(n_prompt):
        labels[i] = IGNORE_INDEX
    input_ids = list(full)[:max_len]
    labels = labels[:max_len]
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": [1] * len(input_ids),
    }
