"""Entrenamiento de un peldaño (Fase 4). Full fine-tune, máscara solo-assistant.

Modos:
  --smoke N   overfit intencional sobre N ejemplos (valida el pipeline; reporta
              exact match, que debe acercarse al 100%).
  (normal)    entrenamiento completo con checkpoints reanudables + tensorboard.

Reanudable: --resume continúa desde el último checkpoint del output-dir.
Reporta VRAM pico y tokens/s (informe de recursos de la sección 3).

Uso:
    uv run --extra train python scripts/train.py --config configs/train_gemma_270m.yaml --smoke 200
    uv run --extra train python scripts/train.py --config configs/train_gemma_270m.yaml
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from routerpolicy.training.config import TrainConfig, load_train_config
from routerpolicy.training.data import gold_completion, load_rows
from routerpolicy.training.prepare import (
    ensure_chat_template,
    merge_system_into_user,
    tokenize_with_completion_mask,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _force_utf8_stdout() -> None:
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if callable(reconfigure):
        reconfigure(encoding="utf-8", errors="replace")


class SelectiveLossTrainer(Trainer):
    """Trainer que proyecta a vocab SOLO en las posiciones del assistant.

    Gemma tiene vocab de 262k; calcular logits para las ~750 posiciones de prompt
    (enmascaradas) domina el coste. Aquí se ejecuta el cuerpo del modelo, se
    seleccionan los hidden states de las posiciones con label != -100, y se aplica
    `lm_head` solo a esos (~decenas por ejemplo). Misma pérdida, ~30x más rápido.
    """

    def compute_loss(  # type: ignore[override]
        self,
        model: Any,
        inputs: dict[str, torch.Tensor],
        return_outputs: bool = False,
        **kwargs: Any,
    ) -> Any:
        labels = inputs["labels"]
        base = model.model(
            input_ids=inputs["input_ids"], attention_mask=inputs.get("attention_mask")
        )
        hidden = base.last_hidden_state
        shift_hidden = hidden[:, :-1, :]
        shift_labels = labels[:, 1:]
        mask = shift_labels != -100
        sel_hidden = shift_hidden[mask]
        sel_labels = shift_labels[mask]
        logits = model.lm_head(sel_hidden)
        loss = torch.nn.functional.cross_entropy(logits.float(), sel_labels)
        return (loss, base) if return_outputs else loss


class PadCollator:
    """Padding manual a tensores (input_ids con pad, labels con -100).

    Si `fixed_len` se indica, pad a esa longitud constante (shapes fijos ->
    evita recompilación de kernels por longitud variable, clave en Blackwell).
    """

    def __init__(self, pad_id: int, fixed_len: int | None = None) -> None:
        self.pad_id = pad_id
        self.fixed_len = fixed_len

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        maxlen = self.fixed_len or max(len(f["input_ids"]) for f in features)
        input_ids, labels, attn = [], [], []
        for f in features:
            pad = maxlen - len(f["input_ids"])
            input_ids.append(f["input_ids"] + [self.pad_id] * pad)
            labels.append(f["labels"] + [-100] * pad)
            attn.append(f["attention_mask"] + [0] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
        }


def _build_dataset(rows: list[dict[str, Any]], tokenizer: Any, max_len: int) -> tuple[Dataset, int]:
    feats = []
    total_tokens = 0
    for row in rows:
        enc = tokenize_with_completion_mask(tokenizer, row["messages"], max_len)
        total_tokens += len(enc["input_ids"])
        feats.append(enc)
    return Dataset.from_list(feats), total_tokens


def _last_checkpoint(output_dir: Path) -> str | None:
    if not output_dir.exists():
        return None
    ckpts = sorted(
        (p for p in output_dir.glob("checkpoint-*") if p.is_dir()),
        key=lambda p: int(p.name.split("-")[1]),
    )
    return str(ckpts[-1]) if ckpts else None


def _eos_ids(tokenizer: Any) -> list[int]:
    """eos + <end_of_turn> (Gemma): la generación para al terminar el turno."""
    ids = [tokenizer.eos_token_id]
    eot = tokenizer.convert_tokens_to_ids("<end_of_turn>")
    if isinstance(eot, int) and eot >= 0 and eot not in ids:
        ids.append(eot)
    return ids


def _generate_decision(model: Any, tokenizer: Any, messages: list[dict[str, Any]]) -> str:
    prepared = merge_system_into_user([m for m in messages if m["role"] != "assistant"])
    input_ids = tokenizer.apply_chat_template(
        prepared,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )["input_ids"].to(model.device)
    with torch.no_grad():
        out = model.generate(
            input_ids,
            max_new_tokens=48,
            do_sample=False,
            eos_token_id=_eos_ids(tokenizer),
            pad_token_id=tokenizer.pad_token_id,
        )
    new_tokens = out[0][input_ids.shape[1] :]
    return str(tokenizer.decode(new_tokens, skip_special_tokens=True).strip())


def _run_smoke_eval(model: Any, tokenizer: Any, rows: list[dict[str, Any]]) -> float:
    model.eval()
    exact = 0
    for row in rows:
        gen = _generate_decision(model, tokenizer, row["messages"])
        if gen == gold_completion(row):
            exact += 1
    return exact / len(rows) if rows else 0.0


def main(argv: list[str] | None = None) -> int:
    _force_utf8_stdout()
    parser = argparse.ArgumentParser(description="Entrenamiento de un peldaño (Fase 4)")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--dataset", type=Path, default=REPO_ROOT / "data" / "dataset" / "train.jsonl"
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--smoke", type=int, default=0, help="overfit N ejemplos (0=run normal)")
    parser.add_argument(
        "--max-steps", type=int, default=-1, help="tope de pasos (para demo/resume)"
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)

    cfg: TrainConfig = load_train_config(args.config)
    output_dir = args.output or (REPO_ROOT / "checkpoints" / cfg.step_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"peldaño: {cfg.step_name}  base: {cfg.model_id}", flush=True)
    tokenizer: Any = AutoTokenizer.from_pretrained(cfg.model_id)
    ensure_chat_template(tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model: Any = AutoModelForCausalLM.from_pretrained(cfg.model_id, dtype=torch.bfloat16)
    model.to("cuda")

    rows = load_rows(args.dataset, limit=args.smoke if args.smoke else None)
    epochs = 40.0 if args.smoke else cfg.num_train_epochs
    print(f"ejemplos: {len(rows)}  epochs: {epochs}", flush=True)

    dataset, total_tokens = _build_dataset(rows, tokenizer, cfg.max_seq_len)
    # shape fijo: pad a max_seq_len constante -> evita recompilación por longitud
    collator = PadCollator(pad_id=tokenizer.pad_token_id, fixed_len=cfg.max_seq_len)

    targs = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        num_train_epochs=epochs,
        max_steps=args.max_steps,
        learning_rate=cfg.learning_rate,
        warmup_ratio=cfg.warmup_ratio,
        weight_decay=cfg.weight_decay,
        lr_scheduler_type=cfg.lr_scheduler_type,
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,
        seed=cfg.seed,
        bf16=cfg.bf16,
        gradient_checkpointing=cfg.gradient_checkpointing,
        report_to=["tensorboard"],
        logging_dir=str(output_dir / "tb"),
        dataloader_num_workers=0,  # Windows: >0 estanca el bucle (spawn)
        remove_unused_columns=False,
    )
    trainer = SelectiveLossTrainer(
        model=model, args=targs, train_dataset=dataset, data_collator=collator
    )

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    resume = _last_checkpoint(output_dir) if args.resume else None
    if resume:
        print(f"reanudando desde {resume}", flush=True)
    result = trainer.train(resume_from_checkpoint=resume)
    elapsed = time.time() - t0

    peak_gb = torch.cuda.max_memory_allocated() / (1024**3)
    runtime = result.metrics.get("train_runtime", elapsed)
    tok_per_s = total_tokens * epochs / runtime if runtime else 0.0
    print("\n===== INFORME DE RECURSOS =====")
    print(f"  parámetros: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    print(f"  VRAM pico: {peak_gb:.2f} GB")
    print(f"  tokens/s (entrenamiento): {tok_per_s:,.0f}")
    print(f"  train_runtime: {runtime:.1f}s  loss final: {result.metrics.get('train_loss', 0):.4f}")

    if args.smoke:
        acc = _run_smoke_eval(model, tokenizer, rows)
        print(f"\n===== SMOKE (overfit {len(rows)}) =====")
        print(f"  exact match: {acc:.3f}")
        return 0 if acc >= 0.95 else 1

    trainer.save_model(str(output_dir / "final"))
    tokenizer.save_pretrained(str(output_dir / "final"))
    print(f"\nmodelo final guardado en {output_dir / 'final'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
