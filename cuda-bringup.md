# 5090 bring-up checklist (Phase 2 tail)

Scope: functional validation of the batched engine on CUDA — *not* the
perf pass (SDPA/`torch.compile`/graphs are Phase 3; einsum on CUDA is
expected to be correct and slow). Disposable — delete once validated.

The code was written device-agnostic and audited for this (masks/aranges
built on the input's device, scheduler tensors moved at the
`runtime.forward_batched` boundary, spawn-context IPC so CUDA never crosses
a fork), but none of it has executed on CUDA yet. Written 2026-07-12 on the
Mac; expect surprises.

1. **Install.** `uv sync` — pulls the default torch wheel, which bundles
   CUDA 12.8+ for torch ≥ 2.8 (sm_120 needs it). Check the driver first:
   `nvidia-smi` must report driver ≥ 570.
   Sanity: `python -c "import torch; print(torch.__version__, torch.cuda.get_device_name(0))"`
2. **Suite on CPU** (scheduler correctness is device-independent):
   `python -m pytest tests/ -v` — should be green exactly as on the Mac.
3. **Sequential engine on CUDA** (einsum path, isolates model-vs-scheduler):
   `python -m cantollm.main serve --model 0.6B --device cuda`
   then a short chat via `python -m cantollm.main chat`. Watch for dtype or
   "no kernel image" errors at the first forward.
4. **Batched engine, in-process first** (fewer moving parts than IPC):
   `python -m cantollm.main serve --engine batched --in-process --device cuda`
   — 2–3 concurrent chats; confirm streams don't interleave content.
5. **Batched engine, process split** (the production shape):
   `python -m cantollm.main serve --engine batched --device cuda`
   — confirm two processes (`nvidia-smi` shows the *engine* process owning
   the VRAM, not the API), streams work, Ctrl-C exits both, `kill -9` the
   engine pid → in-flight requests get error events, API stays up.
6. **Scale sanity:** bump `--max-batch 16 --batch-max-seq-len 8192` on a
   bigger model (4B/8B) and re-run a few concurrent chats; watch
   `nvidia-smi` for the KV pool allocation stepping up accordingly.
7. **Rough numbers** (not the baseline — that waits for the bench spec):
   `python -m cantollm.main bench --prompts prompts.txt -c 10` against
   batched vs sequential, just to confirm the 2.4× MPS story holds or
   improves on CUDA.
