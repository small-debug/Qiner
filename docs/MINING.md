# Miner parameter guide

This guide helps you choose **[Number of threads]** when running Qiner, depending on your hardware.

## Quick reference

| Your setup | What to use for [Number of threads] |
| --------- | ----------------------------------- |
| **GPU only** (NVIDIA GPU, Qiner built with CUDA) | **0** - only the GPU mines; no CPU mining threads. |
| **CPU only** (no GPU or no CUDA build) | **Your CPU core count** - e.g. 4, 8, or 16. |
| **Hybrid** (GPU + some CPU mining) | **2–4** (or a small number) - GPU thread plus a few CPU threads. |

## Command format

```
./Qiner <Node IP> <Node Port> <MiningID> <Signing Seed> <Mining Seed> <Number of threads> [GPU batch size]
```

The 7th argument **[GPU batch size]** is optional (CUDA builds only). Omit it for default 256. See the table below to match your GPU to a recommended batch size.

On Windows, use `Qiner.exe` instead of `./Qiner`.

## GPU batch size by GPU performance

Use this table to pick a batch size that fits your GPU. If you're unsure, start with the default (omit the 7th argument) or the value for your tier; you can try the next tier up and compare **it/s** in the miner output.

| GPU tier | Typical VRAM | Example GPUs | Recommended batch size |
| -------- | ------------- | ------------ | ---------------------- |
| Entry / older | 4–6 GB | GTX 1650, GTX 1060, RTX 3050 | **256** (default) |
| Mid-range | 8 GB | RTX 3060, RTX 4060, RTX 2070 | **512** |
| High-end | 10–12 GB | RTX 3080, RTX 4070, RTX 4070 Ti | **1024** |
| Enthusiast | 16 GB+ | RTX 4080, RTX 4090, RTX 3090 | **2048** or **4096** |

- **Don't know your GPU?** On Windows: open Device Manager → Display adapters. On Linux: run `nvidia-smi`. The listed model and "Memory" (VRAM) tell you the tier.
- If the miner crashes or shows errors with a higher batch size, use the next tier down (e.g. 1024 → 512).
- Allowed range: **64–4096**. Values outside this are clamped automatically.

## Multiple GPUs (VPS or multi-GPU machine)

A single Qiner process uses **one GPU only** (the default, device 0). On a machine with several GPUs, the others are not used unless you run **one Qiner process per GPU**.

No code changes are needed. Set the environment variable **`CUDA_VISIBLE_DEVICES`** so each process sees a different GPU:

- **Linux (example: 3 GPUs):**
  ```bash
  CUDA_VISIBLE_DEVICES=0 ./Qiner <Node IP> <Port> <MiningID> <Signing> <Mining> 0 &
  CUDA_VISIBLE_DEVICES=1 ./Qiner <Node IP> <Port> <MiningID> <Signing> <Mining> 0 &
  CUDA_VISIBLE_DEVICES=2 ./Qiner <Node IP> <Port> <MiningID> <Signing> <Mining> 0 &
  ```

- **Windows (cmd):** Before each Qiner run, set the device index (e.g. GPU 1):
  ```cmd
  set CUDA_VISIBLE_DEVICES=1
  Qiner.exe <Node IP> <Port> <MiningID> <Signing> <Mining> 0
  ```
  Run separate command prompts (or scripts) for each GPU, each with a different `CUDA_VISIBLE_DEVICES` (0, 1, 2, …).

- **Windows (PowerShell):**
  ```powershell
  $env:CUDA_VISIBLE_DEVICES="1"
  .\Qiner.exe <Node IP> <Port> <MiningID> <Signing> <Mining> 0
  ```

GPU indices (0, 1, 2, …) match the order shown by `nvidia-smi`. This works with the current build; no recompile required.

## Copy-paste examples

Replace `<Node IP>`, `<Node Port>`, `<MiningID>`, `<Signing Seed>`, and `<Mining Seed>` with your real values.

**GPU-only (use 0 threads):**
```bash
./Qiner <Node IP> <Node Port> <MiningID> <Signing Seed> <Mining Seed> 0
```

**CPU-only, 8 cores:**
```bash
./Qiner <Node IP> <Node Port> <MiningID> <Signing Seed> <Mining Seed> 8
```

**Hybrid (GPU + 4 CPU threads):**
```bash
./Qiner <Node IP> <Node Port> <MiningID> <Signing Seed> <Mining Seed> 4
```

**GPU-only with larger batch (e.g. 1024 for high-end GPU):**
```bash
./Qiner <Node IP> <Node Port> <MiningID> <Signing Seed> <Mining Seed> 0 1024
```

## Notes

- If you have a dedicated NVIDIA GPU and built Qiner with CUDA, use **0** for threads to mine only on the GPU (no need for a high-performance CPU).
- For CPU-only mining, set threads to the number of cores you want to use (e.g. 8 on an 8-core machine).
- For hybrid, a small value like 2–4 is usually enough so the GPU stays busy and the CPU contributes a bit.
- Optional 7th argument (GPU batch size): default 256; use 512–2048 on high-end GPUs for higher it/s. Allowed range 64–4096.
