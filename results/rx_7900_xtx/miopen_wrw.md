# The slow convolution weight gradient on the Radeon RX 7900 XTX (gfx1100)

Investigation record, 2026-11-09/10. Container `igm-bench:rocm` (ROCm 7.0.2, MIOpen 3.5.0, TensorFlow 2.17.1); also
checked with ROCm 7.2.4 and 10.1. The MIOpenDriver probe tables are in [`miopen/`](miopen/), the IGM and
`ceilings.py` runs of the variants in [`experiments/`](experiments/), the working configuration in
[`../rx_7900_xtx_patches/`](../rx_7900_xtx_patches/).

## Symptom

IGM training (`online_f32`) is 5-12× slower than on an RTX 4090 while inference is on par. `scripts/ceilings.py`
isolates it to the convolution gradient with respect to the weights (∂weights), 3×3, C → C channels, float32, on
the 400 m grid (1 × 1505 × 2255):

| achieved TFLOP/s | forward | ∂input | ∂weights |
|---|---|---|---|
| C = 24 | 11.9 | 11.5 | **0.26** |
| C = 48 | 17.8 | 19.2 | **0.99** |
| MI250X (one GCD), for comparison | 8.7 / 13.6 | 8.6 / 13.6 | 6.2 / 10.9 |

## Cause

MIOpen has exactly two weight-gradient solvers applicable on gfx1100 for this problem:

| solver | time, C = 24 | note |
|---|---|---|
| `GemmWrwUniversal` | 130 ms | chosen. Im2Col into a `H·W·C·9·4` byte work buffer (2.93 GB here), then one rocBLAS GEMM with a 24 × 216 result and a 3.4-million-long reduction. The time barely depends on C (138 ms at C = 48). |
| `ConvDirectNaiveConvWrw` | 918 ms | reference kernel |

Every other weight-gradient solver reports "not applicable" on this GPU, with `MIOPEN_FIND_MODE=1` as well
(`MIOpenDriver ... -S <solver>`, [`miopen/drv_c24_solvers.txt`](miopen/drv_c24_solvers.txt)):

- `ConvOclBwdWrW2<1..16>`, `ConvOclBwdWrW2NonTunable`, `ConvOclBwdWrW53`: disabled on gfx11/gfx12 by MIOpen
  itself, "due to random GPU memory access faults" (`WORKAROUND_LWPMIOPEN_1392`, `conv_ocl_dir2D_bwdWrW_2.cpp`),
  and deprecated (`ThisSolverIsDeprecatedStatic`: allowed on gfx803/900/906/908/90a and gfx103x only). Re-enabled
  with two debug switches and tested below: fast and wrong.
- `ConvAsmBwdWrW3x3`: "Only Gfx8 and Gfx9 ISA is supported".
- `ConvHipImplicitGemmV4R1WrW`, `ConvHipImplicitGemmV4R4WrW`: HIPRTC compilation error on gfx1100.
- `ConvMlirIgemmWrW`: `MIIR_INVALID_PARAM`.
- all `*Xdlops*` solvers: need matrix cores (MFMA), absent on RDNA3.
- `ConvBinWinogradRxSf2x3`: applicable only below a size limit, see below.

MIOpen 3.5 ships tuned databases for 13 GPU families (down to gfx803) and none for gfx1100, so tuning
(`MIOPEN_FIND_ENFORCE`) has nothing to work on either. ROCm 7.2.4 behaves identically; ROCm 10.1 picks the same
solver and is slower (212 ms, [`miopen/drv_newer_rocm.txt`](miopen/drv_newer_rocm.txt)).

The 4 GiB "Failed to allocate scratch memory" abort of IGM at the finer grids is TensorFlow allocating, before its
algorithm search, the work buffer of the largest *applicable* algorithm, i.e. this Im2Col buffer (5.2 GB at 300 m,
11.7 GB at 200 m), even where MIOpen then selects a kernel without one: for the forward and ∂input convolutions
over the whole 200 m grid, find picks Winograd (4.8 ms, no buffer) over GEMM (49 ms, 11.7 GB),
[`miopen/drv_200m_fwd_bwd.txt`](miopen/drv_200m_fwd_bwd.txt). `TF_CUDNN_WORKSPACE_LIMIT_IN_MB` raises the cap.

## The Winograd limit

`ConvBinWinogradRxSf2x3` does run the weight gradient on gfx11, and it is 2.8× faster on this problem, but only
within its shader limits (`conv_winoRxS.cpp`, `IsShaderConstraintsMetV30`). For the weight gradient the image
dimensions take the place of the filter and the batch takes the place of the channels, and the binding constraint
`(C+1)·R·S < 2²²` becomes

    (N + 1) × H × W < 4 194 304      (N images of H × W pixels in one call)

| `MIOpenDriver conv ... -F 4`, C = 24 | solver | time |
|---|---|---|
| 1 × 1505 × 2255 (whole 400 m grid, 3.39 Mpx) | GemmWrwUniversal | 130 ms |
| 13 × 512 × 512 (same area as patches) | Winograd | 46 ms |
| 52 × 256 × 256 | Winograd | 53 ms |
| 4 × 752 × 1128 (5 × 0.85 Mpx = 4.24 M: just above) | GemmWrwUniversal | 92 ms |
| 1 × 1024 × 1024 | Winograd | 26 ms |

([`miopen/drv_wino_limit.txt`](miopen/drv_wino_limit.txt), [`drv_wino_limit2.txt`](miopen/drv_wino_limit2.txt)).

## What works: training on patches, in mini-batches

IGM's `unified.data_preparation.framesizemax` cuts the emulator's training domain into non-overlapping patches
of at most that many cells, and `patches_per_batch` is the mini-batch size with which Adam sweeps all patches
in every iteration (one update per mini-batch; `igm/processes/iceflow/unified/optimizers/adam.py`). On the
patch shapes IGM produces (`ny // (ny // F + 1)` cells a side), [`miopen/drv_patch_shapes.txt`](miopen/drv_patch_shapes.txt):

| grid | `framesizemax=512` | all patches in one call | `framesizemax=1024` | one call |
|---|---|---|---|---|
| 1600 m | 2 × 376 × 281 | Winograd, 3.9 ms | 1 × 376 × 563 | Winograd, 5.7 ms |
| 800 m | 6 × 376 × 375 | Winograd, 11.2 ms | 2 × 752 × 563 | Winograd, 14.9 ms |
| 400 m | 15 × 501 × 451 | Winograd, 39.4 ms | 6 × 752 × 751 | Winograd, 44.1 ms |
| 300 m | 24 × 501 × 501 | GemmWrwUniversal, 240 ms | 6 × 1003 × 1002 | GemmWrwUniversal, 238 ms |
| 200 m | 54 × 501 × 501 | GemmWrwUniversal, 540 ms | 15 × 1003 × 902 | GemmWrwUniversal, 565 ms |

One batch of all patches is therefore enough at 400 m but not finer, and in IGM it ran out of memory at 400 m
anyway (a 7.5 GB XLA buffer that the whole-grid run does not have, [`experiments/patches_one_batch`](experiments/patches_one_batch)).
Mini-batches of 3 patches of 501² (`(3+1) × 501 × 501 = 1.0 M < 2²²`) keep the Winograd kernel at every grid,
divide the patch counts 15, 24 and 54, and bound the memory:

| `online_f32`, ice flow ms/step (TF peak GiB) | whole grid | patches ≤ 512, mini-batches of 3 |
|---|---|---|
| 1600 m | 57 (0.74) | 44 (0.74) ¹ |
| 800 m | 210 (2.79) | 138 (3.03) ¹ |
| 400 m | 1168 (16.75) | 584 (2.76) |
| 300 m | OOM | 1029 (3.39) |
| 200 m | not run | 2313 (6.03) ² |

¹ measured with all patches in one batch (2 patches at 1600 m, i.e. the same configuration; 6 at 800 m).
² with `TF_CUDNN_WORKSPACE_LIMIT_IN_MB=16384`, see above.

At 400 m this is 6.3× the RTX 4090's step (93 ms) instead of 12.6×. The remaining gap is the weight gradient
still: Winograd on these shapes runs at 0.9 TFLOP/s, against 12 for the forward pass. The physics is the same
(same cells, same cost function); the training trajectory differs slightly because one IGM step now makes 5
optimizer updates at 400 m instead of 1 (ice volume after 88 years: 2524 vs 2409 km³).

## What does not work

**The hidden OpenCL kernel.** With `MIOPEN_DEBUG_CONV_DIRECT_OCL_WRW2=1 MIOPEN_DEBUG_ENABLE_DEPRECATED_SOLVERS=1`,
`ConvOclBwdWrW53` becomes applicable on gfx1100 (the `ConvOclBwdWrW2<n>` variants stay "not applicable" at these
sizes; `ConvOclBwdWrW2NonTunable` runs on 1024² only). It is fast, needs no work buffer, and with
`MIOPEN_FIND_MODE=1` find selects it (TensorFlow's default find mode does not evaluate it):

| ∂weights, C = 24 | GemmWrwUniversal | Winograd | ConvOclBwdWrW53 |
|---|---|---|---|
| 1 × 1505 × 2255 (400 m) | 130 ms | not applicable | 18 ms |
| 1 × 1024 × 1024 | | 26 ms | 2.3 ms |
| `ceilings.py` ∂weights, C = 24 / 48 | 0.26 / 0.99 TFLOP/s | | 2.42 / 2.55 TFLOP/s |
| IGM online 1600 / 800 / 400 m | 57 / 210 / 1168 ms | | 17.7 / 63 / 231 ms |

But it is wrong. Verified with `MIOpenDriver -V 1` against MIOpen's reference kernel and against the driver's CPU
reference (`MIOPEN_DRIVER_USE_GPU_REFERENCE=0`), [`miopen/drv_oclwrw53_verify.txt`](miopen/drv_oclwrw53_verify.txt),
[`drv_oclwrw53_cpuref.txt`](miopen/drv_oclwrw53_cpuref.txt):

| shape (N × H × W, C = 24 unless noted) | ConvOclBwdWrW53 |
|---|---|
| 1 × 1024 × 1024, 1 × 1200 × 1200, 1 × 1024 × 2255, 3 × 501 × 501, 15 × 501 × 451, C = 48: 1 × 1024 × 1024 | correct (2e-7) |
| 1 × 376 × 563 (1600 m), 1 × 752 × 1127 (800 m), 1 × 1505 × 2255 (400 m), 1 × 2006 × 3006 (300 m), 1 × 1400², 1 × 1505², 1 × 1600², 6 × 752 × 751, C = 48: 1 × 752 × 1127 | **NaN** |

The naive and Winograd kernels are exact on the same shapes. In IGM the ice volume is NaN from step 13 at 400 m
([`experiments/oclwrw53`](experiments/oclwrw53)); at 1600 and 800 m it stays finite but cannot be trusted. AMD's
default is right; the kernel is evidence for the issue below, not a workaround.

**Find mode and tuning.** `MIOPEN_FIND_MODE=1` gives the same solvers and times as the default DYNAMIC_HYBRID
mode ([`experiments/find_mode_1/ceilings.json`](experiments/find_mode_1/ceilings.json)); `MIOPEN_FIND_ENFORCE`
has no tunable solver to work on.

**NCHW inside the model.** TensorFlow on ROCm transposes NHWC to NCHW and back around each convolution, and raw
NCHW convolutions run 2× faster in `ceilings.py --nchw` (26 vs 12 TFLOP/s forward at C = 24). Rebuilding the
dahunet CNN with channels-first layers and one transpose in and out, however, changed nothing in IGM (offline
9.0 / 31.8 / 128 / 523 ms vs 9.1 / 31.7 / 128 / 521 at 1600 / 800 / 400 / 200 m, [`experiments/nchw_offline`](experiments/nchw_offline)):
TensorFlow's layout optimizer already does this in graph mode. The eager micro-benchmark overstates the cost.

**Newer ROCm.** 7.2.4 (TensorFlow 2.20) is identical; 10.1 is slower on the weight gradient (212 ms).

## Draft issue for ROCm/MIOpen

**Title:** gfx1100: no correct and efficient convolution weight-gradient solver for large single-image float32
problems (GemmWrwUniversal at 0.3 TFLOP/s; ConvOclBwdWrW53 7× faster but returns NaN)

On a Radeon RX 7900 XTX (gfx1100), ROCm 7.0.2 / MIOpen 3.5.0 (also 7.2.4 and 10.1), the backward-weights
convolution below runs at 0.26 TFLOP/s, while forward and backward-data run at 12 TFLOP/s:

    MIOpenDriver conv -n 1 -c 24 -H 1505 -W 2255 -k 24 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 4 -t 1 -V 0
    -> Solution: GemmWrwUniversal, 130 ms (workspace 2 932 221 600 bytes); ConvDirectNaiveConvWrw: 918 ms

`MIOPEN_LOG_LEVEL=6` shows these two as the only applicable solvers, with `MIOPEN_FIND_MODE=1` too. Forcing the
others with `-S`: `ConvAsmBwdWrW3x3` is gfx8/9 only, `ConvHipImplicitGemmV4R1WrW`/`V4R4WrW` fail to compile, the
Xdlops solvers need MFMA. `ConvBinWinogradRxSf2x3` works and is 2.8× faster, but its `(C+1)·R·S < 2^22` shader
limit, which for the weight gradient means `(N+1)·H·W < 2^22`, excludes images above ~2 Mpx.

The OpenCL solvers disabled on gfx11/12 by LWPMIOPEN-1392, re-enabled with `MIOPEN_DEBUG_CONV_DIRECT_OCL_WRW2=1
MIOPEN_DEBUG_ENABLE_DEPRECATED_SOLVERS=1`: `ConvOclBwdWrW53` runs the problem above in 18 ms (2.4 TFLOP/s, no
workspace) but fails verification (`-V 1`, GPU and CPU reference) with NaN on most shapes, e.g. 1 × 24 × 376 × 563,
752 × 1127, 1505 × 2255, 2006 × 3006, 1400², 1505², 1600², while 1024², 1200², 1024 × 2255 and batched 3 × 501²,
15 × 501 × 451 verify. The same problem on an MI250X runs at 6.2 TFLOP/s (`ConvAsmImplicitGemmGTCDynamicWrwXdlopsNHWC`).

Use case: physics-informed training of a small CNN on a whole 2D domain (IGM, the Instructed Glacier Model):
batch 1, 24-48 channels, images of 1-14 Mpx, float32.

History on your side: ROCm/MIOpen#2080 deprecated the `ConvOcl*` solvers for everything newer than gfx103x
(2023), with the June 2024 comment there that `ConvOclBwdWrW2<*>` "deliver good performance" and could be kept
"to avoid inadequate performance of WrW convolutions" on the new GPUs; ROCm/MIOpen#3634 then disabled them on
gfx11/12 because of random failures (2025). The result is that gfx11 has no fast weight-gradient path left for
large images.

Expected: a correct weight-gradient solver for RDNA3 that handles large images, e.g. the OpenCL kernel fixed, a
Winograd path without the 2^22 limit, a split-K GEMM for the 24 × 216 × (H·W) shape, or a WMMA implicit-GEMM kernel.
