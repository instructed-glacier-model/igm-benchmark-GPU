# MIOpenDriver probes on the Radeon RX 7900 XTX

Appendix to [miopen_wrw.md](miopen_wrw.md): the raw output of the MIOpenDriver probes (A1-A8) and the notes of the
IGM / `ceilings.py` runs kept in [`experiments/`](experiments/) (B). Container `igm-bench:rocm` (ROCm 7.0.2, MIOpen 3.5.0)
unless stated; `MIOpenDriver` is `/opt/rocm/bin/MIOpenDriver`; `-F 4` is the weight gradient, `-t 1` prints GPU kernel
time, `-i` the number of iterations, `-V 1` verifies against a reference. 2026-11-09/10.

## A1. Weight-gradient solvers forced one by one

The 400 m problem (1 × 24 × 1505 × 2255, 3×3, C = K = 24, float32), each candidate weight-gradient solver forced with `-S`. Default find mode, ROCm 7.0.2.

```
MIOpenDriver conv -n 1 -c 24 -H 1505 -W 2255 -k 24 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 4 -t 1 -V 0 -i 10 -S <solver>
```

```
--- GemmWrwUniversal
MIOpen Backward Weights Conv. Algorithm: 0, Solution: 102/GemmWrwUniversal
GPU Kernel Time Backward Weights Conv. Elapsed: 130.620880 ms (average)
--- ConvDirectNaiveConvWrw
MIOpen Backward Weights Conv. Algorithm: 1, Solution: 87/ConvDirectNaiveConvWrw
GPU Kernel Time Backward Weights Conv. Elapsed: 902.509338 ms (average)
--- ConvOclBwdWrW2NonTunable
MIOpen Error: /longer_pathname_so_that_rpms_can_support_packaging_the_debug_info_for_all_os_profiles/src/rocm-libraries/projects/miopen/src/ocl/convolutionocl.cpp:1232: The supplied solution id: ConvOclBwdWrW2NonTunable is not applicable to the current problem
--- ConvOclBwdWrW2<1>
MIOpen Error: /longer_pathname_so_that_rpms_can_support_packaging_the_debug_info_for_all_os_profiles/src/rocm-libraries/projects/miopen/src/ocl/convolutionocl.cpp:1232: The supplied solution id: ConvOclBwdWrW2<1> is not applicable to the current problem
--- ConvOclBwdWrW2<2>
MIOpen Error: /longer_pathname_so_that_rpms_can_support_packaging_the_debug_info_for_all_os_profiles/src/rocm-libraries/projects/miopen/src/ocl/convolutionocl.cpp:1232: The supplied solution id: ConvOclBwdWrW2<2> is not applicable to the current problem
--- ConvOclBwdWrW2<4>
MIOpen Error: /longer_pathname_so_that_rpms_can_support_packaging_the_debug_info_for_all_os_profiles/src/rocm-libraries/projects/miopen/src/ocl/convolutionocl.cpp:1232: The supplied solution id: ConvOclBwdWrW2<4> is not applicable to the current problem
--- ConvOclBwdWrW2<8>
MIOpen Error: /longer_pathname_so_that_rpms_can_support_packaging_the_debug_info_for_all_os_profiles/src/rocm-libraries/projects/miopen/src/ocl/convolutionocl.cpp:1232: The supplied solution id: ConvOclBwdWrW2<8> is not applicable to the current problem
--- ConvOclBwdWrW2<16>
MIOpen Error: /longer_pathname_so_that_rpms_can_support_packaging_the_debug_info_for_all_os_profiles/src/rocm-libraries/projects/miopen/src/ocl/convolutionocl.cpp:1232: The supplied solution id: ConvOclBwdWrW2<16> is not applicable to the current problem
--- ConvOclBwdWrW53
MIOpen Error: /longer_pathname_so_that_rpms_can_support_packaging_the_debug_info_for_all_os_profiles/src/rocm-libraries/projects/miopen/src/ocl/convolutionocl.cpp:1232: The supplied solution id: ConvOclBwdWrW53 is not applicable to the current problem
--- ConvAsmBwdWrW3x3
MIOpen(HIP): Error [Do] 'amd_comgr_do_action(kind, handle, in.GetHandle(), out.GetHandle())' AMD_COMGR_ACTION_ASSEMBLE_SOURCE_TO_RELOCATABLE: ERROR (1)
MIOpen(HIP): Error [BuildAsm] comgr status = ERROR (1)
MIOpen(HIP): Warning [BuildAsm] /tmp/comgr-7058c6/input/conv3x3wrw.s:118:5: error: Only Gfx8 and Gfx9 ISA is supported
--- ConvHipImplicitGemmV4R1WrW
MIOpen(HIP): Error [HeuristicInit] All attempts failed
MIOpen(HIP): Error [Compile] 'hiprtcCompileProgram(prog.get(), c_options.size(), c_options.data())' static_kernel_gridwise_convolution_implicit_gemm_v4r1_nchw_kcyx_nkhw_lds_double_buffer.cpp: HIPRTC_ERROR_COMPILATION (6)
MIOpen(HIP): Error [BuildHip] HIPRTC status = HIPRTC_ERROR_COMPILATION (6), source file: static_kernel_gridwise_convolution_implicit_gemm_v4r1_nchw_kcyx_nkhw_lds_double_buffer.cpp
--- ConvHipImplicitGemmV4R4WrW
MIOpen(HIP): Error [HeuristicInit] All attempts failed:
MIOpen(HIP): Error [Compile] 'hiprtcCompileProgram(prog.get(), c_options.size(), c_options.data())' static_kernel_gridwise_convolution_backward_weights_implicit_gemm_v4r4_nchw_kcyx_nkhw.cpp: HIPRTC_ERROR_COMPILATION (6)
MIOpen(HIP): Error [BuildHip] HIPRTC status = HIPRTC_ERROR_COMPILATION (6), source file: static_kernel_gridwise_convolution_backward_weights_implicit_gemm_v4r4_nchw_kcyx_nkhw.cpp
--- ConvMlirIgemmWrW
MIOpen Error: /longer_pathname_so_that_rpms_can_support_packaging_the_debug_info_for_all_os_profiles/src/rocm-libraries/projects/miopen/src/mlir_build.cpp:59: miirLowerTuningParams MIIR_INVALID_PARAM
MIOpen Error: /longer_pathname_so_that_rpms_can_support_packaging_the_debug_info_for_all_os_profiles/src/rocm-libraries/projects/miopen/src/mlir_build.cpp:59: miirLowerTuningParams MIIR_INVALID_PARAM
--- ConvWinograd3x3MultipassWrW<3-2>
MIOpen Error: /longer_pathname_so_that_rpms_can_support_packaging_the_debug_info_for_all_os_profiles/src/rocm-libraries/projects/miopen/src/ocl/convolutionocl.cpp:1232: The supplied solution id: ConvWinograd3x3MultipassWrW<3-2> is not applicable to the current problem
```

## A2. Image size sweep, default find and MIOPEN_FIND_MODE=1

The same convolution at increasing image sizes: which solver find picks and how long it takes, with the default DYNAMIC_HYBRID find mode and with `MIOPEN_FIND_MODE=1` (NORMAL).

```
MIOpenDriver conv -n 1 -c 24 -H <H> -W <W> -k 24 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 4 -t 1 -V 0 -i 10
```

```
##### -n 1 -H 512 -W 512
--- default find
MIOpen Backward Weights Conv. Algorithm: 3, Solution: 84/ConvBinWinogradRxSf2x3g1
GPU Kernel Time Backward Weights Conv. Elapsed: 6.832742 ms (average)
--- MIOPEN_FIND_MODE=1
MIOpen Backward Weights Conv. Algorithm: 3, Solution: 84/ConvBinWinogradRxSf2x3g1
GPU Kernel Time Backward Weights Conv. Elapsed: 6.835442 ms (average)
--- -S ConvOclBwdWrW2<1>
MIOpen Error: /longer_pathname_so_that_rpms_can_support_packaging_the_debug_info_for_all_os_profiles/src/rocm-libraries/projects/miopen/src/ocl/convolutionocl.c
--- -S ConvOclBwdWrW2<4>
MIOpen Error: /longer_pathname_so_that_rpms_can_support_packaging_the_debug_info_for_all_os_profiles/src/rocm-libraries/projects/miopen/src/ocl/convolutionocl.c
--- -S ConvOclBwdWrW2NonTunable
MIOpen Error: /longer_pathname_so_that_rpms_can_support_packaging_the_debug_info_for_all_os_profiles/src/rocm-libraries/projects/miopen/src/ocl/convolutionocl.c
--- -S ConvOclBwdWrW53
MIOpen Error: /longer_pathname_so_that_rpms_can_support_packaging_the_debug_info_for_all_os_profiles/src/rocm-libraries/projects/miopen/src/ocl/convolutionocl.c
##### -n 13 -H 512 -W 512
--- default find
MIOpen Backward Weights Conv. Algorithm: 3, Solution: 84/ConvBinWinogradRxSf2x3g1
GPU Kernel Time Backward Weights Conv. Elapsed: 46.488598 ms (average)
--- MIOPEN_FIND_MODE=1
MIOpen Backward Weights Conv. Algorithm: 3, Solution: 53/ConvBinWinogradRxSf2x3
GPU Kernel Time Backward Weights Conv. Elapsed: 47.304920 ms (average)
--- -S ConvOclBwdWrW2<1>
MIOpen Error: /longer_pathname_so_that_rpms_can_support_packaging_the_debug_info_for_all_os_profiles/src/rocm-libraries/projects/miopen/src/ocl/convolutionocl.c
--- -S ConvOclBwdWrW2<4>
MIOpen Error: /longer_pathname_so_that_rpms_can_support_packaging_the_debug_info_for_all_os_profiles/src/rocm-libraries/projects/miopen/src/ocl/convolutionocl.c
--- -S ConvOclBwdWrW2NonTunable
MIOpen Error: /longer_pathname_so_that_rpms_can_support_packaging_the_debug_info_for_all_os_profiles/src/rocm-libraries/projects/miopen/src/ocl/convolutionocl.c
--- -S ConvOclBwdWrW53
MIOpen Error: /longer_pathname_so_that_rpms_can_support_packaging_the_debug_info_for_all_os_profiles/src/rocm-libraries/projects/miopen/src/ocl/convolutionocl.c
##### -n 1 -H 256 -W 256
--- default find
MIOpen Backward Weights Conv. Algorithm: 3, Solution: 84/ConvBinWinogradRxSf2x3g1
GPU Kernel Time Backward Weights Conv. Elapsed: 1.741864 ms (average)
--- MIOPEN_FIND_MODE=1
MIOpen Backward Weights Conv. Algorithm: 3, Solution: 53/ConvBinWinogradRxSf2x3
GPU Kernel Time Backward Weights Conv. Elapsed: 1.748197 ms (average)
--- -S ConvOclBwdWrW2<1>
MIOpen Error: /longer_pathname_so_that_rpms_can_support_packaging_the_debug_info_for_all_os_profiles/src/rocm-libraries/projects/miopen/src/ocl/convolutionocl.c
--- -S ConvOclBwdWrW2<4>
MIOpen Error: /longer_pathname_so_that_rpms_can_support_packaging_the_debug_info_for_all_os_profiles/src/rocm-libraries/projects/miopen/src/ocl/convolutionocl.c
--- -S ConvOclBwdWrW2NonTunable
MIOpen Error: /longer_pathname_so_that_rpms_can_support_packaging_the_debug_info_for_all_os_profiles/src/rocm-libraries/projects/miopen/src/ocl/convolutionocl.c
--- -S ConvOclBwdWrW53
MIOpen Error: /longer_pathname_so_that_rpms_can_support_packaging_the_debug_info_for_all_os_profiles/src/rocm-libraries/projects/miopen/src/ocl/convolutionocl.c
##### -n 52 -H 256 -W 256
--- default find
MIOpen Backward Weights Conv. Algorithm: 3, Solution: 53/ConvBinWinogradRxSf2x3
GPU Kernel Time Backward Weights Conv. Elapsed: 52.744736 ms (average)
--- MIOPEN_FIND_MODE=1
MIOpen Backward Weights Conv. Algorithm: 3, Solution: 53/ConvBinWinogradRxSf2x3
GPU Kernel Time Backward Weights Conv. Elapsed: 52.134861 ms (average)
--- -S ConvOclBwdWrW2<1>
MIOpen Error: /longer_pathname_so_that_rpms_can_support_packaging_the_debug_info_for_all_os_profiles/src/rocm-libraries/projects/miopen/src/ocl/convolutionocl.c
--- -S ConvOclBwdWrW2<4>
MIOpen Error: /longer_pathname_so_that_rpms_can_support_packaging_the_debug_info_for_all_os_profiles/src/rocm-libraries/projects/miopen/src/ocl/convolutionocl.c
--- -S ConvOclBwdWrW2NonTunable
MIOpen Error: /longer_pathname_so_that_rpms_can_support_packaging_the_debug_info_for_all_os_profiles/src/rocm-libraries/projects/miopen/src/ocl/convolutionocl.c
--- -S ConvOclBwdWrW53
MIOpen Error: /longer_pathname_so_that_rpms_can_support_packaging_the_debug_info_for_all_os_profiles/src/rocm-libraries/projects/miopen/src/ocl/convolutionocl.c
```

## A3. Winograd applicability limit

Width and height sweeps around the limit `(N+1)·H·W < 2²²` of `ConvBinWinogradRxSf2x3` for the weight gradient: the solver switches to `GemmWrwUniversal` exactly when the product crosses 4 194 304.

```
MIOpenDriver conv -n <N> -c 24 -H <H> -W <W> -k 24 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 4 -t 1 -V 0 -i 10
```

`drv_wino_limit.txt`:

```
--- -n 1 -H 1505 -W 2048
MIOpen Backward Weights Conv. Algorithm: 0, Solution: 102/GemmWrwUniversal
GPU Kernel Time Backward Weights Conv. Elapsed: 92.376709 ms (average)
--- -n 1 -H 1505 -W 2049
MIOpen Backward Weights Conv. Algorithm: 0, Solution: 102/GemmWrwUniversal
GPU Kernel Time Backward Weights Conv. Elapsed: 120.782188 ms (average)
--- -n 1 -H 2048 -W 2048
MIOpen Backward Weights Conv. Algorithm: 0, Solution: 102/GemmWrwUniversal
GPU Kernel Time Backward Weights Conv. Elapsed: 161.870056 ms (average)
--- -n 1 -H 1024 -W 2255
MIOpen Backward Weights Conv. Algorithm: 0, Solution: 102/GemmWrwUniversal
GPU Kernel Time Backward Weights Conv. Elapsed: 69.455696 ms (average)
--- -n 1 -H 4096 -W 1024
MIOpen Backward Weights Conv. Algorithm: 0, Solution: 102/GemmWrwUniversal
GPU Kernel Time Backward Weights Conv. Elapsed: 178.315811 ms (average)
--- -n 2 -H 1505 -W 1128
MIOpen Backward Weights Conv. Algorithm: 0, Solution: 102/GemmWrwUniversal
GPU Kernel Time Backward Weights Conv. Elapsed: 135.039352 ms (average)
```

`drv_wino_limit2.txt`:

```
--- -n 1 -H 1024 -W 1024
MIOpen Backward Weights Conv. Algorithm: 3, Solution: 53/ConvBinWinogradRxSf2x3
GPU Kernel Time Backward Weights Conv. Elapsed: 25.966639 ms (average)
--- -n 1 -H 1024 -W 1023
MIOpen Backward Weights Conv. Algorithm: 3, Solution: 53/ConvBinWinogradRxSf2x3
GPU Kernel Time Backward Weights Conv. Elapsed: 25.961580 ms (average)
--- -n 1 -H 896 -W 896
MIOpen Backward Weights Conv. Algorithm: 3, Solution: 84/ConvBinWinogradRxSf2x3g1
GPU Kernel Time Backward Weights Conv. Elapsed: 20.023312 ms (average)
--- -n 1 -H 768 -W 768
MIOpen Backward Weights Conv. Algorithm: 3, Solution: 84/ConvBinWinogradRxSf2x3g1
GPU Kernel Time Backward Weights Conv. Elapsed: 14.797416 ms (average)
--- -n 1 -H 512 -W 2048
MIOpen Backward Weights Conv. Algorithm: 3, Solution: 53/ConvBinWinogradRxSf2x3
GPU Kernel Time Backward Weights Conv. Elapsed: 25.953983 ms (average)
--- -n 1 -H 2048 -W 512
MIOpen Backward Weights Conv. Algorithm: 3, Solution: 84/ConvBinWinogradRxSf2x3g1
GPU Kernel Time Backward Weights Conv. Elapsed: 26.289606 ms (average)
--- -n 1 -H 512 -W 1024
MIOpen Backward Weights Conv. Algorithm: 3, Solution: 84/ConvBinWinogradRxSf2x3g1
GPU Kernel Time Backward Weights Conv. Elapsed: 13.271428 ms (average)
--- -n 1 -H 752 -W 1128
MIOpen Backward Weights Conv. Algorithm: 3, Solution: 84/ConvBinWinogradRxSf2x3g1
GPU Kernel Time Backward Weights Conv. Elapsed: 21.251671 ms (average)
--- -n 4 -H 752 -W 1128
MIOpen Backward Weights Conv. Algorithm: 0, Solution: 102/GemmWrwUniversal
GPU Kernel Time Backward Weights Conv. Elapsed: 92.159256 ms (average)
```

## A4. Newer ROCm images

The A1 problem in `rocm/tensorflow:rocm7.2.4-py3.12-tf2.20-dev` (ROCm 7.2.4) and `rocm10.1-ubuntu24.04-py3.12-tf2.20` (ROCm 10.1), default find and `MIOPEN_FIND_MODE=1`.

```
MIOpenDriver conv -n 1 -c 24 -H 1505 -W 2255 -k 24 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 4 -t 1 -V 0 -i 10
```

```
--- rocm/tensorflow:rocm7.2.4-py3.12-tf2.20-dev: 7.2.4
MIOpen Backward Weights Conv. Algorithm: 0, Solution: 102/GemmWrwUniversal
GPU Kernel Time Backward Weights Conv. Elapsed: 130.173569 ms (average)
MIOpen Backward Weights Conv. Algorithm: 0, Solution: 102/GemmWrwUniversal
GPU Kernel Time Backward Weights Conv. Elapsed: 130.255402 ms (average)
--- rocm/tensorflow:rocm7.2.4-py3.12-tf2.20-dev, 13 x 512x512:
MIOpen Backward Weights Conv. Algorithm: 3, Solution: 84/ConvBinWinogradRxSf2x3g1
GPU Kernel Time Backward Weights Conv. Elapsed: 47.038349 ms (average)
--- rocm/tensorflow:rocm10.1-ubuntu24.04-py3.12-tf2.20: not pulled yet
##### rocm/tensorflow:rocm10.1-ubuntu24.04-py3.12-tf2.20: not pulled yet
##### rocm/tensorflow:rocm10.1-ubuntu24.04-py3.12-tf2.20: ROCm 10.1.0, MIOpen 1.0
--- full grid, default
MIOpen Backward Weights Conv. Algorithm: 0, Solution: 102/GemmWrwUniversal
GPU Kernel Time Backward Weights Conv. Elapsed: 211.790558 ms (average)
--- full grid, MIOPEN_FIND_MODE=1
MIOpen Backward Weights Conv. Algorithm: 0, Solution: 102/GemmWrwUniversal
GPU Kernel Time Backward Weights Conv. Elapsed: 201.626007 ms (average)
--- 13 x 512x512
MIOpen Backward Weights Conv. Algorithm: 3, Solution: 53/ConvBinWinogradRxSf2x3
GPU Kernel Time Backward Weights Conv. Elapsed: 47.613892 ms (average)
```

## A5. IGM's patch shapes

The weight gradient on the patches IGM produces per grid for `framesizemax` 512 and 1024 (`ny // (ny // F + 1)` cells a side), all patches of a grid in one call.

```
MIOpenDriver conv -n <patches> -c 24 -H <ly> -W <lx> -k 24 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 4 -t 1 -V 0 -i 10
```

```
--- ratio 16 framesizemax  512:   2 patches of  376 x  281  ConvBinWinogradRxSf2x3g1 3.868890 ms
--- ratio 16 framesizemax 1024:   1 patches of  376 x  563  ConvBinWinogradRxSf2x3g1 5.694773 ms
--- ratio  8 framesizemax  512:   6 patches of  376 x  375  ConvBinWinogradRxSf2x3g1 11.231956 ms
--- ratio  8 framesizemax 1024:   2 patches of  752 x  563  ConvBinWinogradRxSf2x3g1 14.938568 ms
--- ratio  4 framesizemax  512:  15 patches of  501 x  451  ConvBinWinogradRxSf2x3 39.372810 ms
--- ratio  4 framesizemax 1024:   6 patches of  752 x  751  ConvBinWinogradRxSf2x3 44.095566 ms
--- ratio  3 framesizemax  512:  24 patches of  501 x  501  GemmWrwUniversal 239.828827 ms
--- ratio  3 framesizemax 1024:   6 patches of 1003 x 1002  GemmWrwUniversal 237.508133 ms
--- ratio  2 framesizemax  512:  54 patches of  501 x  501  GemmWrwUniversal 539.915161 ms
--- ratio  2 framesizemax 1024:  15 patches of 1003 x  902  GemmWrwUniversal 564.834717 ms
```

## A6. The OpenCL solvers forced, with both debug switches

`MIOPEN_DEBUG_CONV_DIRECT_OCL_WRW2=1 MIOPEN_DEBUG_ENABLE_DEPRECATED_SOLVERS=1`, each OpenCL weight-gradient solver forced with `-S`, on the full 400 m grid and on 1 × 24 × 1024 × 1024. (With the first switch alone every one of them is "not applicable".)

```
MIOPEN_DEBUG_CONV_DIRECT_OCL_WRW2=1 MIOPEN_DEBUG_ENABLE_DEPRECATED_SOLVERS=1 MIOpenDriver conv -n 1 -c 24 -H 1505 -W 2255 -k 24 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 4 -t 1 -V 0 -i 10 -S <solver>
```

`drv_oclwrw2_forced2.txt`:

```
--- ConvOclBwdWrW2NonTunable   MIOpen Error: ocl/convolutionocl.cpp:1232: The supplied solution id: ConvOclBwdWrW2NonTunable is not applicable to the current problem
--- ConvOclBwdWrW2<1>          MIOpen Error: ocl/convolutionocl.cpp:1232: The supplied solution id: ConvOclBwdWrW2<1> is not applicable to the current problem
--- ConvOclBwdWrW2<2>          MIOpen Error: ocl/convolutionocl.cpp:1232: The supplied solution id: ConvOclBwdWrW2<2> is not applicable to the current problem
--- ConvOclBwdWrW2<4>          MIOpen Error: ocl/convolutionocl.cpp:1232: The supplied solution id: ConvOclBwdWrW2<4> is not applicable to the current problem
--- ConvOclBwdWrW2<8>          MIOpen Error: ocl/convolutionocl.cpp:1232: The supplied solution id: ConvOclBwdWrW2<8> is not applicable to the current problem
--- ConvOclBwdWrW2<16>         MIOpen Error: ocl/convolutionocl.cpp:1232: The supplied solution id: ConvOclBwdWrW2<16> is not applicable to the current problem
--- ConvOclBwdWrW53            ConvOclBwdWrW53 18.341331 ms
```

`drv_oclwrw2_forced2_1k.txt`:

```
--- ConvOclBwdWrW2NonTunable   ConvOclBwdWrW2NonTunable 12.407665 ms
--- ConvOclBwdWrW2<1>          MIOpen Error: ocl/convolutionocl.cpp:1232: The supplied solution id: ConvOclBwdWrW2<1> is not applicable to the current problem
--- ConvOclBwdWrW2<2>          MIOpen Error: ocl/convolutionocl.cpp:1232: The supplied solution id: ConvOclBwdWrW2<2> is not applicable to the current problem
--- ConvOclBwdWrW2<4>          MIOpen Error: ocl/convolutionocl.cpp:1232: The supplied solution id: ConvOclBwdWrW2<4> is not applicable to the current problem
--- ConvOclBwdWrW2<8>          MIOpen Error: ocl/convolutionocl.cpp:1232: The supplied solution id: ConvOclBwdWrW2<8> is not applicable to the current problem
--- ConvOclBwdWrW2<16>         MIOpen Error: ocl/convolutionocl.cpp:1232: The supplied solution id: ConvOclBwdWrW2<16> is not applicable to the current problem
--- ConvOclBwdWrW53            ConvOclBwdWrW53 2.277072 ms
```

## A7. ConvOclBwdWrW53 verification

`ConvOclBwdWrW53` forced and verified (`-V 1`) against MIOpen's GPU reference kernel on 15 shapes (N × C × H × W), then, on the shapes where IGM produced finite results, against the driver's CPU reference (`MIOPEN_DRIVER_USE_GPU_REFERENCE=0`), with the naive and Winograd solvers on the same shapes as controls.

```
MIOPEN_DEBUG_CONV_DIRECT_OCL_WRW2=1 MIOPEN_DEBUG_ENABLE_DEPRECATED_SOLVERS=1 [MIOPEN_DRIVER_USE_GPU_REFERENCE=0] MIOpenDriver conv -n <N> -c <C> -H <H> -W <W> -k 24 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F 4 -t 1 -V 1 -i 1 -S <solver>
```

`drv_oclwrw53_verify.txt`:

```
---  1 x 24 x  376 x  563  1.597406 ms FAILED: nan > 3e-06
---  1 x 24 x  752 x 1127  7.444191 ms FAILED: -nan > 3e-06
---  1 x 24 x 1024 x 1024  2.055008 ms OK (2.36071e-07 < 3e-06)
---  1 x 24 x 1505 x 2255  18.238634 ms FAILED: nan > 3e-06
---  1 x 24 x 2006 x 3006  24.511620 ms FAILED: -nan > 3e-06
---  3 x 24 x  501 x  501  4.360656 ms OK (1.23155e-07 < 3e-06)
--- 15 x 24 x  501 x  451  26.573988 ms OK (8.95778e-08 < 3e-06)
---  6 x 24 x  752 x  751  15.313863 ms FAILED: -nan > 3e-06
---  1 x 24 x 1200 x 1200  5.044741 ms OK (2.47924e-07 < 3e-06)
---  1 x 24 x 1400 x 1400  6.326786 ms FAILED: -nan > 3e-06
---  1 x 24 x 1600 x 1600  8.227794 ms FAILED: nan > 3e-06
---  1 x 24 x 1505 x 1505  11.591608 ms FAILED: nan > 3e-06
---  1 x 24 x 1024 x 2255  12.298851 ms OK (2.30695e-07 < 3e-06)
---  1 x 48 x 1024 x 1024  12.198011 ms OK (2.40737e-07 < 3e-06)
---  1 x 48 x  752 x 1127  21.172367 ms FAILED: -nan > 3e-06
```

`drv_oclwrw53_cpuref.txt`:

```
--- ConvOclBwdWrW53           1 x  376 x  563  1.265007 ms FAILED: nan > 3e-06
--- ConvDirectNaiveConvWrw    1 x  376 x  563  49.116241 ms OK vs CPU reference (0 < 3e-06)
--- ConvBinWinogradRxSf2x3g1  1 x  376 x  563  5.526947 ms OK vs CPU reference (2.31312e-06 < 3e-05)
--- ConvOclBwdWrW53           1 x 1024 x 1024  2.112528 ms OK vs CPU reference (2.36071e-07 < 3e-06)
--- ConvOclBwdWrW53           1 x  752 x 1127  5.337580 ms FAILED: -nan > 3e-06
--- ConvDirectNaiveConvWrw    1 x  752 x 1127  128.202759 ms OK vs CPU reference (0 < 3e-06)
```

## A8. Whole 200 m grid, forward and ∂input

The forward (`-F 1`) and backward-data (`-F 2`) convolutions over the whole 200 m grid (1 × 24 × 3010 × 4510), default find, `MIOPEN_LOG_LEVEL=5`: the algorithms find timed, their work-buffer sizes, and the one chosen. GEMM is applicable with an 11.7 GB buffer, Winograd is chosen with none.

```
MIOPEN_LOG_LEVEL=5 MIOpenDriver conv -n 1 -c 24 -H 3010 -W 4510 -k 24 -y 3 -x 3 -p 1 -q 1 -u 1 -v 1 -l 1 -j 1 -m conv -g 1 -F <1|2> -t 1 -i 1 -V 0
```

```
--- -F 1  miopenConvolutionFwdAlgoWinograd	4.78247	0 miopenConvolutionFwdAlgoGEMM	49.3853	11728886400 FW Chosen Algorithm: ConvBinWinogradRxSf2x3g1 , 0, 4.78247 Forward Conv. Elapsed: 4.393379 ms (average)
--- -F 2  miopenConvolutionBwdDataAlgoWinograd	4.88809	0 miopenConvolutionBwdDataAlgoGEMM	59.0704	11728886400 BWD Chosen Algorithm: ConvBinWinogradRxSf2x3g1 , 0, 4.88809 Backward Data Conv. Elapsed: 4.694382 ms (average)
```

## B. Experiments kept in `experiments/`

Runs of the variants that are not the recommended configuration, as `results.csv` (columns as in the README) or
`ceilings.json`. The notes below were the `meta.json` of each run.

### `experiments/find_mode_1/`

scripts/ceilings.py --nchw with MIOPEN_FIND_MODE=1 (NORMAL find): same as the default find mode; the *_nchw_* entries are the raw NCHW convolutions, without TensorFlow's layout transposes.

- files `ceilings.json`

### `experiments/nchw_offline/`

offline_f32 with the dahunet CNN rebuilt with channels_first Conv2D layers (one transpose in, one out; same weights, checked equal to the original on the GPU). Times identical to the default NHWC model: TensorFlow's layout optimizer already converts the convolution stack in graph mode. The variant is therefore not kept in the repository.

- date 2026-11-10T08:04
- cases offline_f32; ratios 16, 8, 4, 2
- env `IGM_BENCH_NCHW=1`
- files `results.csv`

### `experiments/nchw_online_patches_one_batch/`

online_f32, channels_first model + patches in one batch: identical to experiments/patches_one_batch (NCHW changes nothing).

- date 2026-11-10T08:13
- cases online_f32; ratios 16, 8, 4, 3
- igm_args `processes.iceflow.unified.data_preparation.framesizemax=512 processes.iceflow.unified.data_preparation.patches_per_batch=99`
- env `IGM_BENCH_NCHW=1`
- files `results.csv`

### `experiments/oclwrw53/`

online_f32 with MIOpen's ConvOclBwdWrW53 weight-gradient kernel enabled (MIOPEN_DEBUG_CONV_DIRECT_OCL_WRW2=1 MIOPEN_DEBUG_ENABLE_DEPRECATED_SOLVERS=1 MIOPEN_FIND_MODE=1). INCORRECT RESULTS: the kernel returns NaN on most IGM shapes (results/rx_7900_xtx/miopen/drv_oclwrw53_verify.txt, drv_oclwrw53_cpuref.txt); the ice volume is NaN from step 13 at 400 m. The times only show what a working kernel would give. 300/200 m abort on the 4 GiB work-buffer cap.

- date 2026-11-10T08:25
- cases online_f32; ratios 16, 8, 4, 3, 2
- env `MIOPEN_FIND_MODE=1 MIOPEN_DEBUG_CONV_DIRECT_OCL_WRW2=1 MIOPEN_DEBUG_ENABLE_DEPRECATED_SOLVERS=1`
- files `ceilings.json`, `results.csv`

### `experiments/patches_one_batch/`

online_f32, framesizemax=512 with all patches in one batch (patches_per_batch=99): 400 m runs out of memory (a 7.5 GB XLA buffer), hence the mini-batches of 3 in results/rx_7900_xtx_patches.

- date 2026-11-10T08:01
- cases online_f32; ratios 16, 8, 4, 3
- igm_args `processes.iceflow.unified.data_preparation.framesizemax=512 processes.iceflow.unified.data_preparation.patches_per_batch=99`
- files `results.csv`
