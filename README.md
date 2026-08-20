# Mini Inference Engine (C++/CUDA)

I wanted to understand what actually happens under vLLM and TensorRT, so I
built a small inference engine from scratch: hand-written CUDA kernels, a
pinned-memory pool, dual streams, and a multithreaded dynamic batcher.
Everything runs end-to-end in Colab on a T4.

## Results (T4, FP32)

| Metric | Result |
|---|---:|
| Optimized GEMM | 0.94 TFLOP/s (30% of cuBLAS) |
| V1 → V2 kernel uplift | 1.71× |
| Two-stream copy/compute overlap | 2.53× |
| Batching throughput | 119K req/s |
| p50 / p99 latency | 0.31 ms / 0.83 ms |

## What I built

- **Two GEMM kernels.** V1 is a standard shared-memory tiled kernel. V2 adds
  2×2 register tiling and float4 loads, which made it 1.7× faster while
  *dropping* occupancy from 95% to 79%. That surprised me at first, and it
  ended up being the most interesting lesson in the project (more below).
- **A fused LayerNorm** using warp-shuffle reductions. Nsight shows it's
  memory-bound (~52 GB/s DRAM), the opposite profile of GEMM.
- **A pinned-memory pool** so hot-path requests reuse buffers instead of
  hitting cudaMallocHost. 1,320 uses, 4 allocations.
- **Dual CUDA streams** overlapping copies with compute.
- **A dynamic batcher**: worker threads and condition variables collecting
  requests into near-full batches (31.6/32 average) before dispatch.

## Run it

Open the notebook in Colab with a T4 runtime and hit Run All. It compiles
with nvcc, verifies correctness against a CPU reference (max error < 5e-3),
benchmarks, and profiles with Nsight Compute if the runtime has it.

## What I learned

Occupancy isn't performance. V2 keeps fewer warps resident but gives each
thread more registers and more independent work, so the SM stays busier
doing math instead of waiting on memory. Chasing the occupancy number would
have made the kernel slower.
