# micro-trt

A minimal CUDA inference engine written from scratch in C++17 — hand-written kernels, no cuBLAS, no cuDNN.

The goal is to run a neural network forward pass entirely on custom CUDA kernels, optimizing each stage against the GPU memory hierarchy and measuring the result honestly at every step.

## Results

Tesla T4 (SM 7.5, 40 SMs, 320 GB/s), fp32, 1024×1024×1024.
Median of 9 timing reps × 50 iterations, after a sustained warm-up to steady-state clocks.
All kernels validated against a CPU reference before timing.

| Kernel | Median (ms) | GFLOP/s | vs. naive | Spread | % of fp32 peak |
|---|---:|---:|---:|---:|---:|
| `gemm_naive` — one thread per output, global memory only | 4.354 | 493.21 | 1.00× | 1.6% | 6.1% |
| `gemm_tiled` — 32×32 shared-memory tiling | 2.340 | 917.83 | 1.86× | 1.6% | 11.3% |
| `gemm_regtiled` — 128×128×16 block tile, 8×8 outputs per thread | 0.793 | 2708.35 | 5.49× | 4.3% | 33.3% |

T4 fp32 peak is ~8.14 TFLOP/s (2560 cores × 2 flops × 1.59 GHz).

### Tile size

Tiled kernel at N=1024, all measurements within ~1.5% spread. Measured in a separate session from the table above, so absolute values differ by a few percent while the ratios hold:

| TILE | GFLOP/s | vs. naive | Shared mem/block |
|---:|---:|---:|---:|
| 8 | 513.14 | 1.11× | 0.5 KB |
| 16 | 749.47 | 1.62× | 2 KB |
| 32 | 870.33 | 1.91× | 8 KB |

Performance increases monotonically with tile size: a larger tile reuses each staged element more times, cutting global traffic by a factor of `TILE`. At 32 the block is 1024 threads and exactly one block is resident per SM, but the reuse gain still dominates the occupancy loss. **32 is the configuration used above.**

### Scaling with problem size

TILE=32, square matrices, same methodology:

All three kernels, square matrices, median of 15 reps:

| N | naive | tiled | regtiled | regtiled vs. naive | regtiled blocks launched | Working set | L2-resident? |
|---:|---:|---:|---:|---:|---:|---:|:---:|
| 256 | 556.81 | 766.25 | 427.69 | **0.77×** | 4 | 0.8 MB | yes |
| 512 | 555.39 | 957.92 | 1889.73 | 3.40× | 16 | 3.1 MB | yes |
| 1024 | 493.21 | 917.83 | 2708.35 | 5.49× | 64 | 12.6 MB | no |
| 2048 | 412.74 | 821.62 | 2786.40 | 6.75× | 256 | 50.3 MB | no |

Three separate effects show up here.

**The register-blocked kernel is the slowest of the three at N=256.** Its 128×128 block tile means only `(256/128)² = 4` blocks get launched onto a 40-SM GPU, leaving roughly 90% of the device idle. Large tiles need large problems; below a certain size the coarser decomposition costs more than the improved read ratio gains. The crossover sits between 256 and 512.

**The two memory-bound kernels peak at N=512, then decline.** The T4 has 4 MB of L2, and three fp32 matrices at N=512 occupy 3.1 MB — the largest size here that stays fully cached. Past that, naive and tiled both fall off as data starts coming from DRAM.

**The register-blocked kernel does not.** It rises monotonically all the way to 2048, because at roughly 82 GB/s of global traffic it is not memory-bound in the first place, so running out of cache costs it very little. The L2 cliff is visible only in the kernels that depend on L2.

Absolute GFLOP/s varies a few percent between sessions (different physical T4s, different thermal conditions). Speedup ratios measured within a single run are the stable quantity and reproduce to within about 0.05×.

### Why the naive kernel is slow

It issues `2·M·N·K` global loads — 2.15 billion — to read only ~2.1 million distinct values. Every element of A is re-read N times, every element of B M times.

At 4.374 ms that is ~1960 GB/s of *requested* bandwidth against a 320 GB/s memory bus, so roughly 84% of those requests are being served by L1/L2 rather than DRAM. The hardware absorbs most of the redundancy, which is why the kernel still reaches 6% of peak despite an arithmetic intensity far below what the ALUs need.

### What tiling changed

Staging tiles through shared memory cuts global loads by a factor of `TILE`, dropping requested global bandwidth to ~115 GB/s — comfortably inside the 320 GB/s bus. Global memory is no longer the constraint.

The 1.87× gain is smaller than the 32× traffic reduction because the naive kernel was never actually DRAM-bound; the cache was already absorbing most of its redundant traffic. Tiling replaces cache-serviced global requests with explicitly staged shared-memory reads, and the inner loop still performs two shared loads per fused multiply-add:

```cpp
acc += As[ty][k] * Bs[k][tx];
```

Shared memory delivers 32 words per cycle per SM against 64 FP32 cores, so a 2:1 load-to-FMA ratio caps throughput near a quarter of peak regardless of how little global traffic remains. The kernel is limited by shared-memory issue rate.

### What register blocking changed

Giving each thread an 8×8 patch of C instead of a single element attacks that ratio directly. An 8×8 patch needs only 8 values from A and 8 from B — every A value pairs with every B value, like a multiplication table — so 16 shared loads produce 64 multiply-adds. The ratio drops from 2.0 to 0.25.

Measured result: **2.95× over the tiled kernel**, reaching 33.3% of peak.

The gain is smaller than the 8× ratio improvement because, as at every previous stage, fixing one bottleneck exposes the next. Global traffic here is ~82 GB/s, far inside the bus, so DRAM is not the limit. The remaining candidates are a four-way shared-memory bank conflict on the B reads, instruction issue rate in the unrolled inner loop, and barrier overhead — distinguishing between them needs a profiler rather than arithmetic.

### Per-thread tile size, and why occupancy is the wrong target

Each thread computing more outputs means fewer shared-memory reads per multiply, but more registers per thread and so fewer thread blocks resident per SM. The conventional advice is to maximize occupancy. Measured at N=1024:

| TM×TN | Loads per multiply | Registers | Shared mem/block | Blocks/SM | Occupancy | GFLOP/s |
|---|---:|---:|---:|---:|---:|---:|
| 2×2 | 1.0 | 59 | 4 KB | 4 | 100% | 1146.94 |
| 4×4 | 0.5 | 63 | 8 KB | 4 | 100% | 2146.92 |
| 8×8 | 0.25 | 118 | 16 KB | 2 | 50% | 2620.62 |

Blocks per SM is set by the register budget: the T4 has 65,536 registers per SM, allocated in units of 8 per thread, against 256 threads per block. At 118 registers that rounds to 120, giving 30,720 per block and room for only 2.

The 8×8 configuration runs at **half the occupancy of 4×4 and is 22% faster**. Halving the memory traffic per unit of work more than compensates for halving the parallelism available to hide latency. Occupancy is a means, not the objective — throughput is.

All three configurations spill nothing, so this is a clean comparison of the tradeoff rather than a comparison against a broken kernel.

### The accumulators must stay in registers

The whole optimization depends on the `acc`, `regA` and `regB` arrays living in registers. Registers are not addressable at runtime, so this only happens if every index is resolved at compile time — which requires the loops over them to be fully unrolled.

Measured three ways at N=1024, using the 4×4 configuration:

| Inner loops | Stack frame | Registers | GFLOP/s |
|---|---:|---:|---:|
| `#pragma unroll` (as written) | 0 B | 63 | 2146.92 |
| No pragma | 0 B | 59 | 2178.36 |
| `#pragma unroll 1` (unrolling forced off) | 96 B | 44 | 127.53 |

Two things worth noting.

The pragmas are not what makes this work. `TM`, `TN` and `BK` are `constexpr`, so nvcc at `-O3` already unrolls these loops on its own — removing the pragmas changed nothing measurable. They document a requirement rather than cause an outcome, which still matters if the trip counts ever stop being compile-time constants.

When unrolling is suppressed, the three arrays move to local memory — 96 bytes of stack frame, exactly `16 + 4 + 4` floats — and throughput collapses by 17×, ending up roughly 4× *slower* than the naive kernel this was meant to improve on. Note that `spill stores/loads` stayed at zero throughout; the indicator here is `stack frame`, since these are arrays the compiler never placed in registers rather than registers it was forced to evict.

## Benchmark methodology

Early measurements in this project varied by up to 45% run-to-run on identical code. Diagnosing that produced two fixes worth documenting:

**Median of repeated trials, with spread reported.** A single measurement on a thermally-constrained cloud GPU is not evidence. The harness runs N independent timing reps and reports the median plus min/max, so a difference smaller than the measurement spread is visibly not a result.

**Sustained warm-up before any timing.** The original variance was not random noise — the median sat almost on top of the minimum with a long tail toward the maximum, the signature of a clock ramp rather than a symmetric distribution. GPUs idle at low clocks and boost under load, so early measurements were taken mid-ramp. Worse, whichever kernel ran first paid for the warm-up that later kernels benefited from, which biased every measured speedup in the same direction — a systematic error that averaging would not have removed. The harness now drives the device to steady state before measuring anything.

**Warm-up is measured in wall-clock time, not iterations.** A fixed iteration count warms for very different durations across problem sizes — 300 launches is ~20 ms at N=256 but ~700 ms at N=1024 — so a size sweep would measure its smallest cases mid-ramp while reporting that it had warmed up. This was caught when one sweep point showed 19.8% spread while its neighbours showed under 3%.

**Effect of the fix.** The naive kernel's spread fell from 42–50% to 1.9–2.8%, and its median became reproducible across three independent sessions at 464.04, 460.78 and 455.00 GFLOP/s — about 2% variation on identical code, against roughly 15% before. All figures also came down ~5% from the pre-warm-up measurements, because the T4 is a passively cooled 70 W card: the earlier numbers caught it boosted but not yet thermally saturated. The values reported here are sustained throughput rather than burst.

## Build

Requires an NVIDIA GPU and the CUDA toolkit.

```bash
make ARCH=sm_75              # T4=sm_75, A100=sm_80, RTX 40xx/L4=sm_89, H100=sm_90
./bench_gemm                 # defaults: 1024, 7 reps
./bench_gemm 2048 9          # size, reps
```

## Roadmap

| Stage | Kernel | Status |
|---|---|---|
| 1 | Naive GEMM (global memory only) | done |
| 2 | Tiled GEMM (shared memory) | done |
| 3 | Register-blocked GEMM (4×4 outputs per thread) | done |
| 4 | Fused GEMM + bias + ReLU (single pass) | planned |
| 5 | Softmax / LayerNorm | planned |
| 6 | End-to-end MLP forward pass vs PyTorch | planned |
| 7 | CUDA streams — overlap H2D transfer with compute | planned |

## Design notes

**`Tensor` owns its device memory.** Construction calls `cudaMalloc`, destruction calls `cudaFree`, so GPU memory cannot leak on an early return. The type is move-only: a deep GPU copy is expensive enough that it should be an explicit act rather than something that happens because a value was passed by accident, so the copy operations are `= delete`d and an accidental copy becomes a compile error.

**The kernel API hides CUDA entirely.** `kernels.h` exposes `gemm_*(A, B, C)` — no block dimensions, no thread counts, no `cudaMemcpy`. Every optimization changes the implementation below that line without touching callers, and all kernels share one `GemmFn` signature so the harness can treat them interchangeably.

**Correctness precedes speed.** Every kernel is validated against a CPU reference with a relative tolerance before it is timed — bit-exact comparison would be wrong, since the GPU accumulates in a different order and floating-point addition is not associative. The harness returns non-zero if any kernel disagrees.

**Timing uses CUDA events, not wall-clock.** Kernel launches are asynchronous; a CPU timer around a launch measures enqueue time, not execution time.

**Benchmarks are like-for-like.** Every kernel runs through the same `evaluate()` path — same fixed seed, same input buffers, same warm-up, same iteration count. The kernel is the only variable.

## Known limitations

Known remaining inefficiencies in `gemm_regtiled`, roughly in order of expected payoff:

1. **Four-way shared-memory bank conflict** on the `Bs[k][tx*TN + j]` reads. With `TN=8`, threads 0, 4, 8 and 12 of a half-warp all map to the same bank. Fixable with vectorized loads or padding — and worth measuring, since the 8×8 configuration wins despite having worse conflicts than 4×4.
2. **No double buffering.** Tiles are loaded, then computed, then the next tile is loaded. Loads and compute never overlap. Prefetching tile *t+1* while computing tile *t* is the largest structural win remaining, at the cost of doubling shared memory use.
3. **No vectorized memory access.** `float4` loads move 16 bytes per instruction instead of 4, cutting instruction count on the load path.
4. **A-tile global loads take two transactions per warp** rather than one, because a warp spans two rows of a 16-wide tile.
5. **No tensor cores.** The T4 has them, but they require fp16 or TF32 and the `wmma` API — arguably a different kernel rather than an optimization of this one.

This has not yet been benchmarked against cuBLAS on the same hardware, so no claim is made about relative performance.

## License

MIT
