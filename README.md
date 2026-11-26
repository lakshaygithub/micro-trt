# micro-trt

A minimal CUDA inference engine written from scratch in C++17 — hand-written kernels, no cuBLAS, no cuDNN.

It runs Qwen2.5-0.5B, a 494M-parameter language model, end to end on its own kernels. Its greedy output matches Hugging Face token for token, with the weights stored in fp32 or fp16. With fp16 weights it decodes 170 tokens/s on a Tesla T4, 1.63× its fp32 speed; in fp32 it decodes 199 tokens/s on an A100. Each stage is optimized against the GPU memory hierarchy and measured honestly along the way.

## Qwen2.5-0.5B, end to end

micro-trt runs a real language model on its own kernels: Qwen2.5-0.5B, 494M parameters, 24 transformer layers with RMSNorm, rotary position embeddings, grouped-query attention over a KV cache, and SwiGLU MLPs. The weights are exported from Hugging Face once (`tools/export_qwen.py`); from then on the C++ engine never touches PyTorch.

### Correctness against Hugging Face

Both implementations are compared with Hugging Face on a 7-token prompt at 37 checkpoints: the embedding output, every intermediate inside layer 0, all 24 layer outputs, the final norm and the logits.

| Implementation | Worst relative error | Greedy tokens matching Hugging Face |
|---|---:|---:|
| CPU reference, `tests/check_qwen.cpp` | 6.3e-05 | first token |
| GPU engine on a T4, `tests/check_qwen_gpu.cu`, fp32 weights | 4.4e-05 | 32 / 32 |
| GPU engine on a T4, fp16 weights | 3.0e-05 | 32 / 32 |
| GPU engine on an A100, fp32 weights | 3.3e-05 | 32 / 32 |

Relative error is max |difference| ÷ max |reference| per tensor, and the pass threshold is 1e-3. That threshold was set from both sides. Correct code that only sums in a different order stays below 1e-4. Four real architecture bugs planted in the CPU reference (RoPE pairing adjacent dimensions instead of i and i + d/2, the wrong query-to-KV head mapping, a dropped key bias, attention scaled by 1/d instead of 1/√d) each miss by 45% to 99.7% at the first checkpoint they reach, and each changes the predicted token.

### Decode: a GEMV instead of a GEMM

Text is generated one token at a time, so during decode every projection multiplies a single row by a weight matrix. The register-tiled GEMM handles that correctly, but its 128-row tiles do 128 rows of work for the one row that exists. `src/gemv.cu` is built for this shape instead:

| GPU (peak bandwidth) | Decode path | ms / token | tokens / s | Weight bandwidth | % of peak | Spread |
|---|---|---:|---:|---:|---:|---:|
| Tesla T4 (320 GB/s) | Register-tiled GEMM | 73.61 | 13.6 | 26.8 GB/s | 8.4% | 0.8% |
| Tesla T4 (320 GB/s) | GEMV | 9.50 | 105.3 | 208.0 GB/s | 65.0% | 0.1% |
| A100 40 GB (1,555 GB/s) | Register-tiled GEMM | 71.36 | 14.0 | 27.7 GB/s | 1.8% | 0.0% |
| A100 40 GB (1,555 GB/s) | GEMV | 5.02 | 199.3 | 393.8 GB/s | 25.3% | 0.0% |

A 7-token prompt then 64 decode steps, median of 7 reps after a 2 s warm-up (`make run-qwen`). The GEMV path is **7.75× faster on the T4 and 14.2× faster on the A100**. On both GPUs, one decode step computed both ways agrees to 1.5e-06.

**The GEMM path is limited by padding, spread over too few blocks.** A decode step needs 0.99 GFLOP of useful arithmetic: one multiply-add for each of 494M weights. The GEMM performs 128 times that, about 126 GFLOP per token, of which 127/128 multiplies zeros. A one-row GEMM also launches only one block per 128 output columns: 7 for the 896-wide projections and 1 for the 128-wide keys and values. That is why the A100, with 2.7 times the T4's SMs and 4.9 times its bandwidth, runs this path barely faster: 71.4 ms against 73.6.

**The GEMV is memory-bound by design.** It has no padding and nothing to reuse, since each weight is read once and used for exactly one multiply-add, so its speed is how fast the weights stream in. Each block owns 32 output columns. Eight threads cover them with one 16-byte `float4` load each, a full 128-byte line per row, and the block's 256 threads keep 32 rows in flight. Partial sums are combined in shared memory in a fixed order, so the output is identical from run to run.

**The ceiling is set by bandwidth.** Every decode step must read all 1.98 GB of fp32 weights: at least 6.2 ms on the T4 (162 tokens/s) and 1.3 ms on the A100 (787 tokens/s), however good the kernels are. The GEMV reaches 65% of the T4's ceiling but only 25% of the A100's.

**Where the time goes, measured.** On both GPUs each step takes about 3.3 to 3.7 ms longer than streaming the weights at peak would. Earlier versions of this README guessed the cause was fixed per-step overhead (kernel launches and a synchronous token upload), first for both GPUs and then for the A100. Nsight Systems profiles (`make profile-qwen`) say otherwise for both:

| Per decode step | Tesla T4 | A100 |
|---|---:|---:|
| GEMV kernels (169 per step) | 8.27 ms (86%) | 3.37 ms (65%) |
| Attention | 0.58 ms (6%) | 0.74 ms (14%) |
| RMSNorm, RoPE, residual adds, SwiGLU, KV-cache copies | 0.45 ms (5%) | 0.60 ms (12%) |
| GPU idle | 0.30 ms (3%) | 0.44 ms (9%) |
| Whole step | 9.60 ms | 5.15 ms |

The GPU is busy 97% of each step on the T4 and 91% on the A100, so launch overhead is not the main limit on either: the CPU issues a step's 363 kernel launches in 2.4 to 2.9 ms, then waits. What differs is how well the GPU's time is spent.

- **On the T4, the GEMV is the step.** It moves the weights at 239 GB/s, 75% of peak, and everything else is small.
- **On the A100, the GEMV reaches only 586 GB/s, 38% of peak, because it cannot fill the GPU.** The LM head, one huge matrix, streams at about 1.4 TB/s, 89% of peak. The narrow projections launch too few blocks for 108 SMs: the key and value GEMVs move under 50 GB/s.
- **Attention and the small kernels do not speed up at all.** They are limited by latency rather than bandwidth: decode attention runs one warp per head, and each RMSNorm a single warp. They take a little longer on the A100 than on the T4, whose clock is faster, and grow to 26% of an A100 step.

That is why the gap stayed about the same size across GPUs: most of it is GPU work limited by parallelism or latency, which more bandwidth does not speed up. The fixes differ by GPU. FP16 weights halve the bytes behind 86% of a T4 step. The A100 needs its work spread wider: fused Q/K/V and gate/up projections, narrow GEMVs split along K across more blocks, and decode attention spread over more than one warp per head, with CUDA graphs for the last few percent.

### FP16 weights

On the T4 the profile left one dominant cost: streaming the weights, 86% of a step at 75% of peak bandwidth. The bandwidth was already close to its practical limit, so the lever left was the number of bytes. The projection weights and the LM head are now stored in fp16, 2 bytes each instead of 4 (`HalfTensor`, rounded once on the GPU while the model loads). Every kernel widens them back to fp32 in registers, so all arithmetic is still fp32 and only the storage shrank. The GEMV reads eight fp16 weights per 16-byte load instead of four floats, and the tiled GEMM widens fp16 weights as it stages them into shared memory, so prefill reads the same copy.

**It costs no accuracy on this model.** Qwen2.5 ships in bf16, and fp16 represents every bf16 value in its normal range exactly. Converting the 494M projection weights changes 120,769 of them (0.02%), all smaller than fp16's smallest normal number (6.1e-5), and none by more than 3e-8. With fp16 weights the GPU engine matches Hugging Face to 3.0e-05 at worst, against 4.4e-05 in fp32, and still generates all 32 tokens identically.

| Tesla T4 | Weights | ms / token | tokens / s | Weight bandwidth | % of peak | Spread |
|---|---|---:|---:|---:|---:|---:|
| Register-tiled GEMM | fp32 | 73.95 | 13.5 | 26.7 GB/s | 8.3% | 1.0% |
| Register-tiled GEMM | fp16 | 80.74 | 12.4 | 12.2 GB/s | 3.8% | 0.4% |
| GEMV | fp32 | 9.59 | 104.3 | 206.0 GB/s | 64.4% | 0.1% |
| GEMV | fp16 | 5.89 | **169.8** | 167.7 GB/s | 52.4% | 0.3% |

All four rows come from one session (`make run-qwen`). fp16 weights make decode **1.63× faster**. Their ceiling on the T4 is 324 tokens/s: 0.99 GB per step at 320 GB/s.

**The GEMM path got slower, as a path limited by arithmetic should.** It moves 27 GB/s on a 320 GB/s bus, so its speed was never set by bytes. Halving them saves nothing, and converting each weight back to fp32 adds work: 8% slower.

**The first fp16 GEMV fell short, and the profile showed why.** I predicted 1.6 to 1.7×; the first version measured 1.47×. `make profile-qwen` now ends with `tools/gemv_breakdown.py`, which labels every GEMV launch by its projection and reports the bandwidth each one reaches:

| Projection (K × N) | fp32 | fp16, first version | fp16, final |
|---|---:|---:|---:|
| LM head (896 × 151936) | 73% | 57% | 73% |
| gate, up (896 × 4864) | 77% | 65–66% | 76% |
| down (4864 × 896) | 81% | 72% | 71% |
| q, o (896 × 896) | 69–71% | 55–57% | 53–55% |
| k, v (896 × 128) | 20% | 14% | 13% |

Share of the T4's peak bandwidth reached per launch. Each block of the GEMV streams a strip of 32 output columns: 128 bytes of every row in fp32, but only 64 in fp16, and the big matrices streamed those smaller reads less efficiently. The LM head suffered most, because each of its thousands of short-lived blocks now did half the work. fp16 now uses 64-column strips, 128-byte rows like fp32, wherever that still leaves every SM at least one block: on the T4, gate, up and the LM head. Those matrices returned to fp32's rate, each launch now 2× faster than in fp32, and decode reached 169.8 tokens/s, against a prediction of about 170.

**What is left.** The GEMVs as a whole are now 1.86× faster than in fp32, but a step is only 1.63× faster, because two parts do not scale with bytes. The narrow projections (q, o, k, v) gain just 1.3 to 1.55×: they launch too few blocks to reach full bandwidth, so much of their time is fixed per launch. And attention, the norms and the other small kernels, about 1.5 ms per step, read no weights at all. fp16 leaves them untouched, and they are now a quarter of a step. Both point at the same next steps as before: fewer, larger launches (fused projections, split-K GEMVs) and CUDA graphs.

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

### Fused epilogue: Y = ReLU(A·B + bias)

A fully connected layer applies a bias and an activation to the GEMM output. Done as separate kernels, each of those passes reads the entire output matrix from global memory and writes it back while performing one add or one comparison per element — pure memory traffic for negligible arithmetic.

The GEMM already holds each output value in a register immediately before storing it. Applying the bias and activation *there* costs no additional global traffic at all.

Measured on T4, M=N=1024, median of 15 reps. Output is 4 MB in every row, so fusion removes the same 8 MB of traffic each time; only the cost of the multiply changes:

| K | Arithmetic | Unfused (ms) | Fused (ms) | Speedup |
|---:|---:|---:|---:|---:|
| 1024 | 2.15 GFLOP | 0.8048 | 0.7824 | 1.03× |
| 256 | 0.54 GFLOP | 0.2625 | 0.2340 | 1.12× |
| 64 | 0.13 GFLOP | 0.1197 | 0.0957 | **1.25×** |
| 16 | 0.03 GFLOP | 0.0792 | 0.0638 | 1.24× |

Fusion eliminates traffic proportional to the output (M·N) while the GEMM costs M·N·K, so its share of total runtime scales roughly as 1/K. On square matrices it is a rounding error; on the shallow layers and small batches that dominate real inference it is worth a quarter of the runtime.

The gain plateaus near 1.25× rather than continuing to climb, because both pipelines must still write the 4 MB output once. That write is irreducible, and it sets the floor.

These ratios reproduce across independent sessions to within 0.02× (1.03/1.12/1.25/1.24 and 1.04/1.12/1.24/1.22 on two separate runs).

Worth noting the implied-bandwidth figures the benchmark reports: several exceed the T4's 320 GB/s DRAM bandwidth, which means the output was still resident in the 4 MB L2 when the second pass read it. Where that happens the eliminated kernel never reached DRAM, so these numbers understate what fusion saves on outputs too large to cache.

### Implementation: templated epilogue

The fused and unfused kernels are the same kernel. The write-back is templated over an epilogue functor:

```cpp
struct Identity {
    __device__ __forceinline__ float operator()(float v, int) const { return v; }
};

struct BiasRelu {
    const float* __restrict__ bias;
    __device__ __forceinline__ float operator()(float v, int col) const {
        const float s = v + bias[col];
        return s > 0.0f ? s : 0.0f;
    }
};
```

`gemm_regtiled` instantiates it with `Identity`, `gemm_bias_relu_fused` with `BiasRelu`. Each instantiation compiles to a separate kernel with the epilogue inlined, so there is no runtime branch and no duplicated kernel body — adding another activation costs a six-line struct. This is the same approach CUTLASS takes, for the same reason.

### End-to-end forward pass, against PyTorch

The kernels above compose into a working network. A four-layer classifier &mdash; three hidden layers with fused bias and ReLU, an output layer with bias only, then row-wise softmax:

```
1024 -> 1024 -> 1024 -> 1024 -> 1000,  batch 1024,  8.54 GFLOP per pass
```

Both implementations run the identical network with identical weights on the same T4, timed with CUDA events, median of 15 reps after a wall-clock warm-up:

| | Median (ms) | GFLOP/s | Samples/s | vs. PyTorch |
|---|---:|---:|---:|---:|
| micro-trt | 3.117 | 2740 | 328,576 | **76.0%** |
| PyTorch 2.11 (cuBLAS) | 2.369 | 3604 | 432,188 | 1.00× |

Outputs agree to a worst relative difference of **8.11e-06** &mdash; fp32 rounding noise. PyTorch shares no code with this project, so that agreement is independent evidence rather than two implementations repeating the same mistake.

**No new kernels were needed for this stage.** The only addition was a six-line `BiasOnly` epilogue struct for the output layer, where applying ReLU would clamp away the negative logits softmax depends on. That is the templated-epilogue design from the fusion stage paying off: a new variant costs one instantiation rather than a duplicated kernel.

**The network is GEMM-dominated.** The full forward pass sustains 2740 GFLOP/s while the isolated GEMM measured 2708 &mdash; the fused epilogues and softmax add essentially nothing to total runtime, which is the intended outcome.

**Where the remaining 24% goes.** cuBLAS is tuned at the assembly level, ships dozens of GEMM kernels and selects one per problem shape, and uses double buffering and vectorised loads that this implementation does not. Note that tensor cores are *not* part of the gap on this hardware: Turing's tensor cores support FP16 and INT8 only, and TF32 arrived with Ampere, so cuBLAS is restricted to the same FP32 units used here.

Worth noting that PyTorch itself reaches only 44% of the T4's 8140 GFLOP/s theoretical peak on this workload, so the practical ceiling for FP32 GEMM on this card is well below the specification figure.

### Softmax: parallel reduction

Row-wise softmax is the first operation here where threads must cooperate — normalizing a row requires its maximum and its sum, and no single thread has seen the whole row. It is also strongly memory-bound, so the meaningful metric is achieved bandwidth rather than FLOP/s.

Three implementations, T4, median of 15 reps:

| Kernel | N=1024 | N=4096 | N=16384 | N=65536 |
|---|---:|---:|---:|---:|
| `softmax_naive` — one thread per row | 10.36 | 10.27 | 10.25 | 10.22 |
| `softmax_block` — one block per row, warp-shuffle reduction | **227.48** | 173.52 | 119.91 | 115.94 |
| `softmax_online` — single-pass online reduction | 227.02 | **177.16** | **155.70** | **150.79** |

Figures are GB/s against a 320 GB/s peak, M=1024 throughout. All three kernels use the same `expf`, so the comparison isolates the algorithm rather than the choice of exponential.

**The naive kernel is flat at ~10 GB/s regardless of size — 3% of peak.** With one thread per row, the 32 threads of a warp read addresses N floats apart, so each access becomes its own memory transaction instead of coalescing into one. That penalty is independent of problem size, which is why the number never moves. Switching to one block per row, with threads striding across it, makes access contiguous and is worth **up to 21×**.

**Both cooperative kernels reach ~70% of peak on small rows**, which is close to the practical ceiling for a memory-bound kernel.

### Why the online version only wins on large rows

The online algorithm computes the row max and row sum in a single pass, by rescaling the running sum whenever the maximum changes:

```
m' = max(m, x)
s' = s·exp(m - m') + exp(x - m')
```

The rule is associative, so it also serves as the combining operation inside the warp-shuffle reduction. It eliminates one of three passes over the input — but the measured benefit is **exactly zero at N=1024 and 30% at N=65536**.

The reason is cache residency. At N=1024 a row is 4 KB, so the second and third passes read data still sitting in L1/L2; the pass being eliminated never cost DRAM traffic in the first place. By N=16384 a row is 64 KB, the aggregate working set across concurrent blocks exceeds the 4 MB L2, and the re-reads have to come from memory.

Accounting for actual passes (three reads plus one write versus two reads plus one write) at N=65536:

| Kernel | Real DRAM traffic | Time | Achieved bandwidth |
|---|---:|---:|---:|
| `softmax_block` | 1024 MB | 4.63 ms | 232 GB/s |
| `softmax_online` | 768 MB | 3.56 ms | 226 GB/s |

Both kernels are pinned at the same physical bandwidth limit. The online version is not faster per byte — it simply moves fewer bytes. The 4:3 traffic ratio predicts a 33% advantage; the measured 30.1% is short of that by roughly the amount of caching still occurring.

An earlier version of this comparison used the fast `__expf` intrinsic in the online kernel and the accurate `expf` in the other two, which inflated the apparent small-row advantage to 1.7%. Making them consistent dropped it to zero and raised the large-row figure from 28.5% to 30.1% — sharper in both directions, which is what removing a confound from a real effect tends to look like.

This is the regime FlashAttention targets: the same recurrence, applied where the intermediate is far too large to cache.

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

Qwen2.5-0.5B needs PyTorch and `transformers` once, to export the weights and record Hugging Face's activations:

```bash
make qwen-files       # writes ~2.5 GB to models/
make check-qwen       # CPU reference vs Hugging Face (no GPU needed)
make check-qwen-gpu   # GPU engine vs Hugging Face, plus 32 greedy tokens: fp32, then fp16 weights
make run-qwen         # decode speed: GEMM and GEMV paths, each with fp32 and fp16 weights
make profile-qwen     # Nsight Systems profile of decode, then GEMV time per projection (bash tools/install_nsys.sh installs nsys)
```

## Roadmap

| Stage | Kernel | Status |
|---|---|---|
| 1 | Naive GEMM (global memory only) | done |
| 2 | Tiled GEMM (shared memory) | done |
| 3 | Register-blocked GEMM (8×8 outputs per thread) | done |
| 4 | Fused GEMM + bias + ReLU (single pass) | done |
| 5 | Softmax with parallel reduction (warp shuffles, online single-pass) | done |
| 6 | End-to-end MLP forward pass vs PyTorch | done |
| 7 | Qwen2.5-0.5B end to end: RMSNorm, RoPE, grouped-query attention, KV cache, SwiGLU, checked against Hugging Face | done |
| 8 | GEMV for single-token decode | done |
| 9 | Profile decode with Nsight Systems | done |
| 10 | FP16 weights: halve the bytes every decode step reads | done |
| 11 | PyTorch custom op: call the GEMV from Python, tested against `torch.matmul` | next |
| 12 | Triton GEMV, benchmarked against the CUDA GEMV from the same Python script | planned |
| 13 | Fill bigger GPUs: fused projections, split-K GEMV, multi-warp decode attention | planned |
| 14 | CUDA graphs to remove the remaining launch gaps | planned |
| 15 | FlashAttention-style attention for prefill | planned |
| 16 | INT8 weight-only quantization | planned |
| 17 | Benchmarks against llama.cpp and vLLM; serving with continuous batching | planned |

## Design notes

**`Tensor` owns its device memory.** Construction calls `cudaMalloc`, destruction calls `cudaFree`, so GPU memory cannot leak on an early return. The type is move-only: a deep GPU copy is expensive enough that it should be an explicit act rather than something that happens because a value was passed by accident, so the copy operations are `= delete`d and an accidental copy becomes a compile error.

**The kernel API hides CUDA entirely.** `kernels.h` exposes `gemm_*(A, B, C)` — no block dimensions, no thread counts, no `cudaMemcpy`. Every optimization changes the implementation below that line without touching callers, and all kernels share one `GemmFn` signature so the harness can treat them interchangeably.

**Correctness precedes speed.** Every kernel is validated against a CPU reference with a relative tolerance before it is timed — bit-exact comparison would be wrong, since the GPU accumulates in a different order and floating-point addition is not associative. The harness returns non-zero if any kernel disagrees. Where a structural invariant exists it is checked separately: softmax rows must sum to 1, which catches reduction bugs an elementwise comparison against a similarly-written reference could miss.

**Numerical stability is a correctness requirement, not a tuning choice.** Softmax subtracts the row maximum before exponentiating because `exp(89)` already overflows fp32. The reduction's identity element is `-FLT_MAX` rather than `-INFINITY`, since combining two empty slots would otherwise evaluate `exp(-inf − (−inf))` and poison the result with `NaN`.

**Timing uses CUDA events, not wall-clock.** Kernel launches are asynchronous; a CPU timer around a launch measures enqueue time, not execution time.

**Benchmarks are like-for-like.** Every kernel runs through the same `evaluate()` path — same fixed seed, same input buffers, same warm-up, same iteration count. The kernel is the only variable.

## Known limitations

Known remaining inefficiencies in `gemm_regtiled`, roughly in order of expected payoff:

1. **Four-way shared-memory bank conflict** on the `Bs[k][tx*TN + j]` reads. With `TN=8`, threads 0, 4, 8 and 12 of a half-warp all map to the same bank. Fixable with vectorized loads or padding — and worth measuring, since the 8×8 configuration wins despite having worse conflicts than 4×4.
2. **No double buffering.** Tiles are loaded, then computed, then the next tile is loaded. Loads and compute never overlap. Prefetching tile *t+1* while computing tile *t* is the largest structural win remaining, at the cost of doubling shared memory use.
3. **No vectorized memory access.** `float4` loads move 16 bytes per instruction instead of 4, cutting instruction count on the load path.
4. **A-tile global loads take two transactions per warp** rather than one, because a warp spans two rows of a 16-wide tile.
5. **No tensor cores.** The T4 has them, but they require fp16 or TF32 and the `wmma` API — arguably a different kernel rather than an optimization of this one.

The end-to-end MLP above is compared against PyTorch, which calls cuBLAS; a kernel-by-kernel comparison with cuBLAS on the same shapes has not been run yet.

