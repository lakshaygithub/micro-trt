NVCC      ?= nvcc

# Target GPU architecture. Colab's free T4 is sm_75.
#   T4 = sm_75 | A100 = sm_80 | L4/L40S/RTX 40xx = sm_89 | H100 = sm_90
# Override on the command line:  make ARCH=sm_80
ARCH      ?= sm_75

# -lineinfo keeps source line mapping in the binary so Nsight Compute can point
# at the exact line responsible for a stall. Costs nothing at runtime.
NVCC_FLAGS := -std=c++17 -O3 -arch=$(ARCH) -Iinclude -lineinfo \
              -Wno-deprecated-gpu-targets

LIB_SRCS := src/tensor.cu src/gemm_naive.cu src/gemm_tiled.cu \
            src/gemm_regtiled.cu src/bias_relu.cu src/softmax.cu src/mlp.cu

BINS := bench_gemm bench_fusion bench_softmax bench_mlp

.PHONY: all clean run run-fusion run-softmax run-mlp compare

all: $(BINS)

bench_gemm: $(LIB_SRCS) bench/main.cu
	$(NVCC) $(NVCC_FLAGS) $(LIB_SRCS) bench/main.cu -o $@

bench_fusion: $(LIB_SRCS) bench/bench_fusion.cu
	$(NVCC) $(NVCC_FLAGS) $(LIB_SRCS) bench/bench_fusion.cu -o $@

bench_softmax: $(LIB_SRCS) bench/bench_softmax.cu
	$(NVCC) $(NVCC_FLAGS) $(LIB_SRCS) bench/bench_softmax.cu -o $@

bench_mlp: $(LIB_SRCS) bench/bench_mlp.cu
	$(NVCC) $(NVCC_FLAGS) $(LIB_SRCS) bench/bench_mlp.cu -o $@

run: bench_gemm
	./bench_gemm

run-fusion: bench_fusion
	./bench_fusion

run-softmax: bench_softmax
	./bench_softmax

run-mlp: bench_mlp
	./bench_mlp

# Runs the forward pass, then verifies it against PyTorch on the same GPU.
compare: bench_mlp
	./bench_mlp
	python3 tools/compare_pytorch.py

clean:
	rm -f $(BINS) mlp_dump.bin