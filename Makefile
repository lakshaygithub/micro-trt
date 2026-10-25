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

.PHONY: all clean run run-fusion run-softmax run-mlp compare qwen-files check-qwen

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

# ---- Qwen2.5-0.5B ----------------------------------------------------------
# Export the weights and record Hugging Face's activations (needs torch +
# transformers; ~2.5 GB written to models/), then check the CPU reference
# forward pass against them layer by layer.
QWEN_MODEL ?= Qwen/Qwen2.5-0.5B

qwen-files:
	python3 tools/export_qwen.py $(QWEN_MODEL) models/qwen
	python3 tools/dump_reference.py $(QWEN_MODEL) models/ref

# Host-only: builds and runs without a GPU.
check_qwen: src/weights.cpp tests/check_qwen.cpp include/weights.h
	$(NVCC) -std=c++17 -O3 -Iinclude src/weights.cpp tests/check_qwen.cpp -o $@

check-qwen: check_qwen
	./check_qwen models/qwen models/ref

clean:
	rm -f $(BINS) check_qwen mlp_dump.bin