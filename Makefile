NVCC      ?= nvcc

# Target GPU architecture. Colab's free T4 is sm_75.
#   T4 = sm_75 | A100 = sm_80 | L4/L40S/RTX 40xx = sm_89 | H100 = sm_90
# Override on the command line:  make ARCH=sm_80
ARCH      ?= sm_75

# -lineinfo keeps source line mapping in the binary so Nsight Compute can point
# at the exact line responsible for a stall. Costs nothing at runtime.
NVCC_FLAGS := -std=c++17 -O3 -arch=$(ARCH) -Iinclude -lineinfo \
              -Wno-deprecated-gpu-targets

SRCS := src/tensor.cu src/gemm_naive.cu src/gemm_tiled.cu bench/main.cu
BIN  := bench_gemm

.PHONY: all clean run

all: $(BIN)

$(BIN): $(SRCS)
	$(NVCC) $(NVCC_FLAGS) $(SRCS) -o $(BIN)

run: $(BIN)
	./$(BIN)

clean:
	rm -f $(BIN)