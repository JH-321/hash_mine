NVCC ?= nvcc
TARGET ?= target/release/cuda-miner
A100_TARGET ?= target/release/cuda-miner-a100
H100_TARGET ?= target/release/cuda-miner-h100
RTX2080TI_TARGET ?= target/release/cuda-miner-2080ti

CUDA_ARCH_FLAGS ?= \
	-gencode arch=compute_75,code=sm_75 \
	-gencode arch=compute_80,code=sm_80 \
	-gencode arch=compute_90,code=sm_90 \
	-gencode arch=compute_90,code=compute_90

NVCCFLAGS ?= -O3 -std=c++17 --use_fast_math -lineinfo -Xptxas=-v

.PHONY: all a100 h100 2080ti clean

all: $(TARGET)

a100: $(A100_TARGET)

h100: $(H100_TARGET)

2080ti: $(RTX2080TI_TARGET)

$(TARGET): src/main.cu
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) $(CUDA_ARCH_FLAGS) $< -o $@

$(A100_TARGET): src/main.cu
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) -gencode arch=compute_80,code=sm_80 -gencode arch=compute_80,code=compute_80 $< -o $@

$(H100_TARGET): src/main.cu
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) -gencode arch=compute_90,code=sm_90 -gencode arch=compute_90,code=compute_90 $< -o $@

$(RTX2080TI_TARGET): src/main.cu
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) -gencode arch=compute_75,code=sm_75 -gencode arch=compute_75,code=compute_75 $< -o $@

clean:
	rm -rf target
