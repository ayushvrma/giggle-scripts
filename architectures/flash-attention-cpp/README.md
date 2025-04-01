# Flash Attention C++ Implementation

This is a C++ implementation of the Flash Attention algorithm using CUDA for efficient computation on NVIDIA GPUs.

## Overview

Flash Attention is an efficient attention algorithm that reduces memory usage and improves performance compared to standard attention implementations. This implementation includes:

- Efficient memory access patterns using tiling
- Numerically stable softmax computation
- Support for attention masks
- Optimized matrix multiplication
- CUDA kernel-based implementation

## Requirements

- CUDA-capable GPU (compute capability 6.0 or higher)
- CUDA Toolkit (version 11.0 or higher)
- CMake (version 3.10 or higher)
- C++17 compatible compiler

## Building

```bash
mkdir build
cd build
cmake ..
make
```

## Usage

```cpp
#include "flash_attention.hpp"

// Create Flash Attention instance
FlashAttention flash_attn(batch_size, num_heads, seq_len, head_dim);

// Compute attention
flash_attn.compute(q, k, v, output, mask);  // mask is optional
```

## Implementation Details

### Memory Layout

The input tensors are expected to be in the following format:
- Q: [batch_size, num_heads, seq_len, head_dim]
- K: [batch_size, num_heads, seq_len, head_dim]
- V: [batch_size, num_heads, seq_len, head_dim]
- Output: [batch_size, num_heads, seq_len, head_dim]
- Mask (optional): [batch_size, seq_len]

### Optimizations

1. **Tiling**: Uses shared memory tiles to reduce global memory access
2. **Numerical Stability**: Implements numerically stable softmax computation
3. **Memory Access**: Optimized memory access patterns for better cache utilization
4. **Parallelization**: Efficient thread and block organization for GPU execution

### CUDA Kernels

The implementation uses three main CUDA kernels:

1. `compute_attention_scores_kernel`: Computes attention scores between Q and K
2. `compute_softmax_kernel`: Applies softmax to attention scores
3. `compute_output_kernel`: Computes final output using attention scores and V

## Performance

The implementation is optimized for:
- Memory efficiency: O(N) memory usage instead of O(N²)
- Compute efficiency: Optimized CUDA kernels with efficient memory access patterns
- Numerical stability: Proper handling of softmax computation

## License

MIT License 