#include "flash_attention.hpp"
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// CUDA kernel configuration
constexpr int BLOCK_SIZE = 256;
constexpr int TILE_SIZE = 32;
constexpr int NUM_THREADS = 256;
constexpr int NUM_WARPS = NUM_THREADS / 32;

// Shared memory tile size for Q, K, V matrices
constexpr int TILE_Q = 32;
constexpr int TILE_K = 32;
constexpr int TILE_V = 32;

// Shared memory tile size for attention scores
constexpr int TILE_ATTN = 32;

// Helper function to compute the number of blocks needed
__device__ __forceinline__ int get_num_blocks(int size) {
    return (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
}

// Helper function to compute the number of tiles needed
__device__ __forceinline__ int get_num_tiles(int size) {
    return (size + TILE_SIZE - 1) / TILE_SIZE;
}

// Helper function to compute the scaling factor for attention scores
__device__ __forceinline__ float compute_scale(int head_dim) {
    return 1.0f / std::sqrt(static_cast<float>(head_dim));
}

// Helper function to compute softmax with numerical stability
__device__ __forceinline__ void compute_softmax(
    float* input,
    float* output,
    int size,
    float scale = 1.0f
) {
    // Find maximum value for numerical stability
    float max_val = input[0];
    for (int i = 1; i < size; ++i) {
        max_val = std::max(max_val, input[i]);
    }
    
    // Compute exp and sum
    float sum = 0.0f;
    for (int i = 0; i < size; ++i) {
        output[i] = std::exp((input[i] - max_val) * scale);
        sum += output[i];
    }
    
    // Normalize
    float inv_sum = 1.0f / sum;
    for (int i = 0; i < size; ++i) {
        output[i] *= inv_sum;
    }
}

// CUDA kernel for computing attention scores
__global__ void compute_attention_scores_kernel(
    const float* __restrict__ q,
    const float* __restrict__ k,
    float* __restrict__ scores,
    const float* __restrict__ mask,
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim,
    float scale
) {
    // Shared memory for tiles
    extern __shared__ float shared_mem[];
    float* q_tile = shared_mem;
    float* k_tile = &q_tile[TILE_Q * TILE_K];
    float* score_tile = &k_tile[TILE_Q * TILE_K];
    
    // Get thread indices
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_idx = tid / (num_heads * seq_len);
    int head_idx = (tid / seq_len) % num_heads;
    int seq_idx = tid % seq_len;
    
    if (batch_idx >= batch_size) return;
    
    // Initialize score
    float score = 0.0f;
    
    // Load Q tile
    for (int i = 0; i < head_dim; i += TILE_Q) {
        int q_idx = batch_idx * num_heads * seq_len * head_dim +
                   head_idx * seq_len * head_dim +
                   seq_idx * head_dim + i;
        q_tile[threadIdx.x] = q[q_idx];
        
        // Load K tile
        for (int j = 0; j < seq_len; j += TILE_K) {
            int k_idx = batch_idx * num_heads * seq_len * head_dim +
                       head_idx * seq_len * head_dim +
                       j * head_dim + i;
            k_tile[threadIdx.x] = k[k_idx];
            
            // Compute attention score
            for (int k = 0; k < TILE_K; ++k) {
                score += q_tile[threadIdx.x] * k_tile[k];
            }
        }
    }
    
    // Apply scaling
    score *= scale;
    
    // Apply mask if provided
    if (mask) {
        score *= mask[batch_idx * seq_len + seq_idx];
    }
    
    // Store score
    int score_idx = batch_idx * num_heads * seq_len * seq_len +
                   head_idx * seq_len * seq_len +
                   seq_idx * seq_len;
    scores[score_idx] = score;
}

// CUDA kernel for computing softmax
__global__ void compute_softmax_kernel(
    float* __restrict__ scores,
    int batch_size,
    int num_heads,
    int seq_len
) {
    // Get thread indices
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_idx = tid / (num_heads * seq_len);
    int head_idx = (tid / seq_len) % num_heads;
    int seq_idx = tid % seq_len;
    
    if (batch_idx >= batch_size) return;
    
    // Compute softmax for this sequence position
    int score_idx = batch_idx * num_heads * seq_len * seq_len +
                   head_idx * seq_len * seq_len +
                   seq_idx * seq_len;
    
    compute_softmax(
        &scores[score_idx],
        &scores[score_idx],
        seq_len
    );
}

// CUDA kernel for computing output
__global__ void compute_output_kernel(
    const float* __restrict__ scores,
    const float* __restrict__ v,
    float* __restrict__ output,
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim
) {
    // Shared memory for tiles
    extern __shared__ float shared_mem[];
    float* score_tile = shared_mem;
    float* v_tile = &score_tile[TILE_ATTN * TILE_V];
    
    // Get thread indices
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_idx = tid / (num_heads * seq_len);
    int head_idx = (tid / seq_len) % num_heads;
    int seq_idx = tid % seq_len;
    
    if (batch_idx >= batch_size) return;
    
    // Initialize output
    float out = 0.0f;
    
    // Load score tile
    for (int i = 0; i < seq_len; i += TILE_ATTN) {
        int score_idx = batch_idx * num_heads * seq_len * seq_len +
                       head_idx * seq_len * seq_len +
                       seq_idx * seq_len + i;
        score_tile[threadIdx.x] = scores[score_idx];
        
        // Load V tile
        for (int j = 0; j < head_dim; j += TILE_V) {
            int v_idx = batch_idx * num_heads * seq_len * head_dim +
                       head_idx * seq_len * head_dim +
                       i * head_dim + j;
            v_tile[threadIdx.x] = v[v_idx];
            
            // Compute output
            for (int k = 0; k < TILE_V; ++k) {
                out += score_tile[threadIdx.x] * v_tile[k];
            }
        }
    }
    
    // Store output
    int out_idx = batch_idx * num_heads * seq_len * head_dim +
                 head_idx * seq_len * head_dim +
                 seq_idx * head_dim;
    output[out_idx] = out;
}

// Update the compute_attention_kernel function to use CUDA kernels
void FlashAttention::compute_attention_kernel() {
    // Compute scaling factor
    float scale = compute_scale(head_dim_);
    
    // Launch attention scores kernel
    int num_blocks = get_num_blocks(batch_size_ * num_heads_ * seq_len_);
    int shared_mem_size = (TILE_Q * TILE_K + TILE_Q * TILE_K + TILE_ATTN * TILE_V) * sizeof(float);
    
    compute_attention_scores_kernel<<<num_blocks, BLOCK_SIZE, shared_mem_size>>>(
        d_q_, d_k_, d_softmax_, d_mask_,
        batch_size_, num_heads_, seq_len_, head_dim_,
        scale
    );
    
    // Launch softmax kernel
    compute_softmax_kernel<<<num_blocks, BLOCK_SIZE>>>(
        d_softmax_,
        batch_size_, num_heads_, seq_len_
    );
    
    // Launch output kernel
    compute_output_kernel<<<num_blocks, BLOCK_SIZE, shared_mem_size>>>(
        d_softmax_, d_v_, d_output_,
        batch_size_, num_heads_, seq_len_, head_dim_
    );
    
    // Synchronize device
    cudaDeviceSynchronize();
} 