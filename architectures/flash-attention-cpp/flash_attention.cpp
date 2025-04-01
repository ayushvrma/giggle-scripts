#include "flash_attention.hpp"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <curand_kernel.h>
#include <cuda/atomic>
#include <cuda/std/type_traits>
#include <cuda/std/cstddef>
#include <cuda/std/cstdint>
#include <cuda/std/limits>
#include <cuda/std/chrono>
#include <cuda/std/thread>
#include <cuda/std/atomic>
#include <cuda/std/mutex>
#include <cuda/std/condition_variable>
#include <cuda/std/functional>
#include <cuda/std/array>
#include <cuda/std/tuple>
#include <cuda/std/utility>
#include <cuda/std/initializer_list>
#include <cuda/std/optional>
#include <cuda/std/variant>
#include <cuda/std/string>
#include <cuda/std/string_view>
#include <cuda/std/vector>
#include <cuda/std/algorithm>
#include <cuda/std/execution>
#include <cuda/std/iterator>
#include <cuda/std/memory>
#include <cuda/std/numeric>
#include <cuda/std/random>
#include <cuda/std/ratio>
#include <cuda/std/scoped_allocator>
#include <cuda/std/set>
#include <cuda/std/unordered_set>
#include <cuda/std/utility>
#include <cuda/std/valarray>
#include <cuda/std/version>
#include <cuda/std/type_traits>
#include <cuda/std/limits>
#include <cuda/std/climits>
#include <cuda/std/cfloat>
#include <cuda/std/cstdint>
#include <cuda/std/cstddef>
#include <cuda/std/cstdarg>
#include <cuda/std/cstring>
#include <cuda/std/cctype>
#include <cuda/std/cwchar>
#include <cuda/std/cwctype>
#include <cuda/std/cuchar>
#include <cuda/std/cstdbool>
#include <cuda/std/cstdalign>
#include <cuda/std/cstdarg>
#include <cuda/std/cstddef>
#include <cuda/std/cstdint>
#include <cuda/std/cstdio>
#include <cuda/std/cstdlib>
#include <cuda/std/cstring>
#include <cuda/std/ctime>
#include <cuda/std/cuchar>
#include <cuda/std/cwchar>
#include <cuda/std/cwctype>
#include <cuda/std/cerrno>
#include <cuda/std/clocale>
#include <cuda/std/cmath>
#include <cuda/std/csetjmp>
#include <cuda/std/csignal>

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

// Helper function to compute softmax with numerical stability (FP16 version)
__device__ __forceinline__ void compute_softmax(
    __half* input,
    __half* output,
    int size,
    __half scale = __float2half(1.0f)
) {
    // Find maximum value for numerical stability
    __half max_val = input[0];
    for (int i = 1; i < size; ++i) {
        max_val = __hmax(max_val, input[i]);
    }
    
    // Compute exp and sum
    __half sum = __float2half(0.0f);
    for (int i = 0; i < size; ++i) {
        output[i] = __h2exp(__hmul(__hsub(input[i], max_val), scale));
        sum = __hadd(sum, output[i]);
    }
    
    // Normalize
    __half inv_sum = __hdiv(__float2half(1.0f), sum);
    for (int i = 0; i < size; ++i) {
        output[i] = __hmul(output[i], inv_sum);
    }
}

// Helper function to compute matrix multiplication with tiling
__device__ __forceinline__ void compute_matmul(
    const float* a,
    const float* b,
    float* c,
    int m,
    int n,
    int k,
    int tile_size = TILE_SIZE
) {
    for (int i = 0; i < m; i += tile_size) {
        for (int j = 0; j < n; j += tile_size) {
            for (int k_idx = 0; k_idx < k; k_idx += tile_size) {
                // Compute tile boundaries
                int i_end = std::min(i + tile_size, m);
                int j_end = std::min(j + tile_size, n);
                int k_end = std::min(k_idx + tile_size, k);
                
                // Compute tile multiplication
                for (int ii = i; ii < i_end; ++ii) {
                    for (int jj = j; jj < j_end; ++jj) {
                        float sum = 0.0f;
                        for (int kk = k_idx; kk < k_end; ++kk) {
                            sum += a[ii * k + kk] * b[kk * n + jj];
                        }
                        c[ii * n + jj] += sum;
                    }
                }
            }
        }
    }
}

// Helper function to compute matrix multiplication with tiling (FP16 version)
__device__ __forceinline__ void compute_matmul(
    const __half* a,
    const __half* b,
    __half* c,
    int m,
    int n,
    int k,
    int tile_size = TILE_SIZE
) {
    for (int i = 0; i < m; i += tile_size) {
        for (int j = 0; j < n; j += tile_size) {
            for (int k_idx = 0; k_idx < k; k_idx += tile_size) {
                // Compute tile boundaries
                int i_end = std::min(i + tile_size, m);
                int j_end = std::min(j + tile_size, n);
                int k_end = std::min(k_idx + tile_size, k);
                
                // Compute tile multiplication
                for (int ii = i; ii < i_end; ++ii) {
                    for (int jj = j; jj < j_end; ++jj) {
                        __half sum = __float2half(0.0f);
                        for (int kk = k_idx; kk < k_end; ++kk) {
                            sum = __hadd(sum, __hmul(a[ii * k + kk], b[kk * n + jj]));
                        }
                        c[ii * n + jj] = __hadd(c[ii * n + jj], sum);
                    }
                }
            }
        }
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

// CUDA kernel for computing attention scores (FP16 version)
__global__ void compute_attention_scores_kernel(
    const __half* __restrict__ q,
    const __half* __restrict__ k,
    __half* __restrict__ scores,
    const __half* __restrict__ mask,
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim,
    __half scale
) {
    // Shared memory for tiles
    extern __shared__ __half shared_mem[];
    __half* q_tile = shared_mem;
    __half* k_tile = &q_tile[TILE_Q * TILE_K];
    __half* score_tile = &k_tile[TILE_Q * TILE_K];
    
    // Get thread indices
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_idx = tid / (num_heads * seq_len);
    int head_idx = (tid / seq_len) % num_heads;
    int seq_idx = tid % seq_len;
    
    if (batch_idx >= batch_size) return;
    
    // Initialize score
    __half score = __float2half(0.0f);
    
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
                score = __hadd(score, __hmul(q_tile[threadIdx.x], k_tile[k]));
            }
        }
    }
    
    // Apply scaling
    score = __hmul(score, scale);
    
    // Apply mask if provided
    if (mask) {
        score = __hmul(score, mask[batch_idx * seq_len + seq_idx]);
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

// CUDA kernel for computing softmax (FP16 version)
__global__ void compute_softmax_kernel(
    __half* __restrict__ scores,
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

// CUDA kernel for computing output (FP16 version)
__global__ void compute_output_kernel(
    const __half* __restrict__ scores,
    const __half* __restrict__ v,
    __half* __restrict__ output,
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim
) {
    // Shared memory for tiles
    extern __shared__ __half shared_mem[];
    __half* score_tile = shared_mem;
    __half* v_tile = &score_tile[TILE_ATTN * TILE_V];
    
    // Get thread indices
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_idx = tid / (num_heads * seq_len);
    int head_idx = (tid / seq_len) % num_heads;
    int seq_idx = tid % seq_len;
    
    if (batch_idx >= batch_size) return;
    
    // Initialize output
    __half out = __float2half(0.0f);
    
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
                out = __hadd(out, __hmul(score_tile[threadIdx.x], v_tile[k]));
            }
        }
    }
    
    // Store output
    int out_idx = batch_idx * num_heads * seq_len * head_dim +
                 head_idx * seq_len * head_dim +
                 seq_idx * head_dim;
    output[out_idx] = out;
}

// CUDA kernel for generating dropout mask
__global__ void generate_dropout_mask_kernel(
    float* __restrict__ mask,
    curandState* __restrict__ states,
    int size,
    float dropout_prob
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= size) return;
    
    curandState local_state = states[tid];
    float rand_val = curand_uniform(&local_state);
    mask[tid] = (rand_val > dropout_prob) ? 1.0f : 0.0f;
    states[tid] = local_state;
}

// CUDA kernel for generating dropout mask (FP16 version)
__global__ void generate_dropout_mask_kernel(
    __half* __restrict__ mask,
    curandState* __restrict__ states,
    int size,
    float dropout_prob
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= size) return;
    
    curandState local_state = states[tid];
    float rand_val = curand_uniform(&local_state);
    mask[tid] = (rand_val > dropout_prob) ? __float2half(1.0f) : __float2half(0.0f);
    states[tid] = local_state;
}

// Constructor with FP32 support
FlashAttention::FlashAttention(
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim,
    AttentionType type,
    int window_size
)
    : batch_size_(batch_size)
    , num_heads_(num_heads)
    , seq_len_(seq_len)
    , head_dim_(head_dim)
    , use_fp16_(false)
    , attention_type_(type)
    , window_size_(window_size)
    , d_q_(nullptr)
    , d_k_(nullptr)
    , d_v_(nullptr)
    , d_output_(nullptr)
    , d_mask_(nullptr)
    , d_softmax_(nullptr)
    , d_dropout_mask_(nullptr)
    , d_causal_mask_(nullptr)
    , d_window_mask_(nullptr)
    , d_rand_states_(nullptr)
{
    // Create CUDA events for timing
    cudaEventCreate(&start_event_);
    cudaEventCreate(&stop_event_);
    
    allocate_device_memory();
    initialize_random_states();
}

// Constructor with FP16 support
FlashAttention::FlashAttention(
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim,
    bool use_fp16,
    AttentionType type,
    int window_size
)
    : batch_size_(batch_size)
    , num_heads_(num_heads)
    , seq_len_(seq_len)
    , head_dim_(head_dim)
    , use_fp16_(use_fp16)
    , attention_type_(type)
    , window_size_(window_size)
    , d_q_(nullptr)
    , d_k_(nullptr)
    , d_v_(nullptr)
    , d_output_(nullptr)
    , d_mask_(nullptr)
    , d_softmax_(nullptr)
    , d_dropout_mask_(nullptr)
    , d_causal_mask_(nullptr)
    , d_window_mask_(nullptr)
    , d_rand_states_(nullptr)
    , d_q_fp16_(nullptr)
    , d_k_fp16_(nullptr)
    , d_v_fp16_(nullptr)
    , d_output_fp16_(nullptr)
    , d_mask_fp16_(nullptr)
    , d_softmax_fp16_(nullptr)
    , d_dropout_mask_fp16_(nullptr)
    , d_causal_mask_fp16_(nullptr)
    , d_window_mask_fp16_(nullptr)
{
    // Create CUDA events for timing
    cudaEventCreate(&start_event_);
    cudaEventCreate(&stop_event_);
    
    allocate_device_memory();
    initialize_random_states();
}

FlashAttention::~FlashAttention() {
    free_device_memory();
    cudaEventDestroy(start_event_);
    cudaEventDestroy(stop_event_);
}

void FlashAttention::allocate_device_memory() {
    if (use_fp16_) {
        // Allocate FP16 memory
        cudaMalloc(&d_q_fp16_, batch_size_ * num_heads_ * seq_len_ * head_dim_ * sizeof(__half));
        cudaMalloc(&d_k_fp16_, batch_size_ * num_heads_ * seq_len_ * head_dim_ * sizeof(__half));
        cudaMalloc(&d_v_fp16_, batch_size_ * num_heads_ * seq_len_ * head_dim_ * sizeof(__half));
        cudaMalloc(&d_output_fp16_, batch_size_ * num_heads_ * seq_len_ * head_dim_ * sizeof(__half));
        cudaMalloc(&d_mask_fp16_, batch_size_ * seq_len_ * sizeof(__half));
        cudaMalloc(&d_softmax_fp16_, batch_size_ * num_heads_ * seq_len_ * seq_len_ * sizeof(__half));
        cudaMalloc(&d_dropout_mask_fp16_, batch_size_ * num_heads_ * seq_len_ * seq_len_ * sizeof(__half));
        
        // Allocate attention masks if needed
        if (attention_type_ == AttentionType::CAUSAL) {
            cudaMalloc(&d_causal_mask_fp16_, batch_size_ * num_heads_ * seq_len_ * seq_len_ * sizeof(__half));
        } else if (attention_type_ == AttentionType::SLIDING_WINDOW) {
            cudaMalloc(&d_window_mask_fp16_, batch_size_ * num_heads_ * seq_len_ * seq_len_ * sizeof(__half));
        }
    } else {
        // Allocate FP32 memory
        cudaMalloc(&d_q_, batch_size_ * num_heads_ * seq_len_ * head_dim_ * sizeof(float));
        cudaMalloc(&d_k_, batch_size_ * num_heads_ * seq_len_ * head_dim_ * sizeof(float));
        cudaMalloc(&d_v_, batch_size_ * num_heads_ * seq_len_ * head_dim_ * sizeof(float));
        cudaMalloc(&d_output_, batch_size_ * num_heads_ * seq_len_ * head_dim_ * sizeof(float));
        cudaMalloc(&d_mask_, batch_size_ * seq_len_ * sizeof(float));
        cudaMalloc(&d_softmax_, batch_size_ * num_heads_ * seq_len_ * seq_len_ * sizeof(float));
        cudaMalloc(&d_dropout_mask_, batch_size_ * num_heads_ * seq_len_ * seq_len_ * sizeof(float));
        
        // Allocate attention masks if needed
        if (attention_type_ == AttentionType::CAUSAL) {
            cudaMalloc(&d_causal_mask_, batch_size_ * num_heads_ * seq_len_ * seq_len_ * sizeof(float));
        } else if (attention_type_ == AttentionType::SLIDING_WINDOW) {
            cudaMalloc(&d_window_mask_, batch_size_ * num_heads_ * seq_len_ * seq_len_ * sizeof(float));
        }
    }
}

void FlashAttention::free_device_memory() {
    if (use_fp16_) {
        if (d_q_fp16_) cudaFree(d_q_fp16_);
        if (d_k_fp16_) cudaFree(d_k_fp16_);
        if (d_v_fp16_) cudaFree(d_v_fp16_);
        if (d_output_fp16_) cudaFree(d_output_fp16_);
        if (d_mask_fp16_) cudaFree(d_mask_fp16_);
        if (d_softmax_fp16_) cudaFree(d_softmax_fp16_);
        if (d_dropout_mask_fp16_) cudaFree(d_dropout_mask_fp16_);
        if (d_causal_mask_fp16_) cudaFree(d_causal_mask_fp16_);
        if (d_window_mask_fp16_) cudaFree(d_window_mask_fp16_);
    } else {
        if (d_q_) cudaFree(d_q_);
        if (d_k_) cudaFree(d_k_);
        if (d_v_) cudaFree(d_v_);
        if (d_output_) cudaFree(d_output_);
        if (d_mask_) cudaFree(d_mask_);
        if (d_softmax_) cudaFree(d_softmax_);
        if (d_dropout_mask_) cudaFree(d_dropout_mask_);
        if (d_causal_mask_) cudaFree(d_causal_mask_);
        if (d_window_mask_) cudaFree(d_window_mask_);
    }
    if (d_rand_states_) cudaFree(d_rand_states_);
}

void FlashAttention::initialize_random_states() {
    int num_threads = batch_size_ * num_heads_ * seq_len_;
    cudaMalloc(&d_rand_states_, num_threads * sizeof(curandState));
    
    // Initialize random states
    curand_init<<<(num_threads + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
        d_rand_states_,
        num_threads,
        0,  // seed
        0   // sequence
    );
}

void FlashAttention::check_cuda_error(const char* message) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s: %s\n", message, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

// CUDA kernel for generating causal mask
__global__ void generate_causal_mask_kernel(
    float* __restrict__ mask,
    int batch_size,
    int num_heads,
    int seq_len
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_idx = tid / (num_heads * seq_len * seq_len);
    int head_idx = (tid / (seq_len * seq_len)) % num_heads;
    int i = (tid / seq_len) % seq_len;
    int j = tid % seq_len;
    
    if (batch_idx >= batch_size) return;
    
    int mask_idx = batch_idx * num_heads * seq_len * seq_len +
                  head_idx * seq_len * seq_len +
                  i * seq_len + j;
    
    mask[mask_idx] = (j <= i) ? 1.0f : 0.0f;
}

// CUDA kernel for generating sliding window mask
__global__ void generate_window_mask_kernel(
    float* __restrict__ mask,
    int batch_size,
    int num_heads,
    int seq_len,
    int window_size
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_idx = tid / (num_heads * seq_len * seq_len);
    int head_idx = (tid / (seq_len * seq_len)) % num_heads;
    int i = (tid / seq_len) % seq_len;
    int j = tid % seq_len;
    
    if (batch_idx >= batch_size) return;
    
    int mask_idx = batch_idx * num_heads * seq_len * seq_len +
                  head_idx * seq_len * seq_len +
                  i * seq_len + j;
    
    mask[mask_idx] = (std::abs(i - j) <= window_size) ? 1.0f : 0.0f;
}

// FP16 versions of the mask generation kernels
__global__ void generate_causal_mask_kernel(
    __half* __restrict__ mask,
    int batch_size,
    int num_heads,
    int seq_len
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_idx = tid / (num_heads * seq_len * seq_len);
    int head_idx = (tid / (seq_len * seq_len)) % num_heads;
    int i = (tid / seq_len) % seq_len;
    int j = tid % seq_len;
    
    if (batch_idx >= batch_size) return;
    
    int mask_idx = batch_idx * num_heads * seq_len * seq_len +
                  head_idx * seq_len * seq_len +
                  i * seq_len + j;
    
    mask[mask_idx] = (j <= i) ? __float2half(1.0f) : __float2half(0.0f);
}

__global__ void generate_window_mask_kernel(
    __half* __restrict__ mask,
    int batch_size,
    int num_heads,
    int seq_len,
    int window_size
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_idx = tid / (num_heads * seq_len * seq_len);
    int head_idx = (tid / (seq_len * seq_len)) % num_heads;
    int i = (tid / seq_len) % seq_len;
    int j = tid % seq_len;
    
    if (batch_idx >= batch_size) return;
    
    int mask_idx = batch_idx * num_heads * seq_len * seq_len +
                  head_idx * seq_len * seq_len +
                  i * seq_len + j;
    
    mask[mask_idx] = (std::abs(i - j) <= window_size) ? __float2half(1.0f) : __float2half(0.0f);
}

void FlashAttention::update_performance_metrics(float kernel_time_ms, float transfer_time_ms) {
    last_metrics_.kernel_time_ms = kernel_time_ms;
    last_metrics_.memory_transfer_time_ms = transfer_time_ms;
    last_metrics_.total_time_ms = kernel_time_ms + transfer_time_ms;
    
    // Calculate GFLOPS
    // For each position, we compute:
    // 1. Q * K^T: seq_len * head_dim * seq_len operations
    // 2. Softmax: seq_len * seq_len operations
    // 3. (Softmax) * V: seq_len * seq_len * head_dim operations
    float total_ops = batch_size_ * num_heads_ * (
        seq_len_ * head_dim_ * seq_len_ +  // Q * K^T
        seq_len_ * seq_len_ +              // Softmax
        seq_len_ * seq_len_ * head_dim_    // (Softmax) * V
    );
    
    last_metrics_.gflops = (total_ops / (last_metrics_.total_time_ms * 1e6)) / 1e9;
}

// FP32 compute method
void FlashAttention::compute(
    const float* q,
    const float* k,
    const float* v,
    float* output,
    const float* mask,
    float dropout_prob
) {
    if (use_fp16_) {
        fprintf(stderr, "Error: FP16 instance called with FP32 data\n");
        exit(EXIT_FAILURE);
    }
    
    // Record start time
    cudaEventRecord(start_event_);
    
    // Copy input data to device
    cudaMemcpy(d_q_, q, batch_size_ * num_heads_ * seq_len_ * head_dim_ * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_k_, k, batch_size_ * num_heads_ * seq_len_ * head_dim_ * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v_, v, batch_size_ * num_heads_ * seq_len_ * head_dim_ * sizeof(float), cudaMemcpyHostToDevice);
    
    if (mask) {
        cudaMemcpy(d_mask_, mask, batch_size_ * seq_len_ * sizeof(float), cudaMemcpyHostToDevice);
    }
    
    // Generate attention masks if needed
    if (attention_type_ == AttentionType::CAUSAL) {
        int num_threads = batch_size_ * num_heads_ * seq_len_ * seq_len_;
        generate_causal_mask_kernel<<<(num_threads + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
            d_causal_mask_,
            batch_size_,
            num_heads_,
            seq_len_
        );
    } else if (attention_type_ == AttentionType::SLIDING_WINDOW) {
        int num_threads = batch_size_ * num_heads_ * seq_len_ * seq_len_;
        generate_window_mask_kernel<<<(num_threads + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
            d_window_mask_,
            batch_size_,
            num_heads_,
            seq_len_,
            window_size_
        );
    }
    
    // Generate dropout mask if needed
    if (dropout_prob > 0.0f) {
        int num_threads = batch_size_ * num_heads_ * seq_len_ * seq_len_;
        generate_dropout_mask_kernel<<<(num_threads + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
            d_dropout_mask_,
            d_rand_states_,
            num_threads,
            dropout_prob
        );
    }
    
    // Record time after memory transfers
    cudaEventRecord(stop_event_);
    cudaEventSynchronize(stop_event_);
    float transfer_time_ms;
    cudaEventElapsedTime(&transfer_time_ms, start_event_, stop_event_);
    
    // Record start time for kernel execution
    cudaEventRecord(start_event_);
    
    // Compute attention
    compute_attention_kernel();
    
    // Record time after kernel execution
    cudaEventRecord(stop_event_);
    cudaEventSynchronize(stop_event_);
    float kernel_time_ms;
    cudaEventElapsedTime(&kernel_time_ms, start_event_, stop_event_);
    
    // Update performance metrics
    update_performance_metrics(kernel_time_ms, transfer_time_ms);
    
    // Copy result back to host
    cudaMemcpy(output, d_output_, batch_size_ * num_heads_ * seq_len_ * head_dim_ * sizeof(float), cudaMemcpyDeviceToHost);
}

// FP16 compute method
void FlashAttention::compute(
    const __half* q,
    const __half* k,
    const __half* v,
    __half* output,
    const __half* mask,
    float dropout_prob
) {
    if (!use_fp16_) {
        fprintf(stderr, "Error: FP32 instance called with FP16 data\n");
        exit(EXIT_FAILURE);
    }
    
    // Record start time
    cudaEventRecord(start_event_);
    
    // Copy input data to device
    cudaMemcpy(d_q_fp16_, q, batch_size_ * num_heads_ * seq_len_ * head_dim_ * sizeof(__half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_k_fp16_, k, batch_size_ * num_heads_ * seq_len_ * head_dim_ * sizeof(__half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v_fp16_, v, batch_size_ * num_heads_ * seq_len_ * head_dim_ * sizeof(__half), cudaMemcpyHostToDevice);
    
    if (mask) {
        cudaMemcpy(d_mask_fp16_, mask, batch_size_ * seq_len_ * sizeof(__half), cudaMemcpyHostToDevice);
    }
    
    // Generate attention masks if needed
    if (attention_type_ == AttentionType::CAUSAL) {
        int num_threads = batch_size_ * num_heads_ * seq_len_ * seq_len_;
        generate_causal_mask_kernel<<<(num_threads + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
            d_causal_mask_fp16_,
            batch_size_,
            num_heads_,
            seq_len_
        );
    } else if (attention_type_ == AttentionType::SLIDING_WINDOW) {
        int num_threads = batch_size_ * num_heads_ * seq_len_ * seq_len_;
        generate_window_mask_kernel<<<(num_threads + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
            d_window_mask_fp16_,
            batch_size_,
            num_heads_,
            seq_len_,
            window_size_
        );
    }
    
    // Generate dropout mask if needed
    if (dropout_prob > 0.0f) {
        int num_threads = batch_size_ * num_heads_ * seq_len_ * seq_len_;
        generate_dropout_mask_kernel<<<(num_threads + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(
            d_dropout_mask_fp16_,
            d_rand_states_,
            num_threads,
            dropout_prob
        );
    }
    
    // Record time after memory transfers
    cudaEventRecord(stop_event_);
    cudaEventSynchronize(stop_event_);
    float transfer_time_ms;
    cudaEventElapsedTime(&transfer_time_ms, start_event_, stop_event_);
    
    // Record start time for kernel execution
    cudaEventRecord(start_event_);
    
    // Compute attention
    compute_attention_kernel();
    
    // Record time after kernel execution
    cudaEventRecord(stop_event_);
    cudaEventSynchronize(stop_event_);
    float kernel_time_ms;
    cudaEventElapsedTime(&kernel_time_ms, start_event_, stop_event_);
    
    // Update performance metrics
    update_performance_metrics(kernel_time_ms, transfer_time_ms);
    
    // Copy result back to host
    cudaMemcpy(output, d_output_fp16_, batch_size_ * num_heads_ * seq_len_ * head_dim_ * sizeof(__half), cudaMemcpyDeviceToHost);
}

void FlashAttention::compute_attention_kernel() {
    // Compute scaling factor
    float scale = compute_scale(head_dim_);
    
    // Launch attention scores kernel
    int num_blocks = get_num_blocks(batch_size_ * num_heads_ * seq_len_);
    int shared_mem_size = (TILE_Q * TILE_K + TILE_Q * TILE_K + TILE_ATTN * TILE_V) * sizeof(float);
    
    if (use_fp16_) {
        compute_attention_scores_kernel<<<num_blocks, BLOCK_SIZE, shared_mem_size>>>(
            d_q_fp16_, d_k_fp16_, d_softmax_fp16_, d_mask_fp16_,
            batch_size_, num_heads_, seq_len_, head_dim_,
            __float2half(scale)
        );
        
        // Launch softmax kernel
        compute_softmax_kernel<<<num_blocks, BLOCK_SIZE>>>(
            d_softmax_fp16_,
            batch_size_, num_heads_, seq_len_
        );
        
        // Launch output kernel
        compute_output_kernel<<<num_blocks, BLOCK_SIZE, shared_mem_size>>>(
            d_softmax_fp16_, d_v_fp16_, d_output_fp16_,
            batch_size_, num_heads_, seq_len_, head_dim_
        );
    } else {
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
    }
    
    // Synchronize device
    cudaDeviceSynchronize();
}