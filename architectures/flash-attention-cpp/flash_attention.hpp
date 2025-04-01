#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <memory>
#include <vector>

// Attention type enum
enum class AttentionType {
    STANDARD,    // Standard attention
    CAUSAL,      // Causal attention (each position can only attend to previous positions)
    SLIDING_WINDOW  // Sliding window attention (each position can only attend to positions within a window)
};

class FlashAttention {
public:
    // Constructor with FP32 support
    FlashAttention(
        int batch_size,
        int num_heads,
        int seq_len,
        int head_dim,
        AttentionType type = AttentionType::STANDARD,
        int window_size = 0  // Only used for SLIDING_WINDOW
    );
    
    // Constructor with FP16 support
    FlashAttention(
        int batch_size,
        int num_heads,
        int seq_len,
        int head_dim,
        bool use_fp16,
        AttentionType type = AttentionType::STANDARD,
        int window_size = 0  // Only used for SLIDING_WINDOW
    );
    
    ~FlashAttention();

    // FP32 compute method
    void compute(
        const float* q,
        const float* k,
        const float* v,
        float* output,
        const float* mask = nullptr,
        float dropout_prob = 0.0f
    );

    // FP16 compute method
    void compute(
        const __half* q,
        const __half* k,
        const __half* v,
        __half* output,
        const __half* mask = nullptr,
        float dropout_prob = 0.0f
    );

    // Getters for dimensions and configuration
    int get_batch_size() const { return batch_size_; }
    int get_num_heads() const { return num_heads_; }
    int get_seq_len() const { return seq_len_; }
    int get_head_dim() const { return head_dim_; }
    bool is_fp16() const { return use_fp16_; }
    AttentionType get_attention_type() const { return attention_type_; }
    int get_window_size() const { return window_size_; }

    // Performance benchmarking
    struct PerformanceMetrics {
        float kernel_time_ms;
        float memory_transfer_time_ms;
        float total_time_ms;
        float gflops;
    };

    PerformanceMetrics get_last_performance_metrics() const { return last_metrics_; }

private:
    // Memory management
    void allocate_device_memory();
    void free_device_memory();

    // Core computation
    void compute_attention_kernel();

    // Helper functions
    void check_cuda_error(const char* message);
    void initialize_random_states();
    void update_performance_metrics(float kernel_time_ms, float transfer_time_ms);

    // Member variables
    int batch_size_;
    int num_heads_;
    int seq_len_;
    int head_dim_;
    bool use_fp16_;
    AttentionType attention_type_;
    int window_size_;
    PerformanceMetrics last_metrics_;

    // FP32 device pointers
    float* d_q_;
    float* d_k_;
    float* d_v_;
    float* d_output_;
    float* d_mask_;
    float* d_softmax_;
    float* d_dropout_mask_;
    float* d_causal_mask_;  // For causal attention
    float* d_window_mask_;  // For sliding window attention

    // FP16 device pointers
    __half* d_q_fp16_;
    __half* d_k_fp16_;
    __half* d_v_fp16_;
    __half* d_output_fp16_;
    __half* d_mask_fp16_;
    __half* d_softmax_fp16_;
    __half* d_dropout_mask_fp16_;
    __half* d_causal_mask_fp16_;  // For causal attention
    __half* d_window_mask_fp16_;  // For sliding window attention

    // CUDA random states for dropout
    curandState* d_rand_states_;

    // CUDA events for timing
    cudaEvent_t start_event_;
    cudaEvent_t stop_event_;
}; 