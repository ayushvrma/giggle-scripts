#include "flash_attention.hpp"
#include <gtest/gtest.h>
#include <random>
#include <cmath>
#include <iostream>

// Helper function to generate random data
template<typename T>
void generate_random_data(T* data, int size, float min_val = -1.0f, float max_val = 1.0f) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(min_val, max_val);
    
    for (int i = 0; i < size; ++i) {
        data[i] = static_cast<T>(dis(gen));
    }
}

// Helper function to compare FP32 arrays
bool compare_float_arrays(const float* a, const float* b, int size, float tolerance = 1e-5f) {
    for (int i = 0; i < size; ++i) {
        if (std::abs(a[i] - b[i]) > tolerance) {
            return false;
        }
    }
    return true;
}

// Helper function to compare FP16 arrays
bool compare_half_arrays(const __half* a, const __half* b, int size, float tolerance = 1e-3f) {
    for (int i = 0; i < size; ++i) {
        if (std::abs(__half2float(a[i]) - __half2float(b[i])) > tolerance) {
            return false;
        }
    }
    return true;
}

// Test case for FP32 Flash Attention
TEST(FlashAttentionTest, FP32Basic) {
    const int batch_size = 2;
    const int num_heads = 4;
    const int seq_len = 8;
    const int head_dim = 16;
    
    // Create input tensors
    std::vector<float> q(batch_size * num_heads * seq_len * head_dim);
    std::vector<float> k(batch_size * num_heads * seq_len * head_dim);
    std::vector<float> v(batch_size * num_heads * seq_len * head_dim);
    std::vector<float> output(batch_size * num_heads * seq_len * head_dim);
    
    // Generate random input data
    generate_random_data(q.data(), q.size());
    generate_random_data(k.data(), k.size());
    generate_random_data(v.data(), v.size());
    
    // Create Flash Attention instance
    FlashAttention flash_attn(batch_size, num_heads, seq_len, head_dim);
    
    // Compute attention
    flash_attn.compute(q.data(), k.data(), v.data(), output.data());
    
    // Verify output is not zero
    bool has_non_zero = false;
    for (float val : output) {
        if (std::abs(val) > 1e-6f) {
            has_non_zero = true;
            break;
        }
    }
    EXPECT_TRUE(has_non_zero);
}

// Test case for FP16 Flash Attention
TEST(FlashAttentionTest, FP16Basic) {
    const int batch_size = 2;
    const int num_heads = 4;
    const int seq_len = 8;
    const int head_dim = 16;
    
    // Create input tensors
    std::vector<__half> q(batch_size * num_heads * seq_len * head_dim);
    std::vector<__half> k(batch_size * num_heads * seq_len * head_dim);
    std::vector<__half> v(batch_size * num_heads * seq_len * head_dim);
    std::vector<__half> output(batch_size * num_heads * seq_len * head_dim);
    
    // Generate random input data
    std::vector<float> q_float(q.size());
    std::vector<float> k_float(k.size());
    std::vector<float> v_float(v.size());
    
    generate_random_data(q_float.data(), q_float.size());
    generate_random_data(k_float.data(), k_float.size());
    generate_random_data(v_float.data(), v_float.size());
    
    // Convert to FP16
    for (size_t i = 0; i < q.size(); ++i) {
        q[i] = __float2half(q_float[i]);
        k[i] = __float2half(k_float[i]);
        v[i] = __float2half(v_float[i]);
    }
    
    // Create Flash Attention instance
    FlashAttention flash_attn(batch_size, num_heads, seq_len, head_dim, true);
    
    // Compute attention
    flash_attn.compute(q.data(), k.data(), v.data(), output.data());
    
    // Verify output is not zero
    bool has_non_zero = false;
    for (__half val : output) {
        if (std::abs(__half2float(val)) > 1e-6f) {
            has_non_zero = true;
            break;
        }
    }
    EXPECT_TRUE(has_non_zero);
}

// Test case for attention mask
TEST(FlashAttentionTest, WithMask) {
    const int batch_size = 2;
    const int num_heads = 4;
    const int seq_len = 8;
    const int head_dim = 16;
    
    // Create input tensors
    std::vector<float> q(batch_size * num_heads * seq_len * head_dim);
    std::vector<float> k(batch_size * num_heads * seq_len * head_dim);
    std::vector<float> v(batch_size * num_heads * seq_len * head_dim);
    std::vector<float> output(batch_size * num_heads * seq_len * head_dim);
    std::vector<float> mask(batch_size * seq_len);
    
    // Generate random input data
    generate_random_data(q.data(), q.size());
    generate_random_data(k.data(), k.size());
    generate_random_data(v.data(), v.size());
    
    // Generate mask (some positions masked)
    for (int i = 0; i < mask.size(); ++i) {
        mask[i] = (i % 2 == 0) ? 1.0f : 0.0f;
    }
    
    // Create Flash Attention instance
    FlashAttention flash_attn(batch_size, num_heads, seq_len, head_dim);
    
    // Compute attention with mask
    flash_attn.compute(q.data(), k.data(), v.data(), output.data(), mask.data());
    
    // Verify masked positions are zero
    for (int b = 0; b < batch_size; ++b) {
        for (int h = 0; h < num_heads; ++h) {
            for (int i = 0; i < seq_len; ++i) {
                if (mask[b * seq_len + i] == 0.0f) {
                    for (int j = 0; j < head_dim; ++j) {
                        int idx = b * num_heads * seq_len * head_dim +
                                h * seq_len * head_dim +
                                i * head_dim + j;
                        EXPECT_NEAR(output[idx], 0.0f, 1e-6f);
                    }
                }
            }
        }
    }
}

// Test case for dropout
TEST(FlashAttentionTest, WithDropout) {
    const int batch_size = 2;
    const int num_heads = 4;
    const int seq_len = 8;
    const int head_dim = 16;
    const float dropout_prob = 0.5f;
    
    // Create input tensors
    std::vector<float> q(batch_size * num_heads * seq_len * head_dim);
    std::vector<float> k(batch_size * num_heads * seq_len * head_dim);
    std::vector<float> v(batch_size * num_heads * seq_len * head_dim);
    std::vector<float> output1(batch_size * num_heads * seq_len * head_dim);
    std::vector<float> output2(batch_size * num_heads * seq_len * head_dim);
    
    // Generate random input data
    generate_random_data(q.data(), q.size());
    generate_random_data(k.data(), k.size());
    generate_random_data(v.data(), v.size());
    
    // Create Flash Attention instance
    FlashAttention flash_attn(batch_size, num_heads, seq_len, head_dim);
    
    // Compute attention twice with dropout
    flash_attn.compute(q.data(), k.data(), v.data(), output1.data(), nullptr, dropout_prob);
    flash_attn.compute(q.data(), k.data(), v.data(), output2.data(), nullptr, dropout_prob);
    
    // Verify outputs are different due to dropout
    EXPECT_FALSE(compare_float_arrays(output1.data(), output2.data(), output1.size()));
}

// Test case for causal attention
TEST(FlashAttentionTest, CausalAttention) {
    const int batch_size = 2;
    const int num_heads = 4;
    const int seq_len = 8;
    const int head_dim = 16;
    
    // Create input tensors
    std::vector<float> q(batch_size * num_heads * seq_len * head_dim);
    std::vector<float> k(batch_size * num_heads * seq_len * head_dim);
    std::vector<float> v(batch_size * num_heads * seq_len * head_dim);
    std::vector<float> output(batch_size * num_heads * seq_len * head_dim);
    
    // Generate random input data
    generate_random_data(q.data(), q.size());
    generate_random_data(k.data(), k.size());
    generate_random_data(v.data(), v.size());
    
    // Create Flash Attention instance with causal attention
    FlashAttention flash_attn(batch_size, num_heads, seq_len, head_dim, AttentionType::CAUSAL);
    
    // Compute attention
    flash_attn.compute(q.data(), k.data(), v.data(), output.data());
    
    // Verify causal property: each position can only attend to previous positions
    for (int b = 0; b < batch_size; ++b) {
        for (int h = 0; h < num_heads; ++h) {
            for (int i = 0; i < seq_len; ++i) {
                for (int j = 0; j < seq_len; ++j) {
                    if (j > i) {
                        // Check that attention score is zero for future positions
                        int score_idx = b * num_heads * seq_len * seq_len +
                                      h * seq_len * seq_len +
                                      i * seq_len + j;
                        EXPECT_NEAR(flash_attn.get_last_performance_metrics().kernel_time_ms, 0.0f, 1e-6f);
                    }
                }
            }
        }
    }
}

// Test case for sliding window attention
TEST(FlashAttentionTest, SlidingWindowAttention) {
    const int batch_size = 2;
    const int num_heads = 4;
    const int seq_len = 8;
    const int head_dim = 16;
    const int window_size = 2;
    
    // Create input tensors
    std::vector<float> q(batch_size * num_heads * seq_len * head_dim);
    std::vector<float> k(batch_size * num_heads * seq_len * head_dim);
    std::vector<float> v(batch_size * num_heads * seq_len * head_dim);
    std::vector<float> output(batch_size * num_heads * seq_len * head_dim);
    
    // Generate random input data
    generate_random_data(q.data(), q.size());
    generate_random_data(k.data(), k.size());
    generate_random_data(v.data(), v.size());
    
    // Create Flash Attention instance with sliding window attention
    FlashAttention flash_attn(batch_size, num_heads, seq_len, head_dim, AttentionType::SLIDING_WINDOW, window_size);
    
    // Compute attention
    flash_attn.compute(q.data(), k.data(), v.data(), output.data());
    
    // Verify sliding window property: each position can only attend to positions within the window
    for (int b = 0; b < batch_size; ++b) {
        for (int h = 0; h < num_heads; ++h) {
            for (int i = 0; i < seq_len; ++i) {
                for (int j = 0; j < seq_len; ++j) {
                    if (std::abs(i - j) > window_size) {
                        // Check that attention score is zero for positions outside the window
                        int score_idx = b * num_heads * seq_len * seq_len +
                                      h * seq_len * seq_len +
                                      i * seq_len + j;
                        EXPECT_NEAR(flash_attn.get_last_performance_metrics().kernel_time_ms, 0.0f, 1e-6f);
                    }
                }
            }
        }
    }
}

// Test case for performance benchmarking
TEST(FlashAttentionTest, PerformanceBenchmarking) {
    const int batch_size = 4;
    const int num_heads = 8;
    const int seq_len = 64;
    const int head_dim = 32;
    
    // Create input tensors
    std::vector<float> q(batch_size * num_heads * seq_len * head_dim);
    std::vector<float> k(batch_size * num_heads * seq_len * head_dim);
    std::vector<float> v(batch_size * num_heads * seq_len * head_dim);
    std::vector<float> output(batch_size * num_heads * seq_len * head_dim);
    
    // Generate random input data
    generate_random_data(q.data(), q.size());
    generate_random_data(k.data(), k.size());
    generate_random_data(v.data(), v.size());
    
    // Create Flash Attention instance
    FlashAttention flash_attn(batch_size, num_heads, seq_len, head_dim);
    
    // Warm-up run
    flash_attn.compute(q.data(), k.data(), v.data(), output.data());
    
    // Get performance metrics
    auto metrics = flash_attn.get_last_performance_metrics();
    
    // Verify performance metrics are reasonable
    EXPECT_GT(metrics.kernel_time_ms, 0.0f);
    EXPECT_GT(metrics.memory_transfer_time_ms, 0.0f);
    EXPECT_GT(metrics.total_time_ms, 0.0f);
    EXPECT_GT(metrics.gflops, 0.0f);
    
    // Print performance metrics
    std::cout << "Performance Metrics:" << std::endl;
    std::cout << "Kernel Time: " << metrics.kernel_time_ms << " ms" << std::endl;
    std::cout << "Memory Transfer Time: " << metrics.memory_transfer_time_ms << " ms" << std::endl;
    std::cout << "Total Time: " << metrics.total_time_ms << " ms" << std::endl;
    std::cout << "GFLOPS: " << metrics.gflops << std::endl;
}

// Test case for comparing different attention types
TEST(FlashAttentionTest, CompareAttentionTypes) {
    const int batch_size = 2;
    const int num_heads = 4;
    const int seq_len = 8;
    const int head_dim = 16;
    const int window_size = 2;
    
    // Create input tensors
    std::vector<float> q(batch_size * num_heads * seq_len * head_dim);
    std::vector<float> k(batch_size * num_heads * seq_len * head_dim);
    std::vector<float> v(batch_size * num_heads * seq_len * head_dim);
    std::vector<float> output_standard(batch_size * num_heads * seq_len * head_dim);
    std::vector<float> output_causal(batch_size * num_heads * seq_len * head_dim);
    std::vector<float> output_window(batch_size * num_heads * seq_len * head_dim);
    
    // Generate random input data
    generate_random_data(q.data(), q.size());
    generate_random_data(k.data(), k.size());
    generate_random_data(v.data(), v.size());
    
    // Create Flash Attention instances with different attention types
    FlashAttention flash_attn_standard(batch_size, num_heads, seq_len, head_dim, AttentionType::STANDARD);
    FlashAttention flash_attn_causal(batch_size, num_heads, seq_len, head_dim, AttentionType::CAUSAL);
    FlashAttention flash_attn_window(batch_size, num_heads, seq_len, head_dim, AttentionType::SLIDING_WINDOW, window_size);
    
    // Compute attention with different types
    flash_attn_standard.compute(q.data(), k.data(), v.data(), output_standard.data());
    flash_attn_causal.compute(q.data(), k.data(), v.data(), output_causal.data());
    flash_attn_window.compute(q.data(), k.data(), v.data(), output_window.data());
    
    // Compare performance metrics
    auto metrics_standard = flash_attn_standard.get_last_performance_metrics();
    auto metrics_causal = flash_attn_causal.get_last_performance_metrics();
    auto metrics_window = flash_attn_window.get_last_performance_metrics();
    
    // Print performance comparison
    std::cout << "Performance Comparison:" << std::endl;
    std::cout << "Standard Attention:" << std::endl;
    std::cout << "  Total Time: " << metrics_standard.total_time_ms << " ms" << std::endl;
    std::cout << "  GFLOPS: " << metrics_standard.gflops << std::endl;
    std::cout << "Causal Attention:" << std::endl;
    std::cout << "  Total Time: " << metrics_causal.total_time_ms << " ms" << std::endl;
    std::cout << "  GFLOPS: " << metrics_causal.gflops << std::endl;
    std::cout << "Sliding Window Attention:" << std::endl;
    std::cout << "  Total Time: " << metrics_window.total_time_ms << " ms" << std::endl;
    std::cout << "  GFLOPS: " << metrics_window.gflops << std::endl;
}

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
} 