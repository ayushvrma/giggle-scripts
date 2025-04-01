cmake_minimum_required(VERSION 3.10)
project(flash_attention CUDA CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find CUDA
find_package(CUDA REQUIRED)

# Find GTest
find_package(GTest REQUIRED)

# Add source files
set(SOURCES
    flash_attention.cpp
    flash_attention_kernel.cu
)

# Create library
cuda_add_library(flash_attention SHARED ${SOURCES})

# Set include directories
target_include_directories(flash_attention PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CUDA_INCLUDE_DIRS}
)

# Set compile options
target_compile_options(flash_attention PRIVATE
    -O3
    -arch=sm_60
    -std=c++17
)

# Link against CUDA libraries
target_link_libraries(flash_attention PRIVATE
    ${CUDA_LIBRARIES}
)

# Add test executable
add_executable(flash_attention_test test_flash_attention.cpp)

# Link test executable
target_link_libraries(flash_attention_test PRIVATE
    flash_attention
    GTest::GTest
    GTest::Main
    ${CUDA_LIBRARIES}
)

# Add test to CTest
add_test(NAME flash_attention_test COMMAND flash_attention_test)

# Install rules
install(TARGETS flash_attention
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

install(FILES flash_attention.hpp
    DESTINATION include
) 