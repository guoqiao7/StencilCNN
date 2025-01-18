#include <mma.h>
#include <cuda_runtime.h>
#include <iostream>
#include "../include/include.h"
#include <chrono>
#include <cudnn.h>
#include <vector>

using namespace nvcuda;

#define BLOCK_SIZE_ROW 32
#define BLOCK_SIZE_COL 64
#define HALO 3
#define D_BLOCK_SIZE_COL (BLOCK_SIZE_COL + HALO * 2)
#define D_BLOCK_SIZE_ROW (BLOCK_SIZE_ROW + HALO * 2)
#define PAD 2
#define SM_SIZE_COL (7 * D_BLOCK_SIZE_ROW + PAD)
#define SM_SIZE_ROW (D_BLOCK_SIZE_COL / 8)
#define UNIT_LENGTH 7
#define TENSOR_CORE_M 8
#define IDX(x, y, ldm) ((x) * (ldm) + (y))
#define WARP_PER_BLOCK 8
#define MMA_NUM 13
#define CHANNELS 3


void cuDNN(double** input, const double params[CHANNELS][UNIT_LENGTH * UNIT_LENGTH], double* output, int rows, int cols){
    int H = rows - 2*HALO;
    int W = cols - 2*HALO;

    // Allocate memory on GPU
    double *input_d, *kernel_d, *output_d;
    cudaMalloc(&input_d, CHANNELS * rows * cols * sizeof(double));
    cudaMalloc(&kernel_d, CHANNELS * UNIT_LENGTH * UNIT_LENGTH * sizeof(double));
    cudaMalloc(&output_d, H * W * sizeof(double));

    for (int c = 0; c < CHANNELS; ++c){
        cudaMemcpy(input_d + c * rows * cols, input[c], rows * cols * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(kernel_d + c * UNIT_LENGTH * UNIT_LENGTH, params[c], UNIT_LENGTH * UNIT_LENGTH *sizeof(double), cudaMemcpyHostToDevice);
        
    }

    // Initialize cuDNN
    cudnnHandle_t cudnn;
    CHECK_CUDNN(cudnnCreate(&cudnn));

    cudnnTensorDescriptor_t input_desc, output_desc;
    cudnnFilterDescriptor_t kernel_desc;
    cudnnConvolutionDescriptor_t conv_desc;

    CHECK_CUDNN(cudnnCreateTensorDescriptor(&input_desc));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(input_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_DOUBLE, 1, CHANNELS, rows , cols));

    CHECK_CUDNN(cudnnCreateFilterDescriptor(&kernel_desc));
    CHECK_CUDNN(cudnnSetFilter4dDescriptor(kernel_desc, CUDNN_DATA_DOUBLE, CUDNN_TENSOR_NCHW, 1, CHANNELS, UNIT_LENGTH, UNIT_LENGTH));

    CHECK_CUDNN(cudnnCreateTensorDescriptor(&output_desc));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(output_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_DOUBLE, 1, 1, H, W));

    CHECK_CUDNN(cudnnCreateConvolutionDescriptor(&conv_desc));
    CHECK_CUDNN(cudnnSetConvolution2dDescriptor(conv_desc, 
                                                0, 0, 
                                                1, 1, 
                                                1, 1, 
                                                CUDNN_CROSS_CORRELATION, 
                                                CUDNN_DATA_DOUBLE));

    // Select algorithm
    int returned_alog_count;
    cudnnConvolutionFwdAlgoPerf_t algoPerf;
    CHECK_CUDNN(cudnnGetConvolutionForwardAlgorithm_v7(cudnn,
                                                    input_desc, kernel_desc, conv_desc, output_desc,
                                                    1,
                                                    &returned_alog_count,
                                                    &algoPerf))
    cudnnConvolutionFwdAlgo_t algo = algoPerf.algo;

    // CHECK_CUDNN(cudnnFindConvolutionForwardAlgorithm(cudnn,
    //                                                 input_desc, kernel_desc, conv_desc, output_desc,
    //                                                 10, 
    //                                                 &returned_alog_count, 
    //                                                 perf_res));
    // cudnnConvolutionFwdAlgo_t algo = perf_res[0].algo;

    // cudnnConvolutionFwdAlgo_t algo;
    // CHECK_CUDNN(cudnnGetConvolutionForwardAlgorithm)(cudnn,
    //                                                 input_desc, kernel_desc, conv_desc, output_desc,
    //                                                 CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
    //                                                 0,
    //                                                 &algo);

    // Workspace size
    void *workspace = nullptr;
    size_t workspace_size = 0;
    CHECK_CUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnn,
                                                        input_desc, kernel_desc, conv_desc, output_desc,
                                                        algo, &workspace_size));
    cudaMalloc(&workspace, workspace_size);
    
    // Perform convolution
    const double alpha = 1.0f, beta = 0.0f;
    CHECK_CUDNN(cudnnConvolutionForward(cudnn, &alpha,
                                        input_desc, input_d, kernel_desc, kernel_d, 
                                        conv_desc, algo, 
                                        workspace, workspace_size,
                                        &beta, 
                                        output_desc, output_d));
    
    cudaMemcpy(output, output_d, H * W * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(input_d);
    cudaFree(kernel_d);
    cudaFree(output_d);
    cudaFree(workspace);
    cudnnDestroyTensorDescriptor(input_desc);
    cudnnDestroyTensorDescriptor(output_desc);
    cudnnDestroyFilterDescriptor(kernel_desc);
    cudnnDestroyConvolutionDescriptor(conv_desc);
    cudnnDestroy(cudnn);
}

int main() {
    const int H = 2048, W = 2048;

    int rows = H + 2 * HALO;
    int cols = W + 2 * HALO;
    size_t matrix_size = (unsigned long)rows * cols * sizeof(double);
    
    double **input = (double **)malloc(CHANNELS * sizeof(double *));
    for(int i = 0; i < CHANNELS; i++){
        input[i] = (double *)malloc(matrix_size);
    }

    initialData(input, CHANNELS, rows * cols);

    double params[CHANNELS][UNIT_LENGTH * UNIT_LENGTH];
    for(int c = 0; c < CHANNELS; c++){
        for(int i = 0; i < UNIT_LENGTH * UNIT_LENGTH; i++){
            params[c][i] = 1.0 / (UNIT_LENGTH * UNIT_LENGTH);
        }
    } 

    // -----------------------------------------CUDNN-----------------------------------------------
    std::vector<double> output_cudnn(H * W);
    cuDNN(input, params, output_cudnn.data(), rows, cols);


    printf("output_cudnn\n");
    for(int i = 0; i < 7; i++){
        for(int j = 0; j < 10; j ++){
            printf("%.2f ", output_cudnn[i * W + j]);
        }
        printf("\n");
    }


    for(int c = 0; c < CHANNELS; c++){
        free(input[c]);
    } 
    free(input);
    return 0;
}
