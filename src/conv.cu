#include <mma.h>
#include <cuda_runtime.h>
#include <iostream>
#include "../include/include.h"
#include <chrono>
#include <cudnn.h>

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
#define CHANNELS3 3

const int stencil_size = 7;
const int stencil_radius = stencil_size / 2;

__constant__ double param_matrix_d_0[2 * 52 * TENSOR_CORE_M];
__constant__ double param_matrix_d_1[2 * 52 * TENSOR_CORE_M];
__constant__ double param_matrix_d_2[2 * 52 * TENSOR_CORE_M];

__global__ void stencilCNN(const double* __restrict__ in, double* __restrict__ out, int rows, const int ldm, const int* __restrict__ lookup_table1, const int* __restrict__ lookup_table2) {
    __shared__ double sharedmem[2][SM_SIZE_ROW * SM_SIZE_COL];

    int begin = IDX(blockIdx.x * BLOCK_SIZE_ROW, blockIdx.y * BLOCK_SIZE_COL + 1, ldm);
    int tid = threadIdx.x;

    int warp_id = threadIdx.x / 32;
    nvcuda::wmma::fragment<wmma::matrix_b, 8, 8, 4, double, wmma::row_major>param_frag[6][MMA_NUM];
#pragma unroll
    for(int i = 0; i < MMA_NUM; i++){
        nvcuda::wmma::load_matrix_sync(param_frag[0][i], param_matrix_d_0 + i * 32, 8);
        nvcuda::wmma::load_matrix_sync(param_frag[1][i], param_matrix_d_0 + 52 * 8 + i * 32, 8);
        nvcuda::wmma::load_matrix_sync(param_frag[2][i], param_matrix_d_1 + i * 32, 8);
        nvcuda::wmma::load_matrix_sync(param_frag[3][i], param_matrix_d_1 + 52 * 8 + i * 32, 8);
        nvcuda::wmma::load_matrix_sync(param_frag[4][i], param_matrix_d_2 + i * 32, 8);
        nvcuda::wmma::load_matrix_sync(param_frag[5][i], param_matrix_d_2 + 52 * 8 + i * 32, 8);
    }
    wmma::fragment<wmma::accumulator, 8, 8, 4, double> acc_frag;
    wmma::fragment<wmma::matrix_a, 8, 8, 4, double, wmma::row_major> in_frag;

    wmma::fragment<wmma::accumulator, 8, 8, 4, double> bias_frag;
    
    for (int c = 0; c < CHANNELS; c++){
#pragma unroll
        for (int i = tid; i < D_BLOCK_SIZE_ROW * D_BLOCK_SIZE_COL; i += blockDim.x) {
            int row = i / D_BLOCK_SIZE_COL;
            int col = i % D_BLOCK_SIZE_COL;
            if (row < D_BLOCK_SIZE_ROW && col < D_BLOCK_SIZE_COL) {
                sharedmem[0][lookup_table1[i]] = in[c * rows * ldm + begin + IDX(row, col, ldm)];
                sharedmem[1][lookup_table2[i]] = in[c * rows * ldm + begin + IDX(row, col, ldm)];
            }
        }
        __syncthreads();
    
        for (int col = warp_id * 28; col < warp_id * 28 + 28; col += UNIT_LENGTH) {
            wmma::load_matrix_sync(bias_frag, out + begin + IDX(HALO + col / 7, HALO, ldm), TENSOR_CORE_M, wmma::mem_row_major);
            wmma::fill_fragment(acc_frag, 0.0);
#pragma unroll
            for (int compute_idx = 0; compute_idx < MMA_NUM; compute_idx++) {
                wmma::load_matrix_sync(in_frag, sharedmem[0] + IDX(0, col + compute_idx * 4, SM_SIZE_COL), SM_SIZE_COL);
                wmma::mma_sync(bias_frag, in_frag, param_frag[2 * c][compute_idx], bias_frag);
            }
#pragma unroll
            for (int compute_idx = 0; compute_idx < MMA_NUM; compute_idx++) {
                wmma::load_matrix_sync(in_frag, sharedmem[1] + IDX(0, col + compute_idx * 4, SM_SIZE_COL), SM_SIZE_COL);
                wmma::mma_sync(bias_frag, in_frag, param_frag[2 * c + 1][compute_idx], bias_frag);
            } 
            wmma::store_matrix_sync(out + begin + IDX(HALO + col / 7, HALO, ldm), bias_frag, TENSOR_CORE_M, wmma::mem_row_major);
        }
    }
}

void stencil_convolution(const double** input, double* output, const double params[CHANNELS][UNIT_LENGTH * UNIT_LENGTH], int rows, int cols) {
    double param_matrix_h[6][52 * 8] = {0.0};

    for (int col = 0; col < TENSOR_CORE_M; col++) {
        for(int i = 0; i < UNIT_LENGTH; i++) {
            for(int j = 0; j < UNIT_LENGTH; j++) {
                if (j >= col) {
                    param_matrix_h[0][(i * UNIT_LENGTH + j) * 8 + col] = params[0][i * UNIT_LENGTH + j - col];
                }
            }
        }
    }
    for (int col = 0; col < TENSOR_CORE_M; col++) {
        for(int i = 0; i < UNIT_LENGTH; i++) {
            for(int j = 0; j < UNIT_LENGTH; j++) {
                if (j < col) {
                    param_matrix_h[1][(i * UNIT_LENGTH + j) * 8 + col] = params[0][i * UNIT_LENGTH + j - col + 7];
                }
            }
        }
    }
    
    for (int col = 0; col < TENSOR_CORE_M; col++) {
        for(int i = 0; i < UNIT_LENGTH; i++) {
            for(int j = 0; j < UNIT_LENGTH; j++) {
                if (j >= col) {
                    param_matrix_h[2][(i * UNIT_LENGTH + j) * 8 + col] = params[1][i * UNIT_LENGTH + j - col];
                }
            }
        }
    }
    for (int col = 0; col < TENSOR_CORE_M; col++) {
        for(int i = 0; i < UNIT_LENGTH; i++) {
            for(int j = 0; j < UNIT_LENGTH; j++) {
                if (j < col) {
                    param_matrix_h[3][(i * UNIT_LENGTH + j) * 8 + col] = params[1][i * UNIT_LENGTH + j - col + 7];
                }
            }
        }
    }

    for (int col = 0; col < TENSOR_CORE_M; col++) {
        for(int i = 0; i < UNIT_LENGTH; i++) {
            for(int j = 0; j < UNIT_LENGTH; j++) {
                if (j >= col) {
                    param_matrix_h[4][(i * UNIT_LENGTH + j) * 8 + col] = params[2][i * UNIT_LENGTH + j - col];
                }
            }
        }
    }
    for (int col = 0; col < TENSOR_CORE_M; col++) {
        for(int i = 0; i < UNIT_LENGTH; i++) {
            for(int j = 0; j < UNIT_LENGTH; j++) {
                if (j < col) {
                    param_matrix_h[5][(i * UNIT_LENGTH + j) * 8 + col] = params[2][i * UNIT_LENGTH + j - col + 7];
                }
            }
        }
    }
    size_t offset_bytes = 0;
    CHECK(cudaMemcpyToSymbol(param_matrix_d_0, param_matrix_h[0], 8 * 52 * sizeof(double), offset_bytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpyToSymbol(param_matrix_d_1, param_matrix_h[2], 8 * 52 * sizeof(double), offset_bytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpyToSymbol(param_matrix_d_2, param_matrix_h[4], 8 * 52 * sizeof(double), offset_bytes, cudaMemcpyHostToDevice));
    offset_bytes += 416 * sizeof(double);
    CHECK(cudaMemcpyToSymbol(param_matrix_d_0, param_matrix_h[1], 8 * 52 * sizeof(double), offset_bytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpyToSymbol(param_matrix_d_1, param_matrix_h[3], 8 * 52 * sizeof(double), offset_bytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpyToSymbol(param_matrix_d_2, param_matrix_h[5], 8 * 52 * sizeof(double), offset_bytes, cudaMemcpyHostToDevice));

    const size_t array_size = rows * cols * sizeof(double);

    double *in_d = nullptr;
    double *out_d = nullptr;
    CHECK(cudaMalloc(&in_d, CHANNELS * array_size));
    CHECK(cudaMalloc(&out_d, array_size));
    CHECK(cudaMemset(in_d, 0, CHANNELS * array_size));
    CHECK(cudaMemset(out_d, 0, array_size));
    for (int c = 0; c < CHANNELS; c++){
        CHECK(cudaMemcpy(in_d + c * rows * cols, input[c], array_size, cudaMemcpyHostToDevice));
    }

    dim3 grid((rows + BLOCK_SIZE_ROW - 1) / BLOCK_SIZE_ROW, (cols + BLOCK_SIZE_COL - 1) / BLOCK_SIZE_COL);
    // dim3 grid(1, 1);
    dim3 block(32 * WARP_PER_BLOCK);

    int lookup_table1_h[D_BLOCK_SIZE_ROW][D_BLOCK_SIZE_COL];
    int lookup_table2_h[D_BLOCK_SIZE_ROW][D_BLOCK_SIZE_COL];
    for (int i = 0; i < D_BLOCK_SIZE_ROW; i++) {
        for (int j = 0; j < D_BLOCK_SIZE_COL; j++) {
            if ((j + 1) % 8 != 0 && j < D_BLOCK_SIZE_COL - 2 * HALO - 1) {
                lookup_table1_h[i][j] = IDX(j / (UNIT_LENGTH + 1), UNIT_LENGTH * i + j % (UNIT_LENGTH + 1), SM_SIZE_COL);
            } else {
                lookup_table1_h[i][j] = SM_SIZE_ROW * SM_SIZE_COL - 1;
            }
            if ((j + 2) % 8 != 0 && j > 2 * HALO) {
                lookup_table2_h[i][j] = IDX((j - UNIT_LENGTH) / (UNIT_LENGTH + 1), UNIT_LENGTH * i + (j - UNIT_LENGTH) % (UNIT_LENGTH + 1), SM_SIZE_COL);
            } else {
                lookup_table2_h[i][j] = SM_SIZE_ROW * SM_SIZE_COL - 1;
            }
        }
    }

    int * lookup_table1_d;
    int * lookup_table2_d;
    CHECK(cudaMalloc(&lookup_table1_d, D_BLOCK_SIZE_ROW * D_BLOCK_SIZE_COL * sizeof(int)));
    CHECK(cudaMalloc(&lookup_table2_d, D_BLOCK_SIZE_ROW * D_BLOCK_SIZE_COL * sizeof(int)));
    CHECK(cudaMemcpy(lookup_table1_d, lookup_table1_h, D_BLOCK_SIZE_ROW * D_BLOCK_SIZE_COL * sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(lookup_table2_d, lookup_table2_h, D_BLOCK_SIZE_ROW * D_BLOCK_SIZE_COL * sizeof(int), cudaMemcpyHostToDevice));

    CUDAKERNELCHECK((stencilCNN<<<grid, block>>>(in_d, out_d, rows, cols, lookup_table1_d, lookup_table2_d)));
    CHECK(cudaDeviceSynchronize()); 

    CHECK(cudaMemcpy(output, out_d, array_size, cudaMemcpyDeviceToHost));

    cudaFree(in_d); 
    cudaFree(out_d); 

    return;
}

void cpu_stencil(const double** input, double** output, const double params[CHANNELS][UNIT_LENGTH * UNIT_LENGTH], int rows, int cols) {
    for (int c = 0; c < CHANNELS; ++c) { 
        for (int i = stencil_radius; i < rows - stencil_radius; ++i) { 
            for (int j = stencil_radius; j < cols - stencil_radius; ++j) { 
                double result = 0.0;
                // Perform Stencil
                for (int ki = -stencil_radius; ki <= stencil_radius; ++ki) {
                    for (int kj = -stencil_radius; kj <= stencil_radius; ++kj) {
                        int x = i + ki;
                        int y = j + kj;
                        result += input[c][x * cols + y] * 
                                  params[c][(ki + stencil_radius) * UNIT_LENGTH + (kj + stencil_radius)];
                    }
                }
                output[c][i * cols + j] = result; 
            }
        }
    }
    for (int i = 0; i < rows * cols; ++i){
        output[0][i] += output[1][i];
        output[0][i] += output[2][i];
    }
}


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

int main(int argc, char* argv[]) {
    int H = 0, W = 0;

    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <Height> <Width>\n";
        return 1;
    }
    
    try{
        H = std::stoi(argv[1]);
        W = std::stoi(argv[2]);
    }
    catch (const std::invalid_argument &e){
        std::cerr << "Invalid argument: cannot convert the parameter(s) to integer.\n";
        return 1;
    }
    catch (const std::out_of_range &e){
        std::cerr << "Argument out of range: the parameter(s) is(are) too large.\n";
        return 1;
    }

    double param[9]={0.0};
    double params[CHANNELS][UNIT_LENGTH * UNIT_LENGTH];
    for(int c = 0; c < CHANNELS; c++){
        for(int i = 0; i < UNIT_LENGTH * UNIT_LENGTH; i++){
            params[c][i] = 1.0 / (UNIT_LENGTH * UNIT_LENGTH);
        }
    } 

    if (argc == 4 && std::string(argv[3]) == "--custom")
    {
        int num_param = 9;
        printf("Please enter %d parameters:\n", num_param);
        double values[num_param];
        for (int i = 0; i < num_param; i++){
            int readNum = scanf("%lf", &values[i]);
            if (readNum != 1)
                return 1;
        }
        for (int i = 0; i < 9; i++){
            param[i] = values[i];
        }
        param_time_fusion(params, param);
    }

    int rows = H + 2 * HALO;
    int cols = W + 2 * HALO;
    size_t matrix_size = (unsigned long)rows * cols * sizeof(double);
    
    double **input = (double **)malloc(CHANNELS * sizeof(double *));
    for(int i = 0; i < CHANNELS; i++){
        input[i] = (double *)malloc(matrix_size);
    }
    double **output_cpu = (double **)malloc(CHANNELS * sizeof(double *));
    for(int i = 0; i < CHANNELS; i++) {
        output_cpu[i] = (double *)malloc(matrix_size);
        memset(output_cpu[i], 0, matrix_size);
    }

    double *output_gpu = (double *)malloc(matrix_size);
    memset(output_gpu, 0, matrix_size);

    initialData(input, CHANNELS, rows * cols);

    // -------------------------------------------CPU----------------------------------------------
    // std::chrono::steady_clock::time_point cpu_begin = std::chrono::steady_clock::now();
    // cpu_stencil((const double**)input, output_cpu, params, rows, cols);
    // std::chrono::steady_clock::time_point cpu_end = std::chrono::steady_clock::now();

    // -------------------------------------------GPU----------------------------------------------
    std::chrono::steady_clock::time_point warmup_begin = std::chrono::steady_clock::now();
    stencil_convolution((const double**)input, output_gpu, params, rows, cols);
    std::chrono::steady_clock::time_point warmup_end = std::chrono::steady_clock::now();
        
    std::chrono::steady_clock::time_point gpu_begin = std::chrono::steady_clock::now();
    stencil_convolution((const double**)input, output_gpu, params, rows, cols);
    std::chrono::steady_clock::time_point gpu_end = std::chrono::steady_clock::now();

    // -----------------------------------------CUDNN-----------------------------------------------
    std::vector<double> output_cudnn(H * W);
    std::chrono::steady_clock::time_point cudnn_warm_begin = std::chrono::steady_clock::now();
    cuDNN(input, params, output_cudnn.data(), rows, cols);
    std::chrono::steady_clock::time_point cudnn_warm_end = std::chrono::steady_clock::now();

    std::chrono::steady_clock::time_point cudnn_begin = std::chrono::steady_clock::now();
    // std::vector<double> output_cudnn(H * W);
    cuDNN(input, params, output_cudnn.data(), rows, cols);
    std::chrono::steady_clock::time_point cudnn_end = std::chrono::steady_clock::now();

    // ----------------------------------------------------------------------------------------------
    // float CPU_TIME = std::chrono::duration_cast<std::chrono::microseconds>(cpu_end - cpu_begin).count()/1000.0f;
    float GPU_TIME = std::chrono::duration_cast<std::chrono::microseconds>(gpu_end - gpu_begin).count()/1000.0f;
    float WARMUP_TIME = std::chrono::duration_cast<std::chrono::microseconds>(warmup_end - warmup_begin).count()/1000.0f;
    float CUDNN_TIME = std::chrono::duration_cast<std::chrono::microseconds>(cudnn_end - cudnn_begin).count()/1000.0f;
    float CUDNN_WARM = std::chrono::duration_cast<std::chrono::microseconds>(cudnn_warm_end - cudnn_warm_begin).count()/1000.0f;


    double* cropped_cpu = cropMatrix(output_cpu[0], rows, cols);
    double* cropped_gpu = cropMatrix(output_gpu, rows, cols);

    // printMatrix("output_cpu", cropped_cpu, H, W, CPU_TIME);
    // printMatrix("output_gpu", cropped_gpu, H, W, GPU_TIME, WARMUP_TIME);
    // printMatrix("output_cudnn", output_cudnn.data(), H, W, CUDNN_TIME, CUDNN_WARM);

    checkResult(cropped_gpu, output_cudnn.data(), H, W, "GPU", "cuDNN", GPU_TIME, WARMUP_TIME, CUDNN_TIME, CUDNN_WARM);

#ifdef SAVE_FILE
    save_to_txt(cropped_cpu, H, W, "cpu.txt");
    save_to_txt(cropped_gpu, H, W, "gpu.txt");
    save_to_txt(output_cudnn.data(), H, W, "cudnn.txt");
#endif


    for(int c = 0; c < CHANNELS; c++){
        free(input[c]);
        free(output_cpu[c]);
    } 
    free(cropped_cpu);
    free(cropped_gpu);
    free(input);
    free(output_gpu);
    free(output_cpu); 
    return 0;
}
