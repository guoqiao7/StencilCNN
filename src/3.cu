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
#define CHANNELS 3

const int stencil_size = 7;
const int stencil_radius = stencil_size / 2;

__constant__ double param_matrix_d[2 * 52 * TENSOR_CORE_M];

__global__ void stencilCNN(const double* __restrict__ in, double* __restrict__ out, const int ldm, const int* __restrict__ lookup_table1, const int* __restrict__ lookup_table2) {
    __shared__ double sharedmem[2][SM_SIZE_ROW * SM_SIZE_COL];

    int begin = IDX(blockIdx.x * BLOCK_SIZE_ROW, blockIdx.y * BLOCK_SIZE_COL + 1, ldm);
    int tid = threadIdx.x;

#pragma unroll
    for (int i = tid; i < D_BLOCK_SIZE_ROW * D_BLOCK_SIZE_COL; i += blockDim.x) {
        int row = i / D_BLOCK_SIZE_COL;
        int col = i % D_BLOCK_SIZE_COL;
        if (row < D_BLOCK_SIZE_ROW && col < D_BLOCK_SIZE_COL) {
            sharedmem[0][lookup_table1[i]] = in[begin + IDX(row, col, ldm)];
            sharedmem[1][lookup_table2[i]] = in[begin + IDX(row, col, ldm)];
        }
    }
    __syncthreads();

    int warp_id = threadIdx.x / 32;
    nvcuda::wmma::fragment<wmma::matrix_b, 8, 8, 4, double, wmma::row_major>param_frag[2][MMA_NUM];
#pragma unroll
    for(int i = 0; i < MMA_NUM; i++){
        nvcuda::wmma::load_matrix_sync(param_frag[0][i], param_matrix_d + i * 32, 8);
        nvcuda::wmma::load_matrix_sync(param_frag[1][i], param_matrix_d + 52 * 8 + i * 32, 8);
    }
    wmma::fragment<wmma::accumulator, 8, 8, 4, double> acc_frag;
    wmma::fragment<wmma::matrix_a, 8, 8, 4, double, wmma::row_major> in_frag;
    
    // wmma::fragment<wmma::matrix_b, 8, 8, 4, double, wmma::row_major> param_frag[2][MMA_NUM];
    // for (int i = 0; i < MMA_NUM; i++) {
    //     wmma::load_matrix_sync(param_frag[0][i], param_matrix_d[channel] + i * 32, 8);
    //     wmma::load_matrix_sync(param_frag[1][i], param_matrix_d[channel] + 52 * 8 + i * 32, 8);
    // }

    for (int col = warp_id * 28; col < warp_id * 28 + 28; col += UNIT_LENGTH) {
        wmma::fill_fragment(acc_frag, 0.0);
#pragma unroll
        for (int compute_idx = 0; compute_idx < MMA_NUM; compute_idx++) {
            wmma::load_matrix_sync(in_frag, sharedmem[0] + IDX(0, col + compute_idx * 4, SM_SIZE_COL), SM_SIZE_COL);
            wmma::mma_sync(acc_frag, in_frag, param_frag[0][compute_idx], acc_frag);
        }
#pragma unroll
        for (int compute_idx = 0; compute_idx < MMA_NUM; compute_idx++) {
            wmma::load_matrix_sync(in_frag, sharedmem[1] + IDX(0, col + compute_idx * 4, SM_SIZE_COL), SM_SIZE_COL);
            wmma::mma_sync(acc_frag, in_frag, param_frag[1][compute_idx], acc_frag);
        }
        wmma::store_matrix_sync(out + begin + IDX(HALO + col / 7, HALO, ldm), acc_frag, TENSOR_CORE_M, wmma::mem_row_major);
    }
}

void run_convolution(const double* input, double* output, const double* params, int rows, int cols) {
    double param_matrix_h[2][52 * 8] = {0.0};

    for (int col = 0; col < TENSOR_CORE_M; col++) {
        for(int i = 0; i < UNIT_LENGTH; i++) {
            for(int j = 0; j < UNIT_LENGTH; j++) {
                if (j >= col) {
                    param_matrix_h[0][(i * UNIT_LENGTH + j) * 8 + col] = params[i * UNIT_LENGTH + j - col];
                }
            }
        }
    }
    for (int col = 0; col < TENSOR_CORE_M; col++) {
        for(int i = 0; i < UNIT_LENGTH; i++) {
            for(int j = 0; j < UNIT_LENGTH; j++) {
                if (j < col) {
                    param_matrix_h[1][(i * UNIT_LENGTH + j) * 8 + col] = params[i * UNIT_LENGTH + j - col + 7];
                }
            }
        }
    }
    CHECK(cudaMemcpyToSymbol(param_matrix_d, param_matrix_h, 2 * 8 * 52 * sizeof(double)));
    
    const size_t array_size = rows * cols * sizeof(double);
    double *in_d = nullptr, *out_d = nullptr;
    CHECK(cudaMalloc(&in_d, array_size));
    CHECK(cudaMalloc(&out_d, array_size));
    CHECK(cudaMemset(in_d, 0, array_size));
    CHECK(cudaMemset(out_d, 0, array_size));
    CHECK(cudaMemcpy(in_d, input, array_size, cudaMemcpyHostToDevice));

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

    CUDAKERNELCHECK((stencilCNN<<<grid, block>>>(in_d, out_d, cols, lookup_table1_d, lookup_table2_d)));
    CHECK(cudaDeviceSynchronize()); 

    CHECK(cudaMemcpy(output, out_d, array_size, cudaMemcpyDeviceToHost));

    cudaFree(in_d); 
    cudaFree(out_d); 

    return;
}



double* cropMatrix(double* matrix, int rows, int cols){
    double* croppedMatrix = (double* )malloc((rows - 2 * HALO)*(cols - 2 * HALO)*sizeof(double));

    for (int i = HALO; i < rows - HALO; ++i){
        for (int j = HALO; j < cols - HALO; ++j){
            croppedMatrix[(i - HALO) * (cols - 2 * HALO) + (j - HALO)] = matrix[i * cols + j];
        }
    }
    return croppedMatrix;
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
    double **output_cpu = (double **)malloc(CHANNELS * sizeof(double *));
    for(int i = 0; i < CHANNELS; i++) {
        output_cpu[i] = (double *)malloc(matrix_size);
        memset(output_cpu[i], 0, matrix_size);
    }
    double **output_gpu = (double **)malloc(CHANNELS *sizeof(double *));
    for(int i = 0; i < CHANNELS; i++){
        output_gpu[i] = (double *)malloc(matrix_size);
        memset(output_gpu[i], 0, matrix_size);
    }
    // double *output_gpu = (double *)malloc(matrix_size);

    initialData(input, CHANNELS, rows * cols);

    double params[CHANNELS][UNIT_LENGTH * UNIT_LENGTH];
    for(int c = 0; c < CHANNELS; c++){
        for(int i = 0; i < UNIT_LENGTH * UNIT_LENGTH; i++){
            params[c][i] = 1.0 / (UNIT_LENGTH * UNIT_LENGTH);
        }
    } 


    // -------------------------------------------GPU----------------------------------------------
    std::chrono::steady_clock::time_point gpu_begin = std::chrono::steady_clock::now();
    for(int c = 0; c < CHANNELS; c++){
        run_convolution(input[c], output_gpu[c], params[c], rows, cols);
    }
    for (int i = 0; i < rows * cols; ++i){
        output_gpu[0][i] += output_gpu[1][i];
        output_gpu[0][i] += output_gpu[2][i];
    }
    std::chrono::steady_clock::time_point gpu_end = std::chrono::steady_clock::now();

    float GPU_TIME = std::chrono::duration_cast<std::chrono::microseconds>(gpu_end - gpu_begin).count()/1000.0f;

    double* cropped_gpu = cropMatrix(output_gpu[0], rows, cols);

    printMatrix("output_gpu", cropped_gpu, W, GPU_TIME);

    // printf("output_cudnn\n");
    // for(int i = 0; i < 7; i++){
    //     for(int j = 0; j < 10; j ++){
    //         printf("%.2f ", output_cudnn[i * W + j]);
    //     }
    //     printf("\n");
    // }

    for(int c = 0; c < CHANNELS; c++){
        free(input[c]);
        free(output_gpu[c]);
    } 
    free(cropped_gpu);
    free(input);
    free(output_gpu);
    return 0;
}
