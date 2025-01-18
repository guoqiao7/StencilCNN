#include <time.h>
#include <random>

#define HALO 3
#define CHANNELS 3
#define UNIT_LENGTH 7
#define IDX(x, y, ldm) ((x) * (ldm) + (y))

#define CHECK(call)\
{\
    const cudaError_t error = call;\
    if (error != cudaSuccess)\
    {\
        printf("ERROR: %s: %d,", __FILE__,__LINE__);\
        printf("code: %d, reason:%s\n", error, cudaGetErrorString(error));\
        exit(1);\
    }\
}

#define CHECK_CUDNN(call)\
{\
    cudnnStatus_t err = call;\
    if (err != CUDNN_STATUS_SUCCESS){\
        std::cerr << "cuDNN error at " << __FILE__ << ":" << __LINE__ << " " << cudnnGetErrorString(err) << std::endl;\
        exit(EXIT_FAILURE);\
    }\
}

#define CUDAKERNELCHECK(expr)                                                               \
    do                                                                                        \
    {                                                                                         \
        expr;                                                                                 \
                                                                                              \
        cudaError_t __err = cudaGetLastError();                                               \
        if (__err != cudaSuccess)                                                             \
        {                                                                                     \
            printf("Line %d: '%s' failed: %s\n", __LINE__, #expr, cudaGetErrorString(__err)); \
            abort();                                                                          \
        }                                                                                     \
    } while (0)

void initialData(double** Matrix, int channel, int size){
    time_t t;
    srand((unsigned)time(&t));
    for (int c = 0; c < channel; c++){
        for (int i = 0; i < size; i++){
            Matrix[c][i] = (double)rand() / (double)RAND_MAX * 255.0;
        }
    }
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

void printMatrix(const char* name, double *M, int rows, const int cols, float time = 0.0, float warmup_time = 0.0)
{
    if(cols > 14){
        printf("Matrix<%s>: %fms     warmup_time:%f \n", name, time, warmup_time);
        for(int i = 0; i < 7; i++){
            for(int j = 0; j < 7; j++){
                printf("%.2f ", M[i * cols + j]);
            }
            printf(".......");
            for(int j = cols - 7; j < cols; j++){
                printf("%06.2f ", M[i * cols + j]);
            }
            printf("\n");
        }
        printf(".\n");
        printf(".\n");
        printf(".\n");
        for(int i = cols - 7; i < cols; i++){
            for(int j = 0; j < 7; j++){
                printf("%06.2f ", M[i * cols + j]);
            }
            printf(".......");
            for(int j = cols - 7; j < cols; j++){
                printf("%.2f ", M[i * cols + j]);
            }
            printf("\n");
        }
        printf("--------------------------------------------------------------------------------------------------------\n");
    }else{
        printf("Matrix<%s>: %fms     warmup_time:%f \n", name, time, warmup_time);
        for(int i = 0; i < rows; i++){
            for (int j = 0; j < cols; j++){
                printf("%06.2f ", M[i * cols + j]);
            }
            printf("\n");
        }
        printf("--------------------------------------------------------------------------------------------------------\n");
    }
}

void save_to_txt(double *arr, int rows, int cols, const char *filename)
{
    FILE *file = fopen(filename, "w");
    if (file == NULL)
    {
        printf("Error opening file!\n");
        return;
    }

    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            fprintf(file, "%.2f\t", arr[IDX(i, j, cols)]);
        }
        fprintf(file, "\n");
    }

    fclose(file);
}

void param_time_fusion(double param_after_fusion[][UNIT_LENGTH * UNIT_LENGTH], double* param){
    for (int c =0; c < CHANNELS; c++){
        param_after_fusion[c][16] = (3 * param[0] * param[0] * param[8] + 6 * param[0] * param[1] * param[7] + 6 * param[0] * param[2] * param[6] + 6 * param[0] * param[3] * param[5] + 3 * param[0] * param[4] * param[4] + 3 * param[1] * param[1] * param[6] + 6 * param[1] * param[3] * param[4] + 3 * param[2] * param[3] * param[3]);
        param_after_fusion[c][15] = (3 * param[0] * param[0] * param[7] + 6 * param[0] * param[1] * param[6] + 6 * param[0] * param[3] * param[4] + 3 * param[1] * param[3] * param[3]);
        param_after_fusion[c][14] = (3 * param[0] * param[0] * param[6] + 3 * param[0] * param[3] * param[3]);
        param_after_fusion[c][17] = (6 * param[0] * param[1] * param[8] + 6 * param[0] * param[2] * param[7] + 6 * param[0] * param[4] * param[5] + 3 * param[1] * param[1] * param[7] + 6 * param[1] * param[2] * param[6] + 6 * param[1] * param[3] * param[5] + 3 * param[1] * param[4] * param[4] + 6 * param[2] * param[3] * param[4]);
        param_after_fusion[c][18] = (6 * param[0] * param[2] * param[8] + 3 * param[0] * param[0] * param[5] + 3 * param[1] * param[1] * param[8] + 6 * param[1] * param[2] * param[7] + 6 * param[1] * param[4] * param[5] + 3 * param[2] * param[2] * param[6] + 6 * param[2] * param[3] * param[5] + 3 * param[2] * param[4] * param[4]);
        param_after_fusion[c][19] = (6 * param[1] * param[2] * param[8] + 3 * param[1] * param[1] * param[5] + 3 * param[2] * param[2] * param[7] + 6 * param[2] * param[4] * param[5]);
        param_after_fusion[c][20] = (3 * param[2] * param[2] * param[8] + 3 * param[2] * param[5] * param[5]);
        param_after_fusion[c][9] = (3 * param[0] * param[0] * param[5] + 6 * param[0] * param[1] * param[4] + 6 * param[0] * param[2] * param[3] + 3 * param[1] * param[1] * param[3]);
        param_after_fusion[c][8] = (3 * param[0] * param[0] * param[4] + 6 * param[0] * param[1] * param[3]);
        param_after_fusion[c][7] = 3 * param[0] * param[0] * param[3];
        param_after_fusion[c][10] = (6 * param[0] * param[1] * param[5] + 6 * param[0] * param[2] * param[4] + 3 * param[1] * param[1] * param[4] + 6 * param[1] * param[2] * param[3]);
        param_after_fusion[c][11] = (6 * param[0] * param[2] * param[5] + 3 * param[1] * param[1] * param[5] + 6 * param[1] * param[2] * param[4] + 3 * param[2] * param[2] * param[3]);
        param_after_fusion[c][12] = (6 * param[1] * param[2] * param[5] + 3 * param[2] * param[2] * param[4]);
        param_after_fusion[c][13] = 3 * param[2] * param[2] * param[5];
        param_after_fusion[c][2] = (3 * param[0] * param[0] * param[2] + 3 * param[0] * param[1] * param[1]);
        param_after_fusion[c][1] = 3 * param[0] * param[0] * param[1];
        param_after_fusion[c][0] = param[0] * param[0] * param[0];
        param_after_fusion[c][3] = (6 * param[0] * param[1] * param[2] + param[1] * param[1] * param[1]);
        param_after_fusion[c][4] = (3 * param[0] * param[2] * param[2] + 3 * param[1] * param[1] * param[2]);
        param_after_fusion[c][5] = 3 * param[1] * param[2] * param[2];
        param_after_fusion[c][6] = param[2] * param[2] * param[2];
        param_after_fusion[c][23] = (6 * param[0] * param[3] * param[8] + 6 * param[0] * param[4] * param[7] + 6 * param[0] * param[5] * param[6] + 6 * param[1] * param[3] * param[7] + 6 * param[1] * param[4] * param[6] + 6 * param[2] * param[3] * param[6] + 3 * param[3] * param[3] * param[5] + 3 * param[3] * param[4] * param[4]);
        param_after_fusion[c][22] = (6 * param[0] * param[3] * param[7] + 6 * param[0] * param[4] * param[6] + 6 * param[1] * param[3] * param[6] + 3 * param[3] * param[3] * param[4]);
        param_after_fusion[c][21] = (6 * param[0] * param[3] * param[6] + param[3] * param[3] * param[3]);
        param_after_fusion[c][24] = (6 * param[0] * param[4] * param[8] + 6 * param[0] * param[5] * param[7] + 6 * param[1] * param[3] * param[8] + 6 * param[1] * param[4] * param[7] + 6 * param[1] * param[5] * param[6] + 6 * param[2] * param[3] * param[7] + 6 * param[2] * param[4] * param[6] + 6 * param[3] * param[4] * param[5] + pow(param[4], 3));
        param_after_fusion[c][25] = (6 * param[0] * param[5] * param[8] + 6 * param[1] * param[4] * param[8] + 6 * param[1] * param[5] * param[7] + 6 * param[2] * param[3] * param[8] + 6 * param[2] * param[4] * param[7] + 6 * param[2] * param[5] * param[6] + 3 * param[3] * param[5] * param[5] + 3 * param[4] * param[4] * param[5]);
        param_after_fusion[c][26] = (6 * param[1] * param[5] * param[8] + 6 * param[2] * param[4] * param[8] + 6 * param[2] * param[5] * param[7] + 3 * param[4] * param[5] * param[5]);
        param_after_fusion[c][27] = (6 * param[2] * param[5] * param[8] + param[5] * param[5] * param[5]);
        param_after_fusion[c][30] = (6 * param[0] * param[6] * param[8] + 3 * param[0] * param[7] * param[7] + 6 * param[1] * param[6] * param[7] + 3 * param[2] * param[6] * param[6] + 3 * param[3] * param[3] * param[8] + 6 * param[3] * param[4] * param[7] + 6 * param[3] * param[5] * param[6] + 3 * param[4] * param[6] * param[6]);
        param_after_fusion[c][29] = (6 * param[0] * param[6] * param[7] + 3 * param[1] * param[6] * param[6] + 3 * param[3] * param[3] * param[7] + 6 * param[3] * param[4] * param[6]);
        param_after_fusion[c][28] = (3 * param[0] * param[6] * param[6] + 3 * param[3] * param[3] * param[6]);
        param_after_fusion[c][31] = (6 * param[0] * param[7] * param[8] + 6 * param[1] * param[6] * param[8] + 3 * param[1] * param[7] * param[7] + 6 * param[2] * param[6] * param[7] + 6 * param[3] * param[4] * param[8] + 6 * param[3] * param[5] * param[7] + 3 * param[4] * param[7] * param[7] + 6 * param[4] * param[5] * param[6]);
        param_after_fusion[c][32] = (3 * param[0] * param[8] * param[8] + 6 * param[1] * param[7] * param[8] + 6 * param[2] * param[6] * param[8] + 3 * param[2] * param[7] * param[7] + 6 * param[3] * param[5] * param[8] + 3 * param[4] * param[8] * param[8] + 6 * param[4] * param[5] * param[7] + 3 * param[5] * param[6] * param[6]);
        param_after_fusion[c][33] = (3 * param[1] * param[8] * param[8] + 6 * param[2] * param[7] * param[8] + 6 * param[4] * param[5] * param[8] + 3 * param[5] * param[7] * param[7]);
        param_after_fusion[c][34] = (3 * param[2] * param[8] * param[8] + 3 * param[5] * param[8] * param[8]);
        param_after_fusion[c][37] = (6 * param[3] * param[6] * param[8] + 3 * param[3] * param[7] * param[7] + 6 * param[4] * param[6] * param[7] + 3 * param[5] * param[6] * param[6]);
        param_after_fusion[c][36] = (6 * param[3] * param[6] * param[7] + 3 * param[4] * param[6] * param[6]);
        param_after_fusion[c][35] = 3 * param[3] * param[3] * param[6];
        param_after_fusion[c][38] = (6 * param[3] * param[7] * param[8] + 6 * param[4] * param[6] * param[8] + 3 * param[4] * param[7] * param[7] + 6 * param[5] * param[6] * param[7]);
        param_after_fusion[c][39] = (3 * param[3] * param[8] * param[8] + 6 * param[4] * param[7] * param[8] + 6 * param[5] * param[6] * param[8] + 3 * param[5] * param[7] * param[7]);
        param_after_fusion[c][40] = (3 * param[4] * param[8] * param[8] + 6 * param[5] * param[7] * param[8]);
        param_after_fusion[c][41] = 3 * param[5] * param[5] * param[8];
        param_after_fusion[c][44] = (3 * param[6] * param[6] * param[8] + 3 * param[6] * param[7] * param[7]);
        param_after_fusion[c][43] = 3 * param[6] * param[6] * param[7];
        param_after_fusion[c][42] = param[6] * param[6] * param[6];
        param_after_fusion[c][45] = (6 * param[6] * param[7] * param[8] + param[7] * param[7] * param[7]);
        param_after_fusion[c][46] = (3 * param[6] * param[8] * param[8] + 3 * param[7] * param[7] * param[8]);
        param_after_fusion[c][47] = 3 * param[7] * param[8] * param[8];
        param_after_fusion[c][48] = param[8] * param[8] * param[8];
    }
}