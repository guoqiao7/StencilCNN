
    // Tensor Core 矩阵乘法
    int warp_id = threadIdx.x / 32;
    wmma::fragment<wmma::accumulator, 8, 8, 4, double> acc_frag;
    wmma::fragment<wmma::matrix_a, 8, 8, 4, double, wmma::row_major> in_frag;
    wmma::fragment<wmma::matrix_b, 8, 8, 4, double, wmma::row_major> param_frag[2][MMA_NUM];

    // 加载权重矩阵
    for (int i = 0; i < MMA_NUM; i++) {
        wmma::load_matrix_sync(param_frag[0][i], param_matrix_d[channel] + i * 32, 8);
        wmma::load_matrix_sync(param_frag[1][i], param_matrix_d[channel] + 52 * 8 + i * 32, 8);
    }

    // 计算卷积
    for (int col = warp_id * 28; col < warp_id * 28 + 28; col += UNIT_LENGTH) {
        wmma::fill_fragment(acc_frag, 0.0);
        for (int compute_idx = 0; compute_idx < MMA_NUM; compute_idx++) {
            wmma::load_matrix_sync(in_frag, sharedmem[0] + IDX(0, col + compute_idx * 4, SM_SIZE_COL), SM_SIZE_COL);
            wmma::mma_sync(acc_frag, in_frag, param_frag[0][compute_idx], acc_frag);

            wmma::load_matrix_sync(in_frag, sharedmem[1] + IDX(0, col + compute_idx * 4, SM_SIZE_COL), SM_SIZE_COL);
            wmma::mma_sync(acc_frag, in_frag, param_frag[1][compute_idx], acc_frag);
        }
        wmma::store_matrix_sync(out + begin + IDX(HALO + col / 7, HALO, ldm), acc_frag, TENSOR_CORE_M, wmma::mem_row_major);
    }