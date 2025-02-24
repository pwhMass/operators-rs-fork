template<int ArrSize, typename ArrayType>
struct ArrayStruct {
    ArrayType a[ArrSize];
};

#ifndef MAX_CONSTRAIN_NUM
#define MAX_CONSTRAIN_NUM
#endif

// 前向声明主模板
template<class Tmem, int ArrSize, typename ArrayType, int constrain_num>
static __device__ void rearrange_1(
    void *__restrict__ dst,
    void const *__restrict__ src,
    unsigned int const block_dim,
    unsigned int const block_len_total,                           // block_len 各元素的乘积
    const ArrayStruct<4, ArrayType> constrains[MAX_CONSTRAIN_NUM],// 约束条件数组
    const ArrayStruct<ArrSize, ArrayType> block_len,              // 各维度的长度
    const ArrayStruct<ArrSize, ArrayType> src_block_stride,       // 源tensor在各维度上的步长(bytes)
    const ArrayStruct<ArrSize, ArrayType> dst_block_stride,       // 目标tensor在各维度上的步长(bytes)
    const ArrayStruct<ArrSize, ArrayType> grid_len,               // 各维度的长度
    const ArrayStruct<ArrSize, ArrayType> src_grid_stride,        // 源tensor在各维度上的步长(bytes)
    const ArrayStruct<ArrSize, ArrayType> dst_grid_stride         // 目标tensor在各维度上的步长(bytes)
);

// constrain_num = 0 的特化版本
template<class Tmem, int ArrSize, typename ArrayType>
static __device__ void rearrange_1(
    void *__restrict__ dst,
    void const *__restrict__ src,
    unsigned int const block_dim,
    unsigned int const block_len_total,
    const ArrayStruct<4, ArrayType> constrains[MAX_CONSTRAIN_NUM],
    const ArrayStruct<ArrSize, ArrayType> block_len,
    const ArrayStruct<ArrSize, ArrayType> src_block_stride,
    const ArrayStruct<ArrSize, ArrayType> dst_block_stride,
    const ArrayStruct<ArrSize, ArrayType> grid_len,
    const ArrayStruct<ArrSize, ArrayType> src_grid_stride,
    const ArrayStruct<ArrSize, ArrayType> dst_grid_stride) {
    int remaining = threadIdx.x;
    if (remaining >= block_len_total) {
        return;
    }

    __shared__ int shared_src_offset;
    __shared__ int shared_dst_offset;

    if (threadIdx.x == 0) {
        int src_offset = 0;
        int dst_offset = 0;
        int remaining = blockIdx.x;

        for (int i = ArrSize - 1; i >= 0; i--) {
            int idx = remaining % grid_len.a[i];
            remaining /= grid_len.a[i];
            src_offset += idx * src_grid_stride.a[i];
            dst_offset += idx * dst_grid_stride.a[i];
        }
        shared_src_offset = src_offset;
        shared_dst_offset = dst_offset;
    }

    __syncthreads();

    int src_offset = shared_src_offset;
    int dst_offset = shared_dst_offset;

    for (int i = ArrSize - 1; i > 0; i--) {
        if (block_len.a[i] > 1) {
            int idx = remaining % block_len.a[i];
            remaining /= block_len.a[i];
            src_offset += idx * src_block_stride.a[i];
            dst_offset += idx * dst_block_stride.a[i];
        }
    }

    if (remaining >= block_len.a[0]) {
        return;
    }
    src_offset += remaining * src_block_stride.a[0];
    dst_offset += remaining * dst_block_stride.a[0];

    *reinterpret_cast<Tmem *>(reinterpret_cast<char *>(dst) + dst_offset) =
        *reinterpret_cast<const Tmem *>(reinterpret_cast<const char *>(src) + src_offset);
}

// 主模板的实现
template<class Tmem, int ArrSize, typename ArrayType, int constrain_num>
static __device__ void rearrange_1(
    void *__restrict__ dst,
    void const *__restrict__ src,
    unsigned int const block_dim,
    unsigned int const block_len_total,                           // block_len 各元素的乘积
    const ArrayStruct<4, ArrayType> constrains[MAX_CONSTRAIN_NUM],// 约束条件数组
    const ArrayStruct<ArrSize, ArrayType> block_len,              // 各维度的长度
    const ArrayStruct<ArrSize, ArrayType> src_block_stride,       // 源tensor在各维度上的步长(bytes)
    const ArrayStruct<ArrSize, ArrayType> dst_block_stride,       // 目标tensor在各维度上的步长(bytes)
    const ArrayStruct<ArrSize, ArrayType> grid_len,               // 各维度的长度
    const ArrayStruct<ArrSize, ArrayType> src_grid_stride,        // 源tensor在各维度上的步长(bytes)
    const ArrayStruct<ArrSize, ArrayType> dst_grid_stride         // 目标tensor在各维度上的步长(bytes)
) {

    int remaining = threadIdx.x;
    if (remaining >= block_len_total) {
        return;
    }

    // 声明共享内存
    __shared__ int shared_src_offset;
    __shared__ int shared_dst_offset;

    // 声明共享内存数组，确保至少有1个元素
    __shared__ int shared_constrains_grid_idx_multiple[constrain_num ? constrain_num : 1];

    if (threadIdx.x == 0) {
        // 计算当前block处理的数据在src和dst中的基础偏移(bytes)
        int src_offset = 0;
        int dst_offset = 0;
        int constrains_grid_idx_multiple[constrain_num ? constrain_num : 1] = {0};
        int remaining = blockIdx.x;

        for (int i = ArrSize - 1; i >= 0; i--) {
            int idx = remaining % grid_len.a[i];
            remaining /= grid_len.a[i];
            src_offset += idx * src_grid_stride.a[i];
            dst_offset += idx * dst_grid_stride.a[i];

// 处理所有约束条件
#pragma unroll
            for (int c = 0; c < constrain_num; c++) {
                if (i == constrains[c].a[0]) {
                    constrains_grid_idx_multiple[c] = idx * constrains[c].a[2];
                }
            }

            // 将结果存入共享内存
            shared_src_offset = src_offset;
            shared_dst_offset = dst_offset;
#pragma unroll
            for (int c = 0; c < constrain_num; c++) {
                shared_constrains_grid_idx_multiple[c] = constrains_grid_idx_multiple[c];
            }
        }
    }

    // 确保所有线程都能看到共享内存中的值
    __syncthreads();

    // 从共享内存加载约束条件的倍数
    int constrains_grid_idx_multiple[constrain_num ? constrain_num : 1];
#pragma unroll
    for (int c = 0; c < constrain_num; c++) {
        constrains_grid_idx_multiple[c] = shared_constrains_grid_idx_multiple[c];
    }

    // 所有线程直接使用计算好的偏移值
    int src_offset = shared_src_offset;
    int dst_offset = shared_dst_offset;

    for (int i = ArrSize - 1; i > 0; i--) {
        if (block_len.a[i] > 1) {
            int idx = remaining % block_len.a[i];
            remaining /= block_len.a[i];
            // 计算偏移量
            src_offset += idx * src_block_stride.a[i];
            dst_offset += idx * dst_block_stride.a[i];

// 检查所有约束条件
#pragma unroll
            for (int c = 0; c < constrain_num; c++) {
                if (constrains[c].a[3] != 0 && i == constrains[c].a[1]) {
                    if (constrains_grid_idx_multiple[c] + idx >= constrains[c].a[3]) {
                        return;
                    }
                }
            }
        }
    }

    // 单独处理第一个维度
    if (remaining >= block_len.a[0]) {
        return;
    }
    src_offset += remaining * src_block_stride.a[0];
    dst_offset += remaining * dst_block_stride.a[0];

// 检查第一个维度的约束条件
#pragma unroll
    for (int c = 0; c < constrain_num; c++) {
        if (constrains[c].a[3] != 0 && 0 == constrains[c].a[1]) {
            if (constrains_grid_idx_multiple[c] + remaining >= constrains[c].a[3]) {
                return;
            }
        }
    }

    // 执行数据拷贝，注意offset已经是字节偏移
    // 增加这个判断有助于优化程序
    *reinterpret_cast<Tmem *>(reinterpret_cast<char *>(dst) + dst_offset) =
        *reinterpret_cast<const Tmem *>(reinterpret_cast<const char *>(src) + src_offset);
}
