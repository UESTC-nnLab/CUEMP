#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cuda_fp16.h>
#include <cuda_runtime.h>


#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Parallel.h>

#define BLOCK 16
#define EPS 1e-6f // 用于防止除以零的小常数

__forceinline__ __device__ bool within_bounds(int h, int w, int H, int W) {
  return h >= 0 && h < H && w >= 0 && w < W;
}
__forceinline__ __device__ bool x2_bounds(int x2, int W) {
  return x2 >= 0 && x2 < W;
}
__forceinline__ __device__ bool y2_bounds(int y2, int H) {
  return y2 >= 0 && y2 < H;
}

// =================================================================================
// FORWARD KERNEL
// =================================================================================
template <typename scalar_t>
__global__ void defCorr_index_forward_kernel_gaussian(
    const torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> volume,
    const torch::PackedTensorAccessor32<float,4,torch::RestrictPtrTraits> coords,
    const torch::PackedTensorAccessor32<scalar_t,6,torch::RestrictPtrTraits> offset,
    const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> variance_map, // 新增: 方差图
    torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> corr,
    int r)
{
  // batch index
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int n = blockIdx.z;

    const int h1 = volume.size(1);
    const int w1 = volume.size(2);
    const int h2 = volume.size(3);
    const int w2 = volume.size(4);

    if (!within_bounds(y, x, h1, w1)) {
        return;
    }

    float x0 = coords[n][0][y][x];
    float y0 = coords[n][1][y][x];
    int rd = 2*r + 1;
  
    // 获取中心点对应的方差
    scalar_t variance = variance_map[n][y][x][0];

    for (int i=0; i<rd; i++) {
        for (int j=0; j<rd; j++) {
            float ofsX = offset[n][y][x][i][j][0] + x0;
            float ofsY = offset[n][y][x][i][j][1] + y0;
            int ofsXFloor = floor(ofsX);
            int ofsYFloor = floor(ofsY);
            float dx = ofsX-ofsXFloor;
            float dy = ofsY-ofsYFloor;
    
            int x1 = static_cast<int>(ofsXFloor) - r + i; // 修正: i -> j (w), j -> i (h)
            int x2 = x1+1;
            int y1 = static_cast<int>(ofsYFloor) - r + j; // 修正
            int y2 = y1+1;

            if (within_bounds(y1, x1, h2, w2)) {
      
                scalar_t Q11 = 0.0;
                scalar_t Q21 = 0.0;
                scalar_t Q12 = 0.0;
                scalar_t Q22 = 0.0;
                    
                Q11 = volume[n][y][x][y1][x1];
                if(x2_bounds(x2,w2)) 
                    Q21 = volume[n][y][x][y1][x2];
                if(y2_bounds(y2,h2)) 
                    Q12 = volume[n][y][x][y2][x1];
                if(y2_bounds(y2,h2)&&x2_bounds(x2,w2))
                    Q22 = volume[n][y][x][y2][x2];
                
                scalar_t correlation_value = Q11 * scalar_t((1.0f - dy) * (1.0f - dx)) + 
                                            Q21 * scalar_t((1.0f - dy) * dx) + 
                                            Q12 * scalar_t(dy * (1.0f - dx)) + 
                                            Q22 * scalar_t(dy * dx);
                
                // --- 新增: 计算并应用高斯权重 ---
                int dx_grid = j - r; // 修正
                int dy_grid = i - r; // 修正
                float dist_sq = dx_grid * dx_grid + dy_grid * dy_grid;
                scalar_t weight = expf(-dist_sq / (2.0f * variance + EPS));
                
                corr[n][i][j][y][x] = correlation_value * weight;
            }
        }
    }
}

// =================================================================================
// BACKWARD KERNEL
// =================================================================================
template <typename scalar_t>
__global__ void defCorr_index_backward_kernel_gaussian(
    const torch::PackedTensorAccessor32<float,4,torch::RestrictPtrTraits> coords,
    const torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> volume,
    const torch::PackedTensorAccessor32<scalar_t,6,torch::RestrictPtrTraits> offset,
    const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> variance_map, // 新增
    const torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> corr_grad,
    torch::PackedTensorAccessor32<float,5,torch::RestrictPtrTraits> volume_grad,
    torch::PackedTensorAccessor32<float,6,torch::RestrictPtrTraits> offset_grad,
    torch::PackedTensorAccessor32<float,4,torch::RestrictPtrTraits> variance_grad, // 新增
    int r)
{
  // batch index
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  const int n = blockIdx.z;

  const int h1 = volume_grad.size(1);
  const int w1 = volume_grad.size(2);
  const int h2 = volume_grad.size(3);
  const int w2 = volume_grad.size(4);

  if (!within_bounds(y, x, h1, w1)) {
    return;
  }

  float x0 = coords[n][0][y][x];
  float y0 = coords[n][1][y][x];
  int rd = 2*r + 1;

  scalar_t variance = variance_map[n][y][x][0];

  for (int i=0; i<rd; i++) {
    for (int j=0; j<rd; j++) {
        float ofsX = offset[n][y][x][i][j][0]+x0;
        float ofsY = offset[n][y][x][i][j][1]+y0;
        int ofsXFloor = floor(ofsX);
        int ofsYFloor = floor(ofsY);
        float dx = ofsX-ofsXFloor;
        float dy = ofsY-ofsYFloor;

        int x1 = static_cast<int>(ofsXFloor) - r + i; // 修正
        int x2 = x1+1;
        int y1 = static_cast<int>(ofsYFloor) - r + j; // 修正
        int y2 = y1+1;

        if (within_bounds(y1, x1, h2, w2)) {
          scalar_t Q11 = 0.0, Q21 = 0.0, Q12 = 0.0, Q22 = 0.0;

          // --- 重新计算前向传播的中间值 ---
          int dx_grid = i - r; // 修正
          int dy_grid = j - r; // 修正
          float dist_sq = dx_grid * dx_grid + dy_grid * dy_grid;
          scalar_t weight = expf(-dist_sq / (2.0f * variance + EPS));

          scalar_t current_corr_grad = corr_grad[n][i][j][y][x];

          // --- 1. 计算对 volume 和 offset 的梯度 ---
          // 梯度需要乘以高斯权重
          float grad_pre_weight = (float)current_corr_grad * (float)weight;

          Q11 = volume[n][y][x][y1][x1];
          atomicAdd(&volume_grad[n][y][x][y1][x1], (float)scalar_t((1.0f - dy) * (1.0f - dx)) * grad_pre_weight);

          if(x2_bounds(x2,w2)) {
            Q21 = volume[n][y][x][y1][x2];
            atomicAdd(&volume_grad[n][y][x][y1][x2], (float)scalar_t((1.0f - dy) * dx) * grad_pre_weight);
          }
          if(y2_bounds(y2,h2)) {
            Q12 = volume[n][y][x][y2][x1];
            atomicAdd(&volume_grad[n][y][x][y2][x1], (float)scalar_t(dy * (1.0f - dx)) * grad_pre_weight);
          }
          if(y2_bounds(y2,h2)&&x2_bounds(x2,w2)) {
            Q22 = volume[n][y][x][y2][x2];
            atomicAdd(&volume_grad[n][y][x][y2][x2], (float)scalar_t(dy * dx) * grad_pre_weight);
          }
      
          atomicAdd(&offset_grad[n][y][x][i][j][1], (float)scalar_t(-Q11*(1.0f-dx) - Q21*dx + Q12*(1.0f-dx) + Q22*dx) * grad_pre_weight);
          atomicAdd(&offset_grad[n][y][x][i][j][0], (float)scalar_t(-Q11*(1.0f-dy) + Q21*(1.0f-dy) - Q12*dy + Q22*dy) * grad_pre_weight);

          // --- 2. 新增: 计算对 variance 的梯度 ---
          // dL/d_var = dL/d_corr * d_corr/d_var
          // d_corr/d_var = correlation_value * d_weight/d_var
          // d_weight/d_var = weight * (dist_sq / (2*var^2))
          scalar_t correlation_value = Q11 * scalar_t((1.0f - dy) * (1.0f - dx)) + 
                                      Q21 * scalar_t((1.0f - dy) * dx) + 
                                      Q12 * scalar_t(dy * (1.0f - dx)) + 
                                      Q22 * scalar_t(dy * dx);
          
          float d_weight_d_variance = (float)weight * (dist_sq / (2.0f * variance * variance + EPS));
          float grad_var = (float)current_corr_grad * (float)correlation_value * d_weight_d_variance;
          
          atomicAdd(&variance_grad[n][y][x][0], grad_var);
        }
    }
  }
}

// =================================================================================
// C++ DISPATCHER INTERFACE
// =================================================================================

// --- FORWARD DISPATCHER ---
std::vector<torch::Tensor> defCorr_gaussian_index_cuda_forward(
    torch::Tensor volume,
    torch::Tensor coords,
    torch::Tensor offset,
    torch::Tensor variance_map, // 新增
    int radius)
{
  const auto batch_size = volume.size(0);
  const auto ht = volume.size(1);
  const auto wd = volume.size(2);

  const dim3 blocks((wd + BLOCK - 1) / BLOCK, 
                    (ht + BLOCK - 1) / BLOCK, 
                    batch_size);
  
  const dim3 threads(BLOCK, BLOCK);

  auto opts = volume.options();
  torch::Tensor corr = torch::zeros(
    {batch_size, 2*radius+1, 2*radius+1, ht, wd}, opts);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(volume.type(), "sampler_forward_kernel_gaussian", ([&] {
    defCorr_index_forward_kernel_gaussian<scalar_t><<<blocks, threads>>>(
      volume.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
      coords.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
      offset.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
      variance_map.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(), // 新增
      corr.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
      radius);
   }));

  return {corr};
}

// --- BACKWARD DISPATCHER ---
std::vector<torch::Tensor> defCorr_gaussian_index_cuda_backward(
  torch::Tensor volume,
  torch::Tensor coords,
  torch::Tensor offset,
  torch::Tensor variance_map, // 新增
  torch::Tensor corr_grad,
  int radius)
{
  const auto batch_size = volume.size(0);
  const auto ht = volume.size(1);
  const auto wd = volume.size(2);

  auto volume_grad = torch::zeros_like(volume);
  auto offset_grad = torch::zeros_like(offset);
  auto variance_grad = torch::zeros_like(variance_map); // 新增

  const dim3 blocks((wd + BLOCK - 1) / BLOCK, 
                    (ht + BLOCK - 1) / BLOCK, 
                    batch_size);

  const dim3 threads(BLOCK, BLOCK);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(volume.type(), "sampler_backward_kernel_gaussian", ([&] {
    defCorr_index_backward_kernel_gaussian<scalar_t><<<blocks, threads>>>(
      coords.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
      volume.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
      offset.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
      variance_map.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(), // 新增
      corr_grad.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
      volume_grad.packed_accessor32<float,5,torch::RestrictPtrTraits>(),
      offset_grad.packed_accessor32<float,6,torch::RestrictPtrTraits>(),
      variance_grad.packed_accessor32<float,4,torch::RestrictPtrTraits>(), // 新增
      radius);
   }));

  // 返回三个梯度
  return {volume_grad, offset_grad, variance_grad};
}