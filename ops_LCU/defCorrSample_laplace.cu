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
// FORWARD KERNEL (Laplace Version)
// =================================================================================
template <typename scalar_t>
__global__ void defCorr_index_forward_kernel_laplace(
    const torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> volume,
    const torch::PackedTensorAccessor32<float,4,torch::RestrictPtrTraits> coords,
    const torch::PackedTensorAccessor32<scalar_t,6,torch::RestrictPtrTraits> offset,
    const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> sigma_map, // MODIFIED: variance_map -> sigma_map
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
  
    // 获取中心点对应的尺度参数 sigma
    scalar_t sigma = sigma_map[n][y][x][0];
//     offset[n][y][x][static_cast<int>(rd/2)][static_cast<int>(rd/2)][0] = 0.0f;
//     offset[n][y][x][static_cast<int>(rd/2)][static_cast<int>(rd/2)][1] = 0.0f;

    for (int i=0; i<rd; i++) {
        for (int j=0; j<rd; j++) {
            float ofsX = offset[n][y][x][i][j][0] + x0;
            float ofsY = offset[n][y][x][i][j][1] + y0;
            int ofsXFloor = floor(ofsX);
            int ofsYFloor = floor(ofsY);
            float dx = ofsX-ofsXFloor;
            float dy = ofsY-ofsYFloor;
    
            int x1 = static_cast<int>(ofsXFloor) - r + i;
            int x2 = x1+1;
            int y1 = static_cast<int>(ofsYFloor) - r + j;
            int y2 = y1+1;

            if (within_bounds(y1, x1, h2, w2)) {
      
                scalar_t Q11 = 0.0, Q21 = 0.0, Q12 = 0.0, Q22 = 0.0;
                    
                Q11 = volume[n][y][x][y1][x1];
                if(x2_bounds(x2,w2)) Q21 = volume[n][y][x][y1][x2];
                if(y2_bounds(y2,h2)) Q12 = volume[n][y][x][y2][x1];
                if(y2_bounds(y2,h2)&&x2_bounds(x2,w2)) Q22 = volume[n][y][x][y2][x2];
                
                scalar_t correlation_value = Q11 * scalar_t((1.0f - dy) * (1.0f - dx)) + 
                                            Q21 * scalar_t((1.0f - dy) * dx) + 
                                            Q12 * scalar_t(dy * (1.0f - dx)) + 
                                            Q22 * scalar_t(dy * dx);
                
                // --- MODIFIED: 计算并应用拉普拉斯权重 ---
                int dx_grid = i - r;
                int dy_grid = j - r;
                float dist_l2 = sqrtf(dx_grid * dx_grid + dy_grid * dy_grid); // L2 距离 |d|
                scalar_t weight = expf(-sqrtf(2.0f) * dist_l2 / (sigma + EPS));
                
                corr[n][i][j][y][x] = correlation_value * weight;
            }
        }
    }
}

// =================================================================================
// BACKWARD KERNEL (Laplace Version)
// =================================================================================
template <typename scalar_t>
__global__ void defCorr_index_backward_kernel_laplace(
    const torch::PackedTensorAccessor32<float,4,torch::RestrictPtrTraits> coords,
    const torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> volume,
    const torch::PackedTensorAccessor32<scalar_t,6,torch::RestrictPtrTraits> offset,
    const torch::PackedTensorAccessor32<scalar_t,4,torch::RestrictPtrTraits> sigma_map, // MODIFIED
    const torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> corr_grad,
   // --- MODIFICATION START ---
    // 梯度张量现在是 float 类型
    torch::PackedTensorAccessor32<float,5,torch::RestrictPtrTraits> volume_grad,
    torch::PackedTensorAccessor32<float,6,torch::RestrictPtrTraits> offset_grad,
    torch::PackedTensorAccessor32<float,4,torch::RestrictPtrTraits> sigma_grad,
    // --- MODIFICATION END ---
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

  scalar_t sigma = sigma_map[n][y][x][0];
//   offset[n][y][x][static_cast<int>(rd/2)][static_cast<int>(rd/2)][0] = 0.0f;
//   offset[n][y][x][static_cast<int>(rd/2)][static_cast<int>(rd/2)][1] = 0.0f;

  for (int i=0; i<rd; i++) {
    for (int j=0; j<rd; j++) {
        float ofsX = offset[n][y][x][i][j][0]+x0;
        float ofsY = offset[n][y][x][i][j][1]+y0;
        int ofsXFloor = floor(ofsX);
        int ofsYFloor = floor(ofsY);
        float dx = ofsX-ofsXFloor;
        float dy = ofsY-ofsYFloor;

        int x1 = static_cast<int>(ofsXFloor) - r + i;
        int x2 = x1+1;
        int y1 = static_cast<int>(ofsYFloor) - r + j;
        int y2 = y1+1;

        if (within_bounds(y1, x1, h2, w2)) {
          scalar_t Q11 = 0.0, Q21 = 0.0, Q12 = 0.0, Q22 = 0.0;

          // --- 重新计算前向传播的中间值 ---
          int dx_grid = i - r;
          int dy_grid = j - r;
          float dist_l2 = sqrtf(dx_grid * dx_grid + dy_grid * dy_grid);
          scalar_t weight = expf(-sqrtf(2.0f) * dist_l2 / (sigma + EPS));

          scalar_t current_corr_grad = corr_grad[n][i][j][y][x];

          // --- 1. 计算对 volume 和 offset 的梯度 ---
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

          // --- 2. MODIFIED: 计算对 sigma 的梯度 ---
          // dL/d_sigma = dL/d_corr * d_corr/d_sigma
          // d_corr/d_sigma = correlation_value * d_weight/d_sigma
          // d_weight/d_sigma = weight * (sqrt(2)*|d| / sigma^2)
          scalar_t correlation_value = Q11 * scalar_t((1.0f - dy) * (1.0f - dx)) + 
                                      Q21 * scalar_t((1.0f - dy) * dx) + 
                                      Q12 * scalar_t(dy * (1.0f - dx)) + 
                                      Q22 * scalar_t(dy * dx);
          
          float d_weight_d_sigma = (float)weight * (sqrtf(2.0f) * dist_l2 / ((float)sigma * (float)sigma + EPS));
          float grad_sigma = (float)current_corr_grad * (float)correlation_value * d_weight_d_sigma;
          
          atomicAdd(&sigma_grad[n][y][x][0], grad_sigma);
        }
    }
  }
}

// =================================================================================
// C++ DISPATCHER INTERFACE
// =================================================================================

// --- FORWARD DISPATCHER ---
std::vector<torch::Tensor> defCorr_laplace_index_cuda_forward(
    torch::Tensor volume,
    torch::Tensor coords,
    torch::Tensor offset,
    torch::Tensor sigma_map, // MODIFIED
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

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(volume.type(), "sampler_forward_kernel_laplace", ([&] {
    defCorr_index_forward_kernel_laplace<scalar_t><<<blocks, threads>>>(
      volume.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
      coords.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
      offset.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
      sigma_map.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(), // MODIFIED
      corr.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
      radius);
   }));

  return {corr};
}

// --- BACKWARD DISPATCHER ---
std::vector<torch::Tensor> defCorr_laplace_index_cuda_backward(
  torch::Tensor volume,
  torch::Tensor coords,
  torch::Tensor offset,
  torch::Tensor sigma_map,
  torch::Tensor corr_grad,
  int radius)
{
  const auto batch_size = volume.size(0);
  const auto ht = volume.size(1);
  const auto wd = volume.size(2);
  
  // --- MODIFICATION START ---
  // 1. 强制创建 Float32 类型的梯度张量
  auto grad_opts = volume.options().dtype(torch::kFloat32);
  auto volume_grad = torch::zeros_like(volume, grad_opts);
  auto offset_grad = torch::zeros_like(offset, grad_opts);
  auto sigma_grad = torch::zeros_like(sigma_map, grad_opts);
  // --- MODIFICATION END ---

  const dim3 blocks((wd + BLOCK - 1) / BLOCK, 
                    (ht + BLOCK - 1) / BLOCK, 
                    batch_size);

  const dim3 threads(BLOCK, BLOCK);

  // 注意：我们仍然根据 volume.type() 来分派模板类型，
  // 但核函数内部将使用 float 来进行原子累加。
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(volume.type(), "sampler_backward_kernel_laplace", ([&] {
    defCorr_index_backward_kernel_laplace<scalar_t><<<blocks, threads>>>(
      coords.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
      volume.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
      offset.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
      sigma_map.packed_accessor32<scalar_t,4,torch::RestrictPtrTraits>(),
      corr_grad.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
      // 传入FP32的梯度张量
      volume_grad.packed_accessor32<float,5,torch::RestrictPtrTraits>(),
      offset_grad.packed_accessor32<float,6,torch::RestrictPtrTraits>(),
      sigma_grad.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
      radius);
   }));
  
  // --- MODIFICATION START ---
  // 2. 将计算完成的FP32梯度转换回原始的输入类型(例如 half)
  return {volume_grad.to(volume.scalar_type()), 
          offset_grad.to(offset.scalar_type()), 
          sigma_grad.to(sigma_map.scalar_type())};
  // --- MODIFICATION END ---
}