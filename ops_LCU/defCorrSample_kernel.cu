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

__forceinline__ __device__ bool within_bounds(int h, int w, int H, int W) {
  return h >= 0 && h < H && w >= 0 && w < W;
}
__forceinline__ __device__ bool x2_bounds(int x2, int W) {
  return x2 >= 0 && x2 < W;
}
__forceinline__ __device__ bool y2_bounds(int y2, int H) {
  return y2 >= 0 && y2 < H;
}

template <typename scalar_t>
__global__ void defCorr_index_forward_kernel(
    const torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> volume,
    const torch::PackedTensorAccessor32<float,4,torch::RestrictPtrTraits> coords,
    torch::PackedTensorAccessor32<scalar_t,6,torch::RestrictPtrTraits> offset,
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

    offset[n][y][x][static_cast<int>(rd/2)][static_cast<int>(rd/2)][0] = 0.0f;
    offset[n][y][x][static_cast<int>(rd/2)][static_cast<int>(rd/2)][1] = 0.0f;
  
  for (int i=0; i<rd; i++) {
    for (int j=0; j<rd; j++) {
        float ofsX = offset[n][y][x][i][j][0] + x0;
        float ofsY = offset[n][y][x][i][j][1] + y0;
        int ofsXFloor = floor(ofsX);
        int ofsYFloor = floor(ofsY);
        float dx = ofsX-ofsXFloor;
        float dy = ofsY-ofsYFloor;
  
        int x1 = static_cast<int>(ofsXFloor) - r + i; //i w ofsx
        int x2 = x1+1;
        int y1 = static_cast<int>(ofsYFloor) - r + j;
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
        
        
        corr[n][i][j][y][x] = Q11 * scalar_t((1.0f - dy) * (1.0f - dx)) + 
                              Q21 * scalar_t((1.0f - dy) * dx) + 
                              Q12 * scalar_t(dy * (1.0f - dx)) + 
                              Q22 * scalar_t(dy * dx);

      }
    }
  }
}

template <typename scalar_t>
__global__ void defCorr_index_backward_kernel(
    const torch::PackedTensorAccessor32<float,4,torch::RestrictPtrTraits> coords,
    const torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> volume,
    torch::PackedTensorAccessor32<scalar_t,6,torch::RestrictPtrTraits> offset,
    const torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> corr_grad,
    torch::PackedTensorAccessor32<scalar_t,5,torch::RestrictPtrTraits> volume_grad,
    torch::PackedTensorAccessor32<scalar_t,6,torch::RestrictPtrTraits> offset_grad,
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

    offset[n][y][x][static_cast<int>(rd/2)][static_cast<int>(rd/2)][0] = 0.0f;
    offset[n][y][x][static_cast<int>(rd/2)][static_cast<int>(rd/2)][1] = 0.0f;

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
          scalar_t Q11 = 0.0;
          scalar_t Q21 = 0.0;
          scalar_t Q12 = 0.0;
          scalar_t Q22 = 0.0;
                
          Q11 = volume[n][y][x][y1][x1];
          volume_grad[n][y][x][y1][x1] += scalar_t((1.0f - dy) * (1.0f - dx))*corr_grad[n][i][j][y][x];
          if(x2_bounds(x2,w2)) 
           {Q21 = volume[n][y][x][y1][x2];
           volume_grad[n][y][x][y1][x2] += scalar_t((1.0f - dy) * dx)*corr_grad[n][i][j][y][x];}
          if(y2_bounds(y2,h2)) 
           {Q12 = volume[n][y][x][y2][x1];
           volume_grad[n][y][x][y2][x1] += scalar_t(dy * (1.0f - dx))*corr_grad[n][i][j][y][x];}
          if(y2_bounds(y2,h2)&&x2_bounds(x2,w2))
           {Q22 = volume[n][y][x][y2][x2];
           volume_grad[n][y][x][y2][x2] += scalar_t(dy * dx)*corr_grad[n][i][j][y][x];}
      
          offset_grad[n][y][x][i][j][1] = scalar_t(-Q11*(1.0f-dx)-Q21*dx+Q12*(1.0f-dx)+Q22*dx)*corr_grad[n][i][j][y][x];
          offset_grad[n][y][x][i][j][0] = scalar_t(-Q11*(1.0f-dy)+Q21*(1.0f-dy)-Q12*dy+Q22*dy)*corr_grad[n][i][j][y][x];
        
      }
    }
  }
}


std::vector<torch::Tensor> defCorr_index_cuda_forward(
    torch::Tensor volume,
    torch::Tensor coords,
    torch::Tensor offset,
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

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(volume.type(), "sampler_forward_kernel", ([&] {
    defCorr_index_forward_kernel<scalar_t><<<blocks, threads>>>(
      volume.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
      coords.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
      offset.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
      corr.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
      radius);
   }));

  return {corr};

}

std::vector<torch::Tensor> defCorr_index_cuda_backward(
  torch::Tensor volume,
  torch::Tensor coords,
  torch::Tensor offset,
  torch::Tensor corr_grad,
  int radius)
{
  const auto batch_size = volume.size(0);
  const auto ht = volume.size(1);
  const auto wd = volume.size(2);

  auto volume_grad = torch::zeros_like(volume);
    auto offset_grad = torch::zeros_like(offset);

  const dim3 blocks((wd + BLOCK - 1) / BLOCK, 
                    (ht + BLOCK - 1) / BLOCK, 
                    batch_size);

  const dim3 threads(BLOCK, BLOCK);


  AT_DISPATCH_FLOATING_TYPES_AND_HALF(volume.type(), "sampler_backward_kernel", ([&] {
    defCorr_index_backward_kernel<scalar_t><<<blocks, threads>>>(
      coords.packed_accessor32<float,4,torch::RestrictPtrTraits>(),
      volume.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
      offset.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
      corr_grad.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
      volume_grad.packed_accessor32<scalar_t,5,torch::RestrictPtrTraits>(),
      offset_grad.packed_accessor32<scalar_t,6,torch::RestrictPtrTraits>(),
      radius);
   }));

  return {volume_grad, offset_grad};
}