#include <iostream>
#include "cuda.h"
#include "cuda_runtime_api.h"
#include "index.cuh"

/**
   Function to add integer arrays
*/
__global__ void add_array(int *a, int *b, int size){
  for (int i = 0; i < size; i++){
    a[i] += b[i];
  }
}

/**
   Function to set thread index for intialization of arrays
*/
__global__ void set_arrays(int *a, int *b, int size){  
  int i = getGlobalIdx_1D_1D(); // stored in register memory
  if (i >= 0 && i < size){
    a[i] = i;
    b[i] = i;
  }
}

int main(){
  // allocated on CPU memory space (Host)
  // int a[]={12}, b[]={1};

  static const int N = 10000;
  int *a = new int[N];
  int *b = new int[N];

  // allocated on GPU memory space (Device)
  int *d_a, *d_b;  
  cudaMalloc(&d_a, N*sizeof(int));
  cudaMalloc(&d_b, N*sizeof(int));

  // copy data to gpu for computations
  // cudaMemcpy(d_a, &a, N*sizeof(int), cudaMemcpyHostToDevice);
  // cudaMemcpy(d_b, &b, N*sizeof(int), cudaMemcpyHostToDevice);

  // perform computations using kernel written for GPU
  // add_array<<<1,1>>>(d_a, d_b, 1);
  dim3 block_dim(10,1,1);
  dim3 grid_dim(N/10,1,1);
  set_arrays<<<block_dim, grid_dim>>>(d_a, d_b, N);
  
  // copy result stored in d_a back to cpu for printing
  cudaMemcpy(a, d_a, N*sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(b, d_b, N*sizeof(int), cudaMemcpyDeviceToHost);
  
  // print the result
  for (int i = N-10; i < N; i++){
    printf("result[%d]=%d\n", i, a[i]);
  }
  
  // free up device memory
  cudaFree(d_a);
  cudaFree(d_b);

  delete [] a;
  delete [] b;
    
  return 0;
}
