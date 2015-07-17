// MP Scan
// Given a list (lst) of length n
// Output its prefix sum = {lst[0], lst[0] + lst[1], lst[0] + lst[1] + ... + lst[n-1]}

#include    <wb.h>

#define BLOCK_SIZE 512 //@@ You can change this

#define wbCheck(stmt) do {                                                    \
        cudaError_t err = stmt;                                               \
        if (err != cudaSuccess) {                                             \
            wbLog(ERROR, "Failed to run stmt ", #stmt);                       \
            wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));    \
            return -1;                                                        \
        }                                                                     \
    } while(0)

__global__ void scan(float * input, float * output, int len) {
  //@@ Modify the body of this function to complete the functionality of
  //@@ the scan on the device
  //@@ You may need multiple kernel calls; write your kernels before this
  //@@ function and call them from here
  int tx = threadIdx.x; 
  int bx = blockIdx.x;
  int start = bx * blockDim.x * 2;

  __shared__ float cache[2 * BLOCK_SIZE];

  // load data
  cache[tx] = ((start + tx) < len )? input[start + tx]: 0.0f;
  if( start + tx + BLOCK_SIZE < len )
    cache[ BLOCK_SIZE + tx ] = input[ BLOCK_SIZE + start + tx];
  else
    cache[ BLOCK_SIZE + tx ] = 0.0f;

  for (int stride = 1; stride < BLOCK_SIZE + 1; stride *= 2){
    __syncthreads();
    int idx = (tx + 1)*stride*2 - 1;
    if( idx < 2 * BLOCK_SIZE)
      cache[idx] += cache[idx - stride];
  }

  for (int stride = BLOCK_SIZE / 2; stride > 0; stride /= 2){
    __syncthreads();
    int idx = (tx + 1)*stride*2 - 1;
    if( idx + stride < 2 * BLOCK_SIZE)
      cache[idx + stride] += cache[idx];
  }

  __syncthreads();

  if ( start + tx < len )  output[start + tx] = cache[tx];
  if ( start + tx + BLOCK_SIZE < len ) output[start + tx + BLOCK_SIZE] = cache[tx + BLOCK_SIZE];
}

__global__ void post_process(float * output, int len) {
  int tx = threadIdx.x; int bx = blockIdx.x;
  int start = bx * blockDim.x * 2;

  for (int offset = start; offset < len; offset += BLOCK_SIZE*2)
  {
    int last_checkpoint = offset - 1;

#ifdef DEBUG

    if ( tx == 1){
      printf("threadIdx: %d, blockIdx: %d, last_checkpoint: %d\n", tx, bx, last_checkpoint );
    } 

#undef DEBUG
#endif

    if ( last_checkpoint >= 0 ){
      // last checkpoint index
      if ( offset + tx < len )  
        output[offset + tx] += output[last_checkpoint];
      if ( offset + tx + BLOCK_SIZE < len ) 
        output[offset + tx + BLOCK_SIZE] += output[last_checkpoint];
    }

    __syncthreads();
  }
}


int main(int argc, char ** argv) {
    wbArg_t args;
    float * hostInput; // The input 1D list
    float * hostOutput; // The output list
    float * deviceInput;
    float * deviceOutput;
    int numElements; // number of elements in the list

    args = wbArg_read(argc, argv);

    wbTime_start(Generic, "Importing data and creating memory on host");
    hostInput = (float *) wbImport(wbArg_getInputFile(args, 0), &numElements);
    hostOutput = (float*) malloc(numElements * sizeof(float));
    wbTime_stop(Generic, "Importing data and creating memory on host");

    wbLog(TRACE, "The number of input elements in the input is ", numElements);

    wbTime_start(GPU, "Allocating GPU memory.");
    wbCheck(cudaMalloc((void**)&deviceInput, numElements*sizeof(float)));
    wbCheck(cudaMalloc((void**)&deviceOutput, numElements*sizeof(float)));
    wbTime_stop(GPU, "Allocating GPU memory.");

    wbTime_start(GPU, "Clearing output memory.");
    wbCheck(cudaMemset(deviceOutput, 0, numElements*sizeof(float)));
    wbTime_stop(GPU, "Clearing output memory.");

    wbTime_start(GPU, "Copying input memory to the GPU.");
    wbCheck(cudaMemcpy(deviceInput, hostInput, numElements*sizeof(float), cudaMemcpyHostToDevice));
    wbTime_stop(GPU, "Copying input memory to the GPU."); 
    //@@ Initialize the grid and block dimensions here
    dim3 dimGrid ( (numElements - 1) / BLOCK_SIZE + 1, 1, 1);
    dim3 dimBlock( BLOCK_SIZE, 1, 1);
    
    wbLog(TRACE, "Grid dimension: ", dimGrid.x, "x", dimGrid.y, "x", dimGrid.z);
    wbLog(TRACE, "Block dimension: ", dimBlock.x, "x", dimBlock.y, "x", dimBlock.z);

    wbTime_start(Compute, "Performing CUDA computation");
    //@@ Modify this to complete the functionality of the scan
    //@@ on the deivce
    scan<<<dimGrid, dimBlock>>>( deviceInput, deviceOutput, numElements );
    
    cudaDeviceSynchronize();

    post_process<<<dim3(1,1,1), dimBlock>>>( deviceOutput, numElements );

    cudaDeviceSynchronize();
    wbTime_stop(Compute, "Performing CUDA computation");

    wbTime_start(Copy, "Copying output memory to the CPU");
    wbCheck(cudaMemcpy(hostOutput, deviceOutput, numElements*sizeof(float), cudaMemcpyDeviceToHost));
    wbTime_stop(Copy, "Copying output memory to the CPU");

    wbTime_start(GPU, "Freeing GPU Memory");
    cudaFree(deviceInput);
    cudaFree(deviceOutput);
    wbTime_stop(GPU, "Freeing GPU Memory");

    wbSolution(args, hostOutput, numElements);

    free(hostInput);
    free(hostOutput);

    return 0;
}
