// MP Reduction
// Given a list (lst) of length n
// Output its sum = lst[0] + lst[1] + ... + lst[n-1];

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

__global__ void total(float * input, float * output, int len) {
    //@@ Load a segment of the input vector into shared memory
    //@@ Traverse the reduction tree
    //@@ Write the computed sum of the block to the output vector at the
    //@@ correct index
  
    // shared memory block for cache result 
    __shared__ float cache[BLOCK_SIZE * 2];
    // current thread id in the block
    int tx  = threadIdx.x; 
    int bx  = blockIdx.x;
    // current start position
    int start = bx * blockDim.x * 2;

    // load input to cache
    cache[tx] = (start + tx < len) ? input[start + tx] : 0.0f; 
    if( start + tx + BLOCK_SIZE  < len )
      cache[BLOCK_SIZE + tx] = input[start + tx + BLOCK_SIZE];
    else
      cache[BLOCK_SIZE + tx] = 0.0f;

    // Loop through different stride for a time, 
    // synchronize the threads after one update to the cache
    for( int stride = blockDim.x; stride > 0; stride /= 2){
      __syncthreads();
      if( tx < stride )
        cache[tx] += cache[tx + stride];
    }

    // for each block, we got a sum of the block
    if(tx == 0)
      output[bx] = cache[0];
}

int main(int argc, char ** argv) {
    int ii;
    wbArg_t args;
    float * hostInput;     // The input 1D list
    float * hostOutput;    // The output list
    float * deviceInput;
    float * deviceOutput;
    int numInputElements;  // number of elements in the input list
    int numOutputElements; // number of elements in the output list

    args = wbArg_read(argc, argv);

    wbTime_start(Generic, "Importing data and creating memory on host");
    hostInput = (float *) wbImport(wbArg_getInputFile(args, 0), &numInputElements);

    numOutputElements = numInputElements / (BLOCK_SIZE<<1);
    if (numInputElements % (BLOCK_SIZE<<1)) {
        numOutputElements++;
    }
    hostOutput = (float*) malloc(numOutputElements * sizeof(float));

    wbTime_stop(Generic, "Importing data and creating memory on host");

    wbLog(TRACE, "The number of input elements in the input is ", numInputElements);
    wbLog(TRACE, "The number of output elements in the input is ", numOutputElements);

    wbTime_start(GPU, "Allocating GPU memory.");
    //@@ Allocate GPU memory here
    wbCheck(cudaMalloc( (void **) &deviceInput, numInputElements * sizeof(float)));
    wbCheck(cudaMalloc( (void **) &deviceOutput, numOutputElements * sizeof(float)));

    wbTime_stop(GPU, "Allocating GPU memory.");

    wbTime_start(GPU, "Copying input memory to the GPU.");
    //@@ Copy memory to the GPU here
    wbCheck(cudaMemcpy( deviceInput, hostInput, numInputElements * sizeof(float), cudaMemcpyHostToDevice ));

    wbTime_stop(GPU, "Copying input memory to the GPU.");
    //@@ Initialize the grid and block dimensions here
    dim3 dimBlock( BLOCK_SIZE, 1, 1);
    dim3 dimGrid ( numOutputElements, 1, 1);

    wbTime_start(Compute, "Performing CUDA computation");
    //@@ Launch the GPU Kernel here
    total<<<dimGrid, dimBlock>>>(deviceInput, deviceOutput, numInputElements);

    cudaDeviceSynchronize();
    wbTime_stop(Compute, "Performing CUDA computation");

    wbTime_start(Copy, "Copying output memory to the CPU");
    //@@ Copy the GPU memory back to the CPU here
    wbCheck(cudaMemcpy( hostOutput, deviceOutput, numOutputElements * sizeof(float), cudaMemcpyDeviceToHost ));

    wbTime_stop(Copy, "Copying output memory to the CPU");

    /********************************************************************
     * Reduce output vector on the host
     * NOTE: One could also perform the reduction of the output vector
     * recursively and support any size input. For simplicity, we do not
     * require that for this lab.
     ********************************************************************/
    for (ii = 1; ii < numOutputElements; ii++) {
        hostOutput[0] += hostOutput[ii];
    }

    wbTime_start(GPU, "Freeing GPU Memory");
    //@@ Free the GPU memory here
    wbCheck(cudaFree(deviceInput));
    wbCheck(cudaFree(deviceOutput));

    wbTime_stop(GPU, "Freeing GPU Memory");

    wbSolution(args, hostOutput, 1);

    free(hostInput);
    free(hostOutput);

    return 0;
}
