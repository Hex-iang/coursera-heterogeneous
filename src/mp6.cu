#include    <wb.h>
//@@ Image convolution with GPU implementation

#define wbCheck(stmt) do {                                                    \
        cudaError_t err = stmt;                                               \
        if (err != cudaSuccess) {                                             \
            wbLog(ERROR, "Failed to run stmt ", #stmt);                       \
            wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));    \
            return -1;                                                        \
        }                                                                     \
    } while(0)

#define Mask_width  5
#define Mask_radius Mask_width/2

#define TILE_WIDTH   16
#define BLOCK_WIDTH  (TILE_WIDTH + Mask_width - 1)

#define CLAMP(x) ( (x < 0.0f)? 0.0f: ( (x > 1.0f)? 1.0f: x ) )

//@@ INSERT CODE HERE
__global__ void convolution2D( float * input, const float * __restrict__  mask , float * output, 
                     int imageHeight,  int imageWidth,  int imageChannels)
{
  //@@ concrete code for 2D image convolution on GPU
  int tx = threadIdx.x; int ty = threadIdx.y;
  int row_out = ty + blockIdx.y * TILE_WIDTH;
  int col_out = tx + blockIdx.x * TILE_WIDTH;

  int row_in = row_out - Mask_radius;
  int col_in = col_out - Mask_radius;

  __shared__ float cache[BLOCK_WIDTH][BLOCK_WIDTH];

  for( int c = 0; c < imageChannels; c++)
  {
    //@@ Major workflow on image data loading
    if( (row_in >= 0 && row_in < imageHeight ) &&
        (col_in >= 0 && col_in < imageWidth  ) ){
      cache[ty][tx] = input[ (row_in * imageWidth + col_in) * imageChannels + c ];
    }else{
      cache[ty][tx] = 0.0f;
    }

    //@@ Need some code for boundary condition of image data loading

    __syncthreads();

    float out_val = 0.0f;
    if( tx < TILE_WIDTH && ty < TILE_WIDTH )
    {
      for( int y = 0; y < Mask_width; y++){
        for( int x = 0; x < Mask_width; x++){
          out_val += cache[ty + y][tx + x] * mask[y*Mask_width + x];
        }
      }
    }
  
    if( row_out < imageHeight && col_out < imageWidth )
      output[ (row_out * imageWidth + col_out) * imageChannels + c ] = CLAMP(out_val);

    __syncthreads();
  }
}


int main(int argc, char* argv[]) {
    wbArg_t args;
    int maskRows;
    int maskColumns;
    int imageChannels;
    int imageWidth;
    int imageHeight;
    char * inputImageFile;
    char * inputMaskFile;
    wbImage_t inputImage;
    wbImage_t outputImage;
    float * hostInputImageData;
    float * hostOutputImageData;
    float * hostMaskData;
    float * deviceInputImageData;
    float * deviceOutputImageData;
    float * deviceMaskData;

    args = wbArg_read(argc, argv); /* parse the input arguments */

    inputImageFile = wbArg_getInputFile(args, 0);
    inputMaskFile = wbArg_getInputFile(args, 1);

    inputImage = wbImport(inputImageFile);
    hostMaskData = (float *) wbImport(inputMaskFile, &maskRows, &maskColumns);

    assert(maskRows == 5); /* mask height is fixed to 5 in this mp */
    assert(maskColumns == 5); /* mask width is fixed to 5 in this mp */

    imageWidth = wbImage_getWidth(inputImage);
    imageHeight = wbImage_getHeight(inputImage);
    imageChannels = wbImage_getChannels(inputImage);

    outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);

    hostInputImageData = wbImage_getData(inputImage);
    hostOutputImageData = wbImage_getData(outputImage);

    wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

    wbTime_start(GPU, "Doing GPU memory allocation");
    cudaMalloc((void **) &deviceInputImageData, imageWidth * imageHeight * imageChannels * sizeof(float));
    cudaMalloc((void **) &deviceOutputImageData, imageWidth * imageHeight * imageChannels * sizeof(float));
    cudaMalloc((void **) &deviceMaskData, maskRows * maskColumns * sizeof(float));
    wbTime_stop(GPU, "Doing GPU memory allocation");


    wbTime_start(Copy, "Copying data to the GPU");
    cudaMemcpy(deviceInputImageData,
               hostInputImageData,
               imageWidth * imageHeight * imageChannels * sizeof(float),
               cudaMemcpyHostToDevice);
    cudaMemcpy(deviceMaskData,
               hostMaskData,
               maskRows * maskColumns * sizeof(float),
               cudaMemcpyHostToDevice);
    wbTime_stop(Copy, "Copying data to the GPU");


    wbTime_start(Compute, "Doing the computation on the GPU");
    //@@ INSERT CODE HERE
    dim3 dimGrid( (imageWidth - 1 ) / TILE_WIDTH + 1, (imageHeight - 1) / TILE_WIDTH + 1, 1 );
    dim3 dimBlock( BLOCK_WIDTH, BLOCK_WIDTH, 1);
    convolution2D<<<dimGrid, dimBlock>>>( deviceInputImageData, deviceMaskData, deviceOutputImageData, 
                                         imageHeight, imageWidth, imageChannels);

    wbTime_stop(Compute, "Doing the computation on the GPU");


    wbTime_start(Copy, "Copying data from the GPU");
    cudaMemcpy(hostOutputImageData,
               deviceOutputImageData,
               imageWidth * imageHeight * imageChannels * sizeof(float),
               cudaMemcpyDeviceToHost);
    wbTime_stop(Copy, "Copying data from the GPU");

    wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

    wbSolution(args, outputImage);

    cudaFree(deviceInputImageData);
    cudaFree(deviceOutputImageData);
    cudaFree(deviceMaskData);

    free(hostMaskData);
    wbImage_delete(outputImage);
    wbImage_delete(inputImage);

    return 0;
}
