#include <iostream>

#ifdef DEBUG // debug mode
#define CUDA_CHECK(x)	do{\
	(x);\
	cudaError_t e =cudaGetLastError();\
	if(cudaSuccess!=e){\
		printf("cuda failure %s at %s : %d \n", \
				cudaGetErrorString(e),\
				__FILE__,__LINE__);\
		exit(1);\
	}\
}while(0)
#else
#define CUDA_CHECK(x) (x)
#endif

//kernel program for the deivce (GPU) : compiled by NVCC

__global__ void addKernel(int* c, const int* a, const int *b){
	int x =threadIdx.x;
	int y =threadIdx.y;
	int i =blockIdx.x*blockDim.x*blockDim.y+y*(blockDim.x)+x; // [y][x]=y*WIDTH+x;
	c[i]=a[i]+b[i];

}

// main program for the CPU :compiled by MS-vc++

int main(int argc,char *argv[]){
  //host-side data
  int temp,temp2 ;
  temp = atoi(argv[1]);
  temp2= atoi(argv[2]);

  const int WIDTH =temp;
  const int WIDTH2=temp2;


  int a[WIDTH][WIDTH2];
  int b[WIDTH][WIDTH2];
  int c[WIDTH][WIDTH2]={0};

  //make a,b matrices
  for (int y =0; y<WIDTH; ++y){
  	for (int x=0; x<WIDTH2; ++x){
		a[y][x]=y*10+x;
		b[y][x]=(y*10+x)*100;
	}
  }
  // device-side data
  int* dev_a=0;
  int* dev_b=0;
  int* dev_c=0;
  //allocate device memory;
  CUDA_CHECK(cudaMalloc((void**)&dev_a,WIDTH * WIDTH2  * sizeof(int)));
  CUDA_CHECK(cudaMalloc((void**)&dev_b,WIDTH * WIDTH2  * sizeof(int)));
  CUDA_CHECK(cudaMalloc((void**)&dev_c,WIDTH * WIDTH2  * sizeof(int)));

  //copy from host to device
  CUDA_CHECK(cudaMemcpy(dev_a,a,WIDTH*WIDTH2*sizeof(int),cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dev_b,b,WIDTH*WIDTH2*sizeof(int),cudaMemcpyHostToDevice));
  //laucnch a kernel on the GPU with one thread for each element
  dim3 dimBlock(16,16,1);
  addKernel <<< ceil(WIDTH*WIDTH2/256.0), dimBlock>>>(dev_c,dev_a,dev_b); // dev_c = dev_a+dev_b;
  CUDA_CHECK(cudaPeekAtLastError());
  //copy from device to host
  CUDA_CHECK(cudaMemcpy(c,dev_c,WIDTH * WIDTH2 * sizeof(int), cudaMemcpyDeviceToHost));
  //free devie memory
  CUDA_CHECK(cudaFree(dev_c));
  CUDA_CHECK(cudaFree(dev_a));
  CUDA_CHECK(cudaFree(dev_b));

  printf("\n 여기는 받아온값 \n");
  //print the result
  for (int y=0; y<WIDTH;++y){
  	for(int x=0; x<WIDTH2; ++x){
		printf("%7d \t",c[y][x]);
	}
	printf("\n");
  }
  //done 
  return 0;

}
