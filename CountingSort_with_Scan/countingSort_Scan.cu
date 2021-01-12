#include <stdio.h>
#include <stdlib.h>
#include<algorithm>
#include <math.h>
#define BLOCK_SIZE 1024
#define COUNT_NUMBER 101
using namespace std;

//INSERT CODE HERE---------------------------------

__global__ void counting(int* g_A, int* g_C,int counting_size ) {

	__shared__ int count_arr[2][COUNT_NUMBER];//edit

	int tx=threadIdx.x;
	int index=tx+blockIdx.x*blockDim.x;
	if(tx<COUNT_NUMBER){
		count_arr[0][tx]=0;
		count_arr[1][tx]=0;
	}
	__syncthreads();
	
	if(index<counting_size){
		atomicAdd(&count_arr[1][g_A[index]],1);
	}


	int flag1=0;
	int flag2=1;
	int temp;
	if(tx<COUNT_NUMBER){
		for (int stride = 1; stride <= 64; stride = stride* 2) {
			__syncthreads();
			temp=flag1;
			flag1=flag2;
			flag2=temp;
			if(tx-stride>=0){
				count_arr[flag2][tx]=count_arr[flag1][tx]+count_arr[flag1][tx-stride];
			}
			else{
				count_arr[flag2][tx]=count_arr[flag1][tx];
			}
		}
	}

	if(tx<COUNT_NUMBER){
		atomicAdd(&(g_C[tx]),count_arr[flag2][tx]);
	}
}

void verify(int* src, int*result, int input_size){
	sort(src, src+input_size);
	long long match_cnt=0;
	for(int i=0; i<input_size;i++)
	{
		if(src[i]==result[i])
			match_cnt++;
	}

	if(match_cnt==input_size)
		printf("TEST PASSED\n\n");
	else
		printf("TEST FAILED\n\n");

}

void genData(int* ptr, unsigned int size) {
	while (size--) {
		*ptr++ = (int)(rand() % 101);
	}
}

int main(int argc, char* argv[]) {
	int* pSource = NULL;
	int* pResult = NULL;
	int input_size=0;

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	if (argc == 2)
		input_size=atoi(argv[1]);
	else
	{
    		printf("\n    Invalid input parameters!"
	   		"\n    Usage: ./sort <input_size>"
           		"\n");
        	exit(0);
	}

	//allocate host memory
	pSource=(int*)malloc(input_size*sizeof(int));
	pResult=(int*)malloc(input_size*sizeof(int));
	
	// generate source data
	genData(pSource, input_size);
	

	// start timer
	cudaEventRecord(start, 0);

	//INSERT CODE HERE--------------------
	int* pSdev=NULL;
	int* pRdev=NULL;
	int*pCount=NULL;
	pCount=(int*)malloc(COUNT_NUMBER*sizeof(int));
	cudaMalloc((void**)&pSdev,input_size*sizeof(int));
	cudaMalloc((void**)&pRdev,COUNT_NUMBER*sizeof(int));
	cudaMemcpy(pSdev,pSource,input_size*sizeof(int),cudaMemcpyHostToDevice);
	
	dim3 dimBlock(BLOCK_SIZE,1,1);
	counting<<<ceil(input_size/float(BLOCK_SIZE)),dimBlock>>>(pSdev,pRdev,input_size);
	
	cudaMemcpy(pCount,pRdev,COUNT_NUMBER*sizeof(int),cudaMemcpyDeviceToHost);

	// 0일때 또 따시해주자.
	for(int i=0;i<=pCount[0]-1;i++){
		pResult[i]=0;
	}
	
	for (int k=1; k<=101; k++){
		for(int i=pCount[k]-1;i>pCount[k-1]-1;i--){
			pResult[i]=k;
		}
	}
  	

	// end timer
	float time;
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	printf("elapsed time = %f msec\n", time);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

    	printf("Verifying results..."); fflush(stdout);
	verify(pSource, pResult, input_size);
	fflush(stdout);

}

