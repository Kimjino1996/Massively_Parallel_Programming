#include <stdio.h>
#include <stdlib.h>// for rand(),malloc,free
#include <sys/time.h>
#include <math.h>
#ifdef DEBUG // debug mode
#define CUDA_CHECK(x) do{\
	(x);\
	cudaError_t e = cudaGetLastError();\
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

int A_col_B_row;
int A_row;
int B_col;
int GRID_row,GRID_col;

const int TILE_WIDTH=32;


// random data gen
void genData(float* ptr, unsigned int size){
	while(size){
		*ptr++=(float)size/(float)1000;
		size--;
	}
}

__global__ void matmul(float* g_C, const float* g_A, const float* g_B, const int a_row_b_col, const int b_col){

	__shared__ float s_A[TILE_WIDTH][TILE_WIDTH];
	__shared__ float s_B[TILE_WIDTH][TILE_WIDTH];
	int by= blockIdx.y; int bx=blockIdx.x;
	int ty = threadIdx.y; int tx = threadIdx.x;
	int gy=by*TILE_WIDTH +ty;
	int gx=bx*TILE_WIDTH +tx;
	float sum=0.0F;
	for (register int m =0; m<ceil(a_row_b_col/(float)TILE_WIDTH); ++m){
	
		s_A[ty][tx]=g_A[gy*a_row_b_col+(m*TILE_WIDTH+tx)];
		s_B[ty][tx]=g_B[(m*TILE_WIDTH+ty)*b_col+gx];
		__syncthreads();
		for(register int k =0; k<TILE_WIDTH;++k){
			sum+=s_A[ty][k]*s_B[k][tx];
		}
		__syncthreads();
	
	}
	g_C[gy*b_col+gx]=sum;
}

int main(int argc, char* argv[]){
	int temp1, temp2,temp3;
	float* pA=NULL;
	float* pB=NULL;
	float* pC=NULL;
	temp1=atoi(argv[1]);
	temp2=atoi(argv[2]);
	temp3=atoi(argv[3]);
	A_row=temp1;
	A_col_B_row=temp2;
	B_col=temp3;
	GRID_row=ceil(A_row/(float)TILE_WIDTH);
	GRID_col=ceil(B_col/(float)TILE_WIDTH);

	struct timeval start_time, end_time;
	

	pA=(float*)malloc(A_row*A_col_B_row*sizeof(float));
	pB=(float*)malloc(A_col_B_row*B_col*sizeof(float));
	pC=(float*)malloc(A_row*B_col*sizeof(float));

	//generate Data
	genData(pA,A_row*A_col_B_row);
	genData(pB,A_col_B_row*B_col);

	float* pAdev=NULL;
	float* pBdev=NULL;
	float* pCdev=NULL;
	CUDA_CHECK(cudaMalloc((void**)&pAdev,A_row*A_col_B_row*sizeof(float)));
	CUDA_CHECK(cudaMalloc((void**)&pBdev,A_col_B_row*B_col*sizeof(float)));
	CUDA_CHECK(cudaMalloc((void**)&pCdev,A_row*B_col*sizeof(float)));

	//copy from host to device
	CUDA_CHECK(cudaMemcpy(pAdev,pA,A_row*A_col_B_row*sizeof(float),cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(pBdev,pB,A_col_B_row*B_col*sizeof(float),cudaMemcpyHostToDevice));
	
	//get current time
	cudaThreadSynchronize();
	gettimeofday(&start_time,NULL);
	//CUDA:launch
	
	dim3 dimGrid(GRID_col,GRID_row,1);
	dim3 dimBlock(TILE_WIDTH,TILE_WIDTH,1);
	matmul<<<dimGrid,dimBlock>>>(pCdev,pAdev,pBdev,A_col_B_row,B_col);
	CUDA_CHECK(cudaPeekAtLastError());
	cudaThreadSynchronize();
	gettimeofday(&end_time,NULL);
	double operating_time=(double)(end_time.tv_sec)+double(end_time.tv_usec)/1000000.0-((double)(start_time.tv_sec)+(double)(start_time.tv_usec)/1000000.0);
	
	printf("Elapsed: %f seconds\n",(double)operating_time);

	//copy from device to host
	int i, j;
//	for(i=0;i<A_row;i++){
//		for(j=0;j<B_col;j++){
//			printf("pCdev:c[%4d][%4d]=%f\n",i,j,pCdev[i*B_col+j]);
//		}
//	}	
	CUDA_CHECK(cudaMemcpy(pC,pCdev,A_row*B_col*sizeof(float),cudaMemcpyDeviceToHost));
	CUDA_CHECK(cudaFree(pAdev));
	CUDA_CHECK(cudaFree(pBdev));
	CUDA_CHECK(cudaFree(pCdev));
	//print sample
	
	i=0;j=0;
	for(i =0; i<A_row; i++){
		for(j=0;j<B_col;j++){
			printf("c[%4d][%4d]=%f\n",i,j,pC[i*B_col+j]);
		}
	}
	//i=A_row/2;j=B_col/2;
	//printf("c[%4d][%4d]=%f\n",i,j,pC[i*B_col+j]);
	//i=A_row-1;j=B_col-1;
	//printf("c[%4d][%4d]=%f\n",i,j,pC[i*B_col+j]);
	//done
	return 0;
}
