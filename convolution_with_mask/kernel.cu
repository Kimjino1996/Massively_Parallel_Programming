__constant__ float Mc[FILTER_SIZE][FILTER_SIZE];
__global__ void convolution(Matrix N, Matrix P)
{
    
    int TILE_SIZE=8;
    //INSERT KERNEL CODE HERE
    __shared__ float N_ds[12][12];
    int ty= threadIdx.y;
    int tx=threadIdx.x;

    int row_o=blockIdx.y*TILE_SIZE+ty;
    int col_o=blockIdx.x*TILE_SIZE+tx;
    int row_i=row_o-FILTER_SIZE/2;
    int col_i=col_o-FILTER_SIZE/2;
    float output=0.0f;

    if ((row_i>=0)&&(row_i<N.height)&&(col_i>=0)&&(col_i<N.width))  
    {
        N_ds[ty][tx]=N.elements[row_i*N.width+col_i];
    }
    else{
        N_ds[ty][tx]=0.0f;
    }
    __syncthreads();

    if(ty<TILE_SIZE&&tx<TILE_SIZE){
        for(int i=0;i<FILTER_SIZE;i++){
            for(int j=0;j<FILTER_SIZE;j++){
                output+=Mc[i][j]*N_ds[i+ty][j+tx];            
            }
        }
        
        if(row_o<P.height&&col_o<P.width){
            P.elements[row_o*P.width+col_o]=output;
        }
    }
   
}
