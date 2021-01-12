# 대규모 병렬 프로그래밍을 활용한 여러 프로그램들.

## Scalable_matrix_add

- nvcc matadd-dev-ext.cu -o ./matadd-dev-ext
- ./matadd-dev-ext 1000 500
- 크기가 변경 가능하고 서로 같은 크기의 A,B 행렬에 대하여 덧셈을 실시

## Matmul_with_sharedMem

- nvcc matadd-shared_ext.cu -o ./matadd-shared-ext 로 실행팡리 생성
- ./matadd-shared-ext 1000 500 2000 으로 실행
- 변화가능한 크기의 행렬 A, B 를 Shared Mem 을 활용하여 곱하고 output 을 생성한다 


## Convolution_with_mask

- make 를 통하여 compile
- ./convolution m n  == > m*n 크기의 이미지에 대해 convolution 수행함
- 다양한 크기의 이미지에 대해 convolution 수행

## CountingSort_with_Scan

- nvcc sort.cu -o sort
- ./sort 10000
- 여러 크기의 input vector 에 대해서 scan 을 통해서 counter table 을 생성하고 Offset 을 정해주고 이에 맞게 buffer 를 채운다.
- vector에 들어가는 정수의 최댓값은 100이다.