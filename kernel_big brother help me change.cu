#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <assert.h>
#include "helper_cuda.h"
//#include "Utilities.cuh"
#include <chrono>
using namespace std;
using namespace std::chrono;
#include <cuda_runtime.h>
#include <cusparse_v2.h>
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
#include <iostream>
#include <fstream>  
#include <vector>
#include "device_launch_parameters.h"  ///  add to color your code
#include<string>
//#include "../common/common.h"
// haha
#define mian main
//#define ? ;
//#define ?,
////#define ? (;
//#define ?)
#define ture true
#define flase false
#define D(x) cout<<#x<<"="<<x<<endl;

double* SMVP(double* h_C_dense, int* nnz_return, int N);
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}


/***************************/
/* CUSPARSE ERROR CHECKING */
/***************************/
static const char* _cusparseGetErrorEnum(cusparseStatus_t error)
{
	switch (error)
	{

	case CUSPARSE_STATUS_SUCCESS:
		return "CUSPARSE_STATUS_SUCCESS";

	case CUSPARSE_STATUS_NOT_INITIALIZED:
		return "CUSPARSE_STATUS_NOT_INITIALIZED";

	case CUSPARSE_STATUS_ALLOC_FAILED:
		return "CUSPARSE_STATUS_ALLOC_FAILED";

	case CUSPARSE_STATUS_INVALID_VALUE:
		return "CUSPARSE_STATUS_INVALID_VALUE";

	case CUSPARSE_STATUS_ARCH_MISMATCH:
		return "CUSPARSE_STATUS_ARCH_MISMATCH";

	case CUSPARSE_STATUS_MAPPING_ERROR:
		return "CUSPARSE_STATUS_MAPPING_ERROR";

	case CUSPARSE_STATUS_EXECUTION_FAILED:
		return "CUSPARSE_STATUS_EXECUTION_FAILED";

	case CUSPARSE_STATUS_INTERNAL_ERROR:
		return "CUSPARSE_STATUS_INTERNAL_ERROR";

	case CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
		return "CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED";

	case CUSPARSE_STATUS_ZERO_PIVOT:
		return "CUSPARSE_STATUS_ZERO_PIVOT";
	}

	return "<unknown>";
}

inline void __cusparseSafeCall(cusparseStatus_t err, const char* file, const int line)
{
	if (CUSPARSE_STATUS_SUCCESS != err) {
		fprintf(stderr, "CUSPARSE error in file '%s', line %Ndims\Nobjs %s\nerror %Ndims: %s\nterminating!\Nobjs", __FILE__, __LINE__, err, \
			_cusparseGetErrorEnum(err)); \
			cudaDeviceReset(); assert(0); \
	}
}

extern "C" void cusparseSafeCall(cusparseStatus_t err) { __cusparseSafeCall(err, __FILE__, __LINE__); }





double* tester(int N, double ratio) {
	////int** a;/*
	double* a = (double*)malloc(sizeof(double*) * N * N + 1); //////////////////////////  不应该是N*N


	//cout << sizeof(a) << " ->size of a";
	int count = 0;//记录随即产生1的个数
	srand(time(0)); //初始种子
	int M = N; // cow = col
	double PE = ratio / N;

	//初始数组
	for (int i = 0; i < M; i++)
		for (int j = 0; j < N; j++)
		{
			a[i * N + j] = rand() % 2;
			if (a[i * N + j] == 1) count++;
		}

	//按比例随即修正
	if (count > PE* M* N)
	{
		for (int n = count, i, j; n - PE * M * N > 0; )
		{
			i = rand() % M;
			j = rand() % N;
			if (1 == a[i * N + j])
			{
				a[i * N + j] = 0;
				n--;
			}
		}
	}
	else if (count < PE* M* N)
	{
		for (int n = count, i, j; PE * M * N - n > 0; )
		{
			i = rand() % M;
			j = rand() % N;
			if (0 == a[i * N + j])
			{
				a[i * N + j] = 1;
				n++;
			}
		}
	}


	return a;

}
void print_matrix(double* h_A_dense, int N) {
	for (int j = 0; j < N; j++) {
		for (int i = 0; i < N; i++) {
			//printf("%f \t", h_C_dense[i * N + j]);
			if (h_A_dense[j * N + i] != 0) {
				cout << 1 << " ";
			}
			else
			{
				cout << 0 << " ";
			}
		}

		printf("\n");
	}
	printf("print_matrix done\n");
}

/********/
/* MAIN */
/********/
int main()
{
	// --- Initialize cuSPARSE
	cusparseHandle_t handle;    cusparseCreate(&handle);

	/**************************/
	/* SETTING UP THE PROBLEM */
	/**************************/
	const int N = 10;                // --- Number of rows and columns

	// --- Host side dense matrices
	double* h_A_dense = (double*)malloc(N * N * sizeof(*h_A_dense));
	double* h_C_dense = (double*)malloc(N * N * sizeof(*h_C_dense));
	double* h_B_dense;

	//// --- Column-major ordering
	//h_A_dense[0] = 0.4612;  h_A_dense[4] = -0.0006;     h_A_dense[8] = 0.3566;     h_A_dense[12] = 0.0;
	//h_A_dense[1] = -0.0006; h_A_dense[5] = 0.4640;      h_A_dense[9] = 0.0723;     h_A_dense[13] = 0.0;
	//h_A_dense[2] = 0.3566;  h_A_dense[6] = 0.0723;      h_A_dense[10] = 0.7543;     h_A_dense[14] = 0.0;
	//h_A_dense[3] = 0.;      h_A_dense[7] = 0.0;         h_A_dense[11] = 0.0;        h_A_dense[15] = 0.1;
		//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	float arr1[16384];
	float arr2[16384];
	int i = 0;
	string path = "toy.txt";

	ifstream myfile(path);// delaunay_n14  toy
	if (!myfile) {
		cout << "Unable to open myfile";
		exit(1); // terminate with error  
	}
	else
	{
		fstream f(path);
		vector<string> words;
		string line;
		while (getline(f, line))
		{
			words.push_back(line);


		}	//dictionary.txt在csdn里面可以下载，里面有4万多个单词，相当于一个字典	
		cout << "Num of edge:" << words.size() << endl;
		/*for (int i = 0; i < words.size(); i++)
		{		cout << words[i] << endl;	}*/
		char str[256] = { 0 };
		for (int i = 0; i < N * N; i++)
		{

			h_A_dense[i] = 0;
			if (i % N == i / N)
			{
				h_A_dense[i] = 1;
			}
		}
		for (int i = 0; i < words.size(); i++)
		{

			myfile.getline(str, 256);
			//cout << words[i] << "\n";
			sscanf(str, "%f %f", &arr1[i], &arr2[i]);
			cout <<i<<" : " << arr1[i] * N << " , " << arr2[i] << "is 1\n";
			h_A_dense[(int)((arr1[i]-1) * N + arr2[i])] = 1;
		}

	}
	//print_matrix(h_A_dense, N);
	h_B_dense = h_A_dense;


	int nnz_return = 0;
	for (int i = 0; i < N-1; i++)
	{
		int temp = nnz_return;
		h_B_dense = SMVP(h_B_dense, &nnz_return, N);
		cout<< nnz_return<< i << " round\n";
		if (temp == nnz_return)
		{
			cout << "finish after "<< i<< " round";
			exit(0); // terminate with error  
		}
	}
	cout << "finish after " << N-1 << " round";
	exit(0);


	// --- Create device arrays and copy host arrays to them
	double* d_A_dense;  gpuErrchk(cudaMalloc(&d_A_dense, N * N * sizeof(*d_A_dense)));
	double* d_B_dense;  gpuErrchk(cudaMalloc(&d_B_dense, N * N * sizeof(*d_B_dense)));
	double* d_C_dense;  gpuErrchk(cudaMalloc(&d_C_dense, N * N * sizeof(*d_C_dense)));
	gpuErrchk(cudaMemcpy(d_A_dense, h_A_dense, N * N * sizeof(*d_A_dense), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(d_B_dense, h_B_dense, N * N * sizeof(*d_B_dense), cudaMemcpyHostToDevice));

	// --- Descriptor for sparse matrix A
	cusparseMatDescr_t descrA;      cusparseSafeCall(cusparseCreateMatDescr(&descrA));
	cusparseSafeCall(cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL));
	cusparseSafeCall(cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ONE));

	// --- Descriptor for sparse matrix B
	cusparseMatDescr_t descrB;      cusparseSafeCall(cusparseCreateMatDescr(&descrB));
	cusparseSafeCall(cusparseSetMatType(descrB, CUSPARSE_MATRIX_TYPE_GENERAL));
	cusparseSafeCall(cusparseSetMatIndexBase(descrB, CUSPARSE_INDEX_BASE_ONE));

	// --- Descriptor for sparse matrix C
	cusparseMatDescr_t descrC;      cusparseSafeCall(cusparseCreateMatDescr(&descrC));
	cusparseSafeCall(cusparseSetMatType(descrC, CUSPARSE_MATRIX_TYPE_GENERAL));
	cusparseSafeCall(cusparseSetMatIndexBase(descrC, CUSPARSE_INDEX_BASE_ONE));

	int nnzA = 0;                           // --- Number of nonzero elements in dense matrix A
	int nnzB = 0;                           // --- Number of nonzero elements in dense matrix B

	const int lda = N;                      // --- Leading dimension of dense matrix

	// --- Device side number of nonzero elements per row of matrix A
	int* d_nnzPerVectorA;   gpuErrchk(cudaMalloc(&d_nnzPerVectorA, N * sizeof(*d_nnzPerVectorA)));
	cusparseSafeCall(cusparseDnnz(handle, CUSPARSE_DIRECTION_ROW, N, N, descrA, d_A_dense, lda, d_nnzPerVectorA, &nnzA));

	// --- Device side number of nonzero elements per row of matrix B
	int* d_nnzPerVectorB;   gpuErrchk(cudaMalloc(&d_nnzPerVectorB, N * sizeof(*d_nnzPerVectorB)));
	cusparseSafeCall(cusparseDnnz(handle, CUSPARSE_DIRECTION_ROW, N, N, descrB, d_B_dense, lda, d_nnzPerVectorB, &nnzB));

	// --- Host side number of nonzero elements per row of matrix A
	int* h_nnzPerVectorA = (int*)malloc(N * sizeof(*h_nnzPerVectorA));
	gpuErrchk(cudaMemcpy(h_nnzPerVectorA, d_nnzPerVectorA, N * sizeof(*h_nnzPerVectorA), cudaMemcpyDeviceToHost));

	// --- Host side number of nonzero elements per row of matrix B
	int* h_nnzPerVectorB = (int*)malloc(N * sizeof(*h_nnzPerVectorB));
	gpuErrchk(cudaMemcpy(h_nnzPerVectorB, d_nnzPerVectorB, N * sizeof(*h_nnzPerVectorB), cudaMemcpyDeviceToHost));

	/*printf("Number of nonzero elements in dense matrix A = %i\n\n", nnzA);
	for (int i = 0; i < N; ++i) printf("Number of nonzero elements in row %i for matrix = %i \n", i, h_nnzPerVectorA[i]);
	printf("\n");*/

	//printf("Number of nonzero elements in dense matrix B = %i\n\n", nnzB);
	//for (int i = 0; i < N; ++i) printf("Number of nonzero elements in row %i for matrix = %i \n", i, h_nnzPerVectorB[i]);
	//printf("\n");

	// --- Device side sparse matrix
	double* d_A;            gpuErrchk(cudaMalloc(&d_A, nnzA * sizeof(*d_A)));
	double* d_B;            gpuErrchk(cudaMalloc(&d_B, nnzB * sizeof(*d_B)));

	int* d_A_RowIndices;    gpuErrchk(cudaMalloc(&d_A_RowIndices, (N + 1) * sizeof(*d_A_RowIndices)));
	int* d_B_RowIndices;    gpuErrchk(cudaMalloc(&d_B_RowIndices, (N + 1) * sizeof(*d_B_RowIndices)));
	int* d_C_RowIndices;    gpuErrchk(cudaMalloc(&d_C_RowIndices, (N + 1) * sizeof(*d_C_RowIndices)));
	int* d_A_ColIndices;    gpuErrchk(cudaMalloc(&d_A_ColIndices, nnzA * sizeof(*d_A_ColIndices)));
	int* d_B_ColIndices;    gpuErrchk(cudaMalloc(&d_B_ColIndices, nnzB * sizeof(*d_B_ColIndices)));

	cusparseSafeCall(cusparseDdense2csr(handle, N, N, descrA, d_A_dense, lda, d_nnzPerVectorA, d_A, d_A_RowIndices, d_A_ColIndices));
	cusparseSafeCall(cusparseDdense2csr(handle, N, N, descrB, d_B_dense, lda, d_nnzPerVectorB, d_B, d_B_RowIndices, d_B_ColIndices));

	// --- Host side sparse matrices
	double* h_A = (double*)malloc(nnzA * sizeof(*h_A));
	double* h_B = (double*)malloc(nnzB * sizeof(*h_B));
	int* h_A_RowIndices = (int*)malloc((N + 1) * sizeof(*h_A_RowIndices));
	int* h_A_ColIndices = (int*)malloc(nnzA * sizeof(*h_A_ColIndices));
	int* h_B_RowIndices = (int*)malloc((N + 1) * sizeof(*h_B_RowIndices));
	int* h_B_ColIndices = (int*)malloc(nnzB * sizeof(*h_B_ColIndices));
	int* h_C_RowIndices = (int*)malloc((N + 1) * sizeof(*h_C_RowIndices));
	gpuErrchk(cudaMemcpy(h_A, d_A, nnzA * sizeof(*h_A), cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy(h_A_RowIndices, d_A_RowIndices, (N + 1) * sizeof(*h_A_RowIndices), cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy(h_A_ColIndices, d_A_ColIndices, nnzA * sizeof(*h_A_ColIndices), cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy(h_B, d_B, nnzB * sizeof(*h_B), cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy(h_B_RowIndices, d_B_RowIndices, (N + 1) * sizeof(*h_B_RowIndices), cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy(h_B_ColIndices, d_B_ColIndices, nnzB * sizeof(*h_B_ColIndices), cudaMemcpyDeviceToHost));


	// --- Performing the matrix - matrix multiplication
	int baseC, nnzC = 0;
	// nnzTotalDevHostPtr points to host memory
	int* nnzTotalDevHostPtr = &nnzC;

	cusparseSafeCall(cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_HOST));

	cusparseSafeCall(cusparseXcsrgemmNnz(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
		N, N, N,
		descrA, nnzA, d_A_RowIndices, d_A_ColIndices,
		descrA, nnzA, d_A_RowIndices, d_A_ColIndices, descrC,
		d_C_RowIndices, nnzTotalDevHostPtr));
	if (NULL != nnzTotalDevHostPtr) nnzC = *nnzTotalDevHostPtr;
	else {
		gpuErrchk(cudaMemcpy(&nnzC, d_C_RowIndices + N, sizeof(int), cudaMemcpyDeviceToHost));
		gpuErrchk(cudaMemcpy(&baseC, d_C_RowIndices, sizeof(int), cudaMemcpyDeviceToHost));
		nnzC -= baseC;
	}
	int* d_C_ColIndices;    gpuErrchk(cudaMalloc(&d_C_ColIndices, nnzC * sizeof(int)));
	double* d_C;            gpuErrchk(cudaMalloc(&d_C, nnzC * sizeof(double)));
	double* h_C = (double*)malloc(nnzC * sizeof(*h_C));
	int* h_C_ColIndices = (int*)malloc(nnzC * sizeof(*h_C_ColIndices));

	cusparseSafeCall(cusparseDcsrgemm(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, 
		N, N, N, 
		descrA, nnzA, d_A, d_A_RowIndices, d_A_ColIndices,
		descrA, nnzA, d_A, d_A_RowIndices, d_A_ColIndices, descrC,
		d_C, d_C_RowIndices, d_C_ColIndices));


	cusparseSafeCall(cusparseDcsr2dense(handle, N, N, descrC, d_C, d_C_RowIndices, d_C_ColIndices, d_C_dense, N));

	//gpuErrchk(cudaMemcpy(h_C, d_C, nnzC * sizeof(*h_C), cudaMemcpyDeviceToHost));
	//gpuErrchk(cudaMemcpy(h_C_RowIndices, d_C_RowIndices, (N + 1) * sizeof(*h_C_RowIndices), cudaMemcpyDeviceToHost));
	//gpuErrchk(cudaMemcpy(h_C_ColIndices, d_C_ColIndices, nnzC * sizeof(*h_C_ColIndices), cudaMemcpyDeviceToHost));


	gpuErrchk(cudaMemcpy(h_C_dense, d_C_dense, N * N * sizeof(double), cudaMemcpyDeviceToHost));
	print_matrix(h_C_dense, N);



	//gpuErrchk(cudaDeviceSynchronize());
	//gpuErrchk(cudaMemcpy(h_C_RowIndices, d_C_RowIndices, (N + 1) * sizeof(*h_C_RowIndices), cudaMemcpyDeviceToHost));
	//printf("before  h_C_RowIndices\n");
	//for (int i = 0; i < (N + 1); ++i) printf("h_C_RowIndices[%i] = %i \n", i, h_C_RowIndices[i]); printf("\n");
	///////////////////////// has bug
	///////////////////////////again
	///////////////////////// matrix C give value to matrix A (CSR format)

	//d_A = d_C; d_A_RowIndices = d_C_RowIndices; d_A_ColIndices = d_C_ColIndices;
	//nnzA = nnzC;

	//gpuErrchk(cudaDeviceSynchronize());
	//gpuErrchk(cudaMemcpy(h_C_RowIndices, d_A_RowIndices, (N + 1) * sizeof(*d_A_RowIndices), cudaMemcpyDeviceToHost));
	//printf("A copy C \n");
	//for (int i = 0; i < (N + 1); ++i) printf("h_C_RowIndices[%i] = %i \n", i, h_C_RowIndices[i]); printf("\n");
	////gpuErrchk(cudaMemcpyToSymbol(d_A, d_C, nnzC * sizeof(double)));
	////cudaMemcpyToSymbol(d_A_RowIndices, d_C_RowIndices, (N + 1) * sizeof(*d_C_RowIndices));
	////cudaMemcpyToSymbol(d_A_ColIndices, d_C_ColIndices, nnzC * sizeof(int));

	////// --- Performing the matrix - matrix multiplication
	////baseC, nnzC = 0;
	////// nnzTotalDevHostPtr points to host memory
	////nnzTotalDevHostPtr = &nnzC;

	//cusparseSafeCall(cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_HOST));

	//cudaFree(d_C_RowIndices);
	///*int* d_C_RowIndices;   */ gpuErrchk(cudaMalloc(&d_C_RowIndices, (N + 1) * sizeof(*d_C_RowIndices)));

	//cusparseSafeCall(cusparseXcsrgemmNnz(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
	//	N, N, N,
	//	descrA, nnzA, d_A_RowIndices, d_A_ColIndices,
	//	descrA, nnzA, d_A_RowIndices, d_A_ColIndices, descrC,
	//	d_C_RowIndices, nnzTotalDevHostPtr));

	//if (NULL != nnzTotalDevHostPtr) nnzC = *nnzTotalDevHostPtr;
	//else {
	//	gpuErrchk(cudaMemcpy(&nnzC, d_C_RowIndices + N, sizeof(int), cudaMemcpyDeviceToHost));
	//	gpuErrchk(cudaMemcpy(&baseC, d_C_RowIndices, sizeof(int), cudaMemcpyDeviceToHost));
	//	nnzC -= baseC;
	//}
	//cusparseSafeCall(cusparseDcsrgemm(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
	//	N, N, N,
	//	descrA, nnzA, d_A, d_A_RowIndices, d_A_ColIndices,
	//	descrA, nnzA, d_A, d_A_RowIndices, d_A_ColIndices, descrC,
	//	d_C, d_C_RowIndices, d_C_ColIndices));

	//gpuErrchk(cudaDeviceSynchronize());
	//gpuErrchk(cudaMemcpy(h_C_RowIndices, d_A_RowIndices, (N + 1) * sizeof(*d_A_RowIndices), cudaMemcpyDeviceToHost));
	//printf("result  h_C_RowIndices \n");
	//for (int i = 0; i < (N + 1); ++i) printf("h_C_RowIndices[%i] = %i \n", i, h_C_RowIndices[i]); printf("\n");

	////cusparseSafeCall(cusparseDcsr2dense(handle, N, N, descrC, d_C, d_C_RowIndices, d_C_ColIndices, d_C_dense, N));
	//cusparseSafeCall(cusparseDcsr2dense(handle, N, N, descrA, d_A, d_A_RowIndices, d_A_ColIndices, d_A_dense, N));

	//gpuErrchk(cudaMemcpy(h_C_dense, d_A_dense, N * N * sizeof(double), cudaMemcpyDeviceToHost));
	//print_matrix(h_C_dense, N);


	h_A_dense = h_C_dense;
	cout << "new A: \n";
	print_matrix(h_A_dense,N);

	// --- Create device arrays and copy host arrays to them
/*	double* d_A_dense; */ gpuErrchk(cudaMalloc(&d_A_dense, N * N * sizeof(*d_A_dense)));
/*	double* d_C_dense; */ gpuErrchk(cudaMalloc(&d_C_dense, N * N * sizeof(*d_C_dense)));
	gpuErrchk(cudaMemcpy(d_A_dense, h_A_dense, N * N * sizeof(*d_A_dense), cudaMemcpyHostToDevice));

	// --- Descriptor for sparse matrix A
    cusparseSafeCall(cusparseCreateMatDescr(&descrA));
	cusparseSafeCall(cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL));
	cusparseSafeCall(cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ONE));

	// --- Descriptor for sparse matrix C
	cusparseSafeCall(cusparseCreateMatDescr(&descrC));
	cusparseSafeCall(cusparseSetMatType(descrC, CUSPARSE_MATRIX_TYPE_GENERAL));
	cusparseSafeCall(cusparseSetMatIndexBase(descrC, CUSPARSE_INDEX_BASE_ONE));

	nnzA = 0;                           // --- Number of nonzero elements in dense matrix A
	 //lda = N;                      // --- Leading dimension of dense matrix

	// --- Device side number of nonzero elements per row of matrix A
	  gpuErrchk(cudaMalloc(&d_nnzPerVectorA, N * sizeof(*d_nnzPerVectorA)));
	cusparseSafeCall(cusparseDnnz(handle, CUSPARSE_DIRECTION_ROW, N, N, descrA, d_A_dense, lda, d_nnzPerVectorA, &nnzA));

	// --- Host side number of nonzero elements per row of matrix A
	 h_nnzPerVectorA = (int*)malloc(N * sizeof(*h_nnzPerVectorA));
	gpuErrchk(cudaMemcpy(h_nnzPerVectorA, d_nnzPerVectorA, N * sizeof(*h_nnzPerVectorA), cudaMemcpyDeviceToHost));

	/*printf("Number of nonzero elements in dense matrix A = %i\n\n", nnzA);
	for (int i = 0; i < N; ++i) printf("Number of nonzero elements in row %i for matrix = %i \n", i, h_nnzPerVectorA[i]);
	printf("\n");*/

	//printf("Number of nonzero elements in dense matrix B = %i\n\n", nnzB);
	//for (int i = 0; i < N; ++i) printf("Number of nonzero elements in row %i for matrix = %i \n", i, h_nnzPerVectorB[i]);
	//printf("\n");

	// --- Device side sparse matrix
	        gpuErrchk(cudaMalloc(&d_A, nnzA * sizeof(*d_A)));

	   gpuErrchk(cudaMalloc(&d_A_RowIndices, (N + 1) * sizeof(*d_A_RowIndices)));
	  gpuErrchk(cudaMalloc(&d_C_RowIndices, (N + 1) * sizeof(*d_C_RowIndices)));
	 gpuErrchk(cudaMalloc(&d_A_ColIndices, nnzA * sizeof(*d_A_ColIndices)));

	cusparseSafeCall(cusparseDdense2csr(handle, N, N, descrA, d_A_dense, lda, d_nnzPerVectorA, d_A, d_A_RowIndices, d_A_ColIndices));


	// --- Host side sparse matrices
	 h_A = (double*)malloc(nnzA * sizeof(*h_A));

	 h_A_RowIndices = (int*)malloc((N + 1) * sizeof(*h_A_RowIndices));
	h_A_ColIndices = (int*)malloc(nnzA * sizeof(*h_A_ColIndices));
	h_C_RowIndices = (int*)malloc((N + 1) * sizeof(*h_C_RowIndices));
	gpuErrchk(cudaMemcpy(h_A, d_A, nnzA * sizeof(*h_A), cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy(h_A_RowIndices, d_A_RowIndices, (N + 1) * sizeof(*h_A_RowIndices), cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy(h_A_ColIndices, d_A_ColIndices, nnzA * sizeof(*h_A_ColIndices), cudaMemcpyDeviceToHost));



	// --- Performing the matrix - matrix multiplication
	 baseC, nnzC = 0;
	// nnzTotalDevHostPtr points to host memory
	 nnzTotalDevHostPtr = &nnzC;

	cusparseSafeCall(cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_HOST));

	cusparseSafeCall(cusparseXcsrgemmNnz(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
		N, N, N,
		descrA, nnzA, d_A_RowIndices, d_A_ColIndices,
		descrA, nnzA, d_A_RowIndices, d_A_ColIndices, descrC,
		d_C_RowIndices, nnzTotalDevHostPtr));
	if (NULL != nnzTotalDevHostPtr) nnzC = *nnzTotalDevHostPtr;
	else {
		gpuErrchk(cudaMemcpy(&nnzC, d_C_RowIndices + N, sizeof(int), cudaMemcpyDeviceToHost));
		gpuErrchk(cudaMemcpy(&baseC, d_C_RowIndices, sizeof(int), cudaMemcpyDeviceToHost));
		nnzC -= baseC;
	}
	   gpuErrchk(cudaMalloc(&d_C_ColIndices, nnzC * sizeof(int)));
	          gpuErrchk(cudaMalloc(&d_C, nnzC * sizeof(double)));
	 h_C = (double*)malloc(nnzC * sizeof(*h_C));
	 h_C_ColIndices = (int*)malloc(nnzC * sizeof(*h_C_ColIndices));

	cusparseSafeCall(cusparseDcsrgemm(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
		N, N, N,
		descrA, nnzA, d_A, d_A_RowIndices, d_A_ColIndices,
		descrA, nnzA, d_A, d_A_RowIndices, d_A_ColIndices, descrC,
		d_C, d_C_RowIndices, d_C_ColIndices));


	cusparseSafeCall(cusparseDcsr2dense(handle, N, N, descrC, d_C, d_C_RowIndices, d_C_ColIndices, d_C_dense, N));

	//gpuErrchk(cudaMemcpy(h_C, d_C, nnzC * sizeof(*h_C), cudaMemcpyDeviceToHost));
	//gpuErrchk(cudaMemcpy(h_C_RowIndices, d_C_RowIndices, (N + 1) * sizeof(*h_C_RowIndices), cudaMemcpyDeviceToHost));
	//gpuErrchk(cudaMemcpy(h_C_ColIndices, d_C_ColIndices, nnzC * sizeof(*h_C_ColIndices), cudaMemcpyDeviceToHost));


	gpuErrchk(cudaMemcpy(h_C_dense, d_C_dense, N * N * sizeof(double), cudaMemcpyDeviceToHost));
	print_matrix(h_C_dense, N);
	cout << "new C: \n";
	cout << "num of nnzC " << nnzC;

	cout << "use function\n\n ";

	 nnz_return = 0;
	h_B_dense = SMVP(h_B_dense, &nnz_return, N);

	h_B_dense = SMVP(h_B_dense, &nnz_return, N);
    return 0;
}

double* SMVP(double* h_C_dense, int *nnz_return, int N) {

	double* h_A_dense = h_C_dense;
	cout << "new A: \n";
	print_matrix(h_A_dense, N);
	cusparseHandle_t handle;    cusparseCreate(&handle);

	// --- Create device arrays and copy host arrays to them
	double* d_A_dense;  gpuErrchk(cudaMalloc(&d_A_dense, N * N * sizeof(*d_A_dense)));

	double* d_C_dense;  gpuErrchk(cudaMalloc(&d_C_dense, N * N * sizeof(*d_C_dense)));
	gpuErrchk(cudaMemcpy(d_A_dense, h_A_dense, N * N * sizeof(*d_A_dense), cudaMemcpyHostToDevice));


	// --- Descriptor for sparse matrix A
	cusparseMatDescr_t descrA;      cusparseSafeCall(cusparseCreateMatDescr(&descrA));
	cusparseSafeCall(cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL));
	cusparseSafeCall(cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ONE));


	// --- Descriptor for sparse matrix C
	cusparseMatDescr_t descrC;      cusparseSafeCall(cusparseCreateMatDescr(&descrC));
	cusparseSafeCall(cusparseSetMatType(descrC, CUSPARSE_MATRIX_TYPE_GENERAL));
	cusparseSafeCall(cusparseSetMatIndexBase(descrC, CUSPARSE_INDEX_BASE_ONE));

	int nnzA = 0;                           // --- Number of nonzero elements in dense matrix A
                           // --- Number of nonzero elements in dense matrix B

	 int lda = N;                      // --- Leading dimension of dense matrix

	// --- Device side number of nonzero elements per row of matrix A
	int* d_nnzPerVectorA;   gpuErrchk(cudaMalloc(&d_nnzPerVectorA, N * sizeof(*d_nnzPerVectorA)));
	cusparseSafeCall(cusparseDnnz(handle, CUSPARSE_DIRECTION_ROW, N, N, descrA, d_A_dense, lda, d_nnzPerVectorA, &nnzA));

	// --- Host side number of nonzero elements per row of matrix A
	int* h_nnzPerVectorA = (int*)malloc(N * sizeof(*h_nnzPerVectorA));
	gpuErrchk(cudaMemcpy(h_nnzPerVectorA, d_nnzPerVectorA, N * sizeof(*h_nnzPerVectorA), cudaMemcpyDeviceToHost));
	

	/*printf("Number of nonzero elements in dense matrix A = %i\n\n", nnzA);
	for (int i = 0; i < N; ++i) printf("Number of nonzero elements in row %i for matrix = %i \n", i, h_nnzPerVectorA[i]);
	printf("\n");*/

	//printf("Number of nonzero elements in dense matrix B = %i\n\n", nnzB);
	//for (int i = 0; i < N; ++i) printf("Number of nonzero elements in row %i for matrix = %i \n", i, h_nnzPerVectorB[i]);
	//printf("\n");

	// --- Device side sparse matrix
	double* d_A;            gpuErrchk(cudaMalloc(&d_A, nnzA * sizeof(*d_A)));

	int* d_A_RowIndices;    gpuErrchk(cudaMalloc(&d_A_RowIndices, (N + 1) * sizeof(*d_A_RowIndices)));
	int* d_C_RowIndices;    gpuErrchk(cudaMalloc(&d_C_RowIndices, (N + 1) * sizeof(*d_C_RowIndices)));
	int* d_A_ColIndices;    gpuErrchk(cudaMalloc(&d_A_ColIndices, nnzA * sizeof(*d_A_ColIndices)));

	cusparseSafeCall(cusparseDdense2csr(handle, N, N, descrA, d_A_dense, lda, d_nnzPerVectorA, d_A, d_A_RowIndices, d_A_ColIndices));

	// --- Host side sparse matrices
	double* h_A = (double*)malloc(nnzA * sizeof(*h_A));
	int* h_A_RowIndices = (int*)malloc((N + 1) * sizeof(*h_A_RowIndices));
	int* h_A_ColIndices = (int*)malloc(nnzA * sizeof(*h_A_ColIndices));
	int* h_C_RowIndices = (int*)malloc((N + 1) * sizeof(*h_C_RowIndices));
	gpuErrchk(cudaMemcpy(h_A, d_A, nnzA * sizeof(*h_A), cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy(h_A_RowIndices, d_A_RowIndices, (N + 1) * sizeof(*h_A_RowIndices), cudaMemcpyDeviceToHost));
	gpuErrchk(cudaMemcpy(h_A_ColIndices, d_A_ColIndices, nnzA * sizeof(*h_A_ColIndices), cudaMemcpyDeviceToHost));


	// --- Performing the matrix - matrix multiplication
	int baseC, nnzC = 0;
	// nnzTotalDevHostPtr points to host memory
	int* nnzTotalDevHostPtr = &nnzC;

	cusparseSafeCall(cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_HOST));

	cusparseSafeCall(cusparseXcsrgemmNnz(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
		N, N, N,
		descrA, nnzA, d_A_RowIndices, d_A_ColIndices,
		descrA, nnzA, d_A_RowIndices, d_A_ColIndices, descrC,
		d_C_RowIndices, nnzTotalDevHostPtr));
	if (NULL != nnzTotalDevHostPtr) nnzC = *nnzTotalDevHostPtr;
	else {
		gpuErrchk(cudaMemcpy(&nnzC, d_C_RowIndices + N, sizeof(int), cudaMemcpyDeviceToHost));
		gpuErrchk(cudaMemcpy(&baseC, d_C_RowIndices, sizeof(int), cudaMemcpyDeviceToHost));
		nnzC -= baseC;
	}
	int* d_C_ColIndices;    gpuErrchk(cudaMalloc(&d_C_ColIndices, nnzC * sizeof(int)));
	double* d_C;            gpuErrchk(cudaMalloc(&d_C, nnzC * sizeof(double)));
	double* h_C = (double*)malloc(nnzC * sizeof(*h_C));
	int* h_C_ColIndices = (int*)malloc(nnzC * sizeof(*h_C_ColIndices));

	cusparseSafeCall(cusparseDcsrgemm(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
		N, N, N,
		descrA, nnzA, d_A, d_A_RowIndices, d_A_ColIndices,
		descrA, nnzA, d_A, d_A_RowIndices, d_A_ColIndices, descrC,
		d_C, d_C_RowIndices, d_C_ColIndices));


	cusparseSafeCall(cusparseDcsr2dense(handle, N, N, descrC, d_C, d_C_RowIndices, d_C_ColIndices, d_C_dense, N));
	cudaFree(descrC);
	cudaFree(d_C);
	cudaFree(d_C_RowIndices);
	cudaFree(d_C_ColIndices);
	cudaFree(descrA);
	cudaFree(d_A);
	cudaFree(d_A_RowIndices);
	cudaFree(d_A_ColIndices);
	
	//gpuErrchk(cudaMemcpy(h_C, d_C, nnzC * sizeof(*h_C), cudaMemcpyDeviceToHost));
	//gpuErrchk(cudaMemcpy(h_C_RowIndices, d_C_RowIndices, (N + 1) * sizeof(*h_C_RowIndices), cudaMemcpyDeviceToHost));
	//gpuErrchk(cudaMemcpy(h_C_ColIndices, d_C_ColIndices, nnzC * sizeof(*h_C_ColIndices), cudaMemcpyDeviceToHost));


	gpuErrchk(cudaMemcpy(h_C_dense, d_C_dense, N * N * sizeof(double), cudaMemcpyDeviceToHost));
	cudaFree(d_C_dense);
	//print_matrix(h_C_dense, N);
	cout << "nnzC " << nnzC;
	*nnz_return = nnzC;
	return h_C_dense;

}

