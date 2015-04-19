#ifndef __UTILS_H
#define __UTILS_H

#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <helper_cuda.h>
#include <helper_math.h>
#include <vector_types.h>
#include <vector_functions.h>

#define __CHECK_DATA_

#define FLOAT_EPSILON 0.0001

#define MINIMUM(a, b) ((a) < (b) ? (a) : (b))
#define MAXIMUM(a, b) ((a) > (b) ? (a) : (b))
#define CLIP(a, b, c) MINIMUM(MAXIMUM((a), (b)), (c))					//a = your value, b = left bound, c = rightbound
#define INTERPOLATE(first,last,x) ((x-first)/(last-first))
#define ISININTERVAL(first,last,x) ((first<=x)&&(x<=last))
#define NOTININTERVAL(first,last,x) ((x<first)||(last<x))
#define CHECK_ZERO(x) ((x < FLOAT_EPSILON) || (-x > FLOAT_EPSILON))

#define SAFE_DELETE(p) if(p){delete (p);(p)=0;}
#define SAFE_DELETE_ARRAY(p) if(p){delete[] (p);(p)=0;}

#define SAFE_DELETE_CUDA(p) if(p){cudaFree(p);(p)=0;}
#define SAFE_DELETE_CUDAARRAY(p) if(p){cudaFreeArray(p);(p)=0;}
#define SAFE_DELETE_CUDAHOST(p) if(p){cudaFreeHost(p);(p)=0;}

#define GET_K_BIT(n, k) ((n >> k) & 1)

#ifdef __DEVICE_EMULATION__
	#define EMUSYNC __syncthreads()
#else
#define EMUSYNC
#endif

#define WARP_SIZE 32							//multiple 32   - do not change !!!
#define WARP_SIZE_MINUS_ONE 31
#define WARP_SIZE_SHIFT 5						//= log_2(WARP_SIZE)

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Defines an alias representing the kernel setting. </summary>
/// <remarks>	10. 2. 2013. </remarks>
////////////////////////////////////////////////////////////////////////////////////////////////////
typedef struct __align__(8) KernelSetting
{
public:
	dim3 dimBlock;
	dim3 dimGrid;
	unsigned int blockSize;
	unsigned int sharedMemSize;
	unsigned int noChunks;

	KernelSetting()
	{
		dimBlock = dim3(1,1,1);
		dimGrid = dim3(1,1,1);
		blockSize = 0;
		sharedMemSize = 0;
		noChunks = 1;
	}

	inline void print()
	{
		printf("\n------------------------------ KERNEL SETTING\n");
		printf("Block dimensions: %u %u %u\n", dimBlock.x, dimBlock.y, dimBlock.z);
		printf("Grid dimensions:  %u %u %u\n", dimGrid.x, dimGrid.y, dimGrid.z);
		printf("BlockSize: %u\n", blockSize);
		printf("Shared Memory Size: %u\n", sharedMemSize);
		printf("Number of chunks: %u\n", noChunks);
	}
}KernelSetting;


#pragma region TEMPLATE FUNCTIONS
////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Check host matrix. </summary>
/// <remarks>	Gajdi, 8. 11. 2013. </remarks>
/// <typeparam name="T">	Generic type parameter. </typeparam>
/// <param name="m">	  	The const T * to process. </param>
/// <param name="pitchInBytes">  	The pitch. </param>
/// <param name="rows">   	The rows. </param>
/// <param name="cols">   	The cols. </param>
/// <param name="format"> 	(optional) describes the format to use. </param>
/// <param name="message">	(optional) the message. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////
template< class T> __host__ void checkHostMatrix(const T *m, const unsigned int pitchInBytes, const unsigned int rows, const unsigned int cols, const char* format = "%f ", const char* message = "")
{
#ifdef __CHECK_DATA_
	printf("\nHOST MEMORY: %s [%u %u]\n", message, rows, cols);

	T *ptr = (T*)m;
	for (unsigned int i=0; i<rows; i++)
	{
		for (unsigned int j=0; j<cols; j++)
		{
			printf(format, ptr[j]);
		}
		printf("\n");
		ptr = (T*)(((char*)ptr)+pitchInBytes);
	}
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Check device matrix. </summary>
/// <remarks>	Gajdi, 8. 11. 2013. </remarks>
/// <typeparam name="T">	Generic type parameter. </typeparam>
/// <param name="m">		  	The const T * to process. </param>
/// <param name="pitchInBytes">	  	The pitch. </param>
/// <param name="rows">		  	The rows. </param>
/// <param name="cols">		  	The cols. </param>
/// <param name="format">	  	(optional) describes the format to use. </param>
/// <param name="message">	  	(optional) the message. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////
template< class T> __host__ void checkDeviceMatrix(const T *m, const unsigned int pitchInBytes, const unsigned int rows, const unsigned int cols, const char* format = "%f ", const char* message = "")
{
#ifdef __CHECK_DATA_
	printf("\nDEVICE MEMORY: %s [%u %u]\n", message, rows, cols);
	T *ptr;
	checkCudaErrors(cudaHostAlloc((void**)&ptr, pitchInBytes * rows, cudaHostAllocWriteCombined));
	checkCudaErrors(cudaMemcpy(ptr, m, rows * pitchInBytes, cudaMemcpyDeviceToHost));
	for (unsigned int i=0; i<rows; i++)
	{
		for (unsigned int j=0; j<cols; j++)
		{
			printf(format, ptr[j]);
		}
		printf("\n");
		ptr = (T*)(((char*)ptr)+pitchInBytes);
	}
	cudaFreeHost(ptr);
#endif
}


template< class T> __host__ void checkDeviceArray(const cudaArray *m, const unsigned int pitchInBytes, const unsigned int rows, const unsigned int cols, const char* format = "%f ", const char* message = "")
{
#ifdef __CHECK_DATA_
	printf("\nDEVICE MEMORY: %s [%u %u]\n", message, rows, cols);
	T *ptr;
	checkCudaErrors(cudaHostAlloc((void**)&ptr, pitchInBytes * rows, cudaHostAllocWriteCombined));
	checkCudaErrors(cudaMemcpyFromArray(ptr, m, 0, 0, rows * pitchInBytes, cudaMemcpyDeviceToHost));
	for (unsigned int i = 0; i<rows; i++)
	{
		for (unsigned int j = 0; j<cols; j++)
		{
			printf(format, ptr[j]);
		}
		printf("\n");
		ptr = (T*)(((char*)ptr) + pitchInBytes);
	}
	cudaFreeHost(ptr);
#endif
}


////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Verify device matrix. </summary>
/// <remarks>	Gajdi, 8. 11. 2013. </remarks>
/// <typeparam name="T">	Generic type parameter. </typeparam>
/// <typeparam name="T minValue">	Type of the minimum value. </typeparam>
/// <typeparam name="T maxValue">	Type of the maximum value. </typeparam>
/// <param name="m">		  	The const T * to process. </param>
/// <param name="pitchInBytes">	  	The pitch. </param>
/// <param name="rows">		  	The rows. </param>
/// <param name="cols">		  	The cols. </param>
/// <param name="format">	  	(optional) describes the format to use. </param>
/// <param name="message">	  	(optional) the message. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////
template< class T> __host__ void verifyDeviceMatrix(const T *m, const unsigned int pitchInBytes, const unsigned int rows, const unsigned int cols, const T minValue, const T maxValue, const char* message = "")
{
#ifdef __CHECK_DATA_
	printf("\nDEVICE MEMORY: %s [%u %u]\n", message, rows, cols);
	T *ptr;
	checkCudaErrors(cudaHostAlloc((void**)&ptr, pitchInBytes * rows, cudaHostAllocWriteCombined));
	checkCudaErrors(cudaMemcpy(ptr, m, rows * pitchInBytes, cudaMemcpyDeviceToHost));
	for (unsigned int i=0; i<rows; i++)
	{
		for (unsigned int j=0; j<cols; j++)
		{
			printf("%c", ISININTERVAL(minValue, maxValue, ptr[j]) ? ' ' : 'x');
		}
		printf("\n");
		ptr = (T*)(((char*)ptr)+pitchInBytes);
	}
	cudaFreeHost(ptr);
#endif
}


template< class T> struct check_data
{
	static __host__ void checkDeviceMatrix(const T *m, const unsigned int rows, const unsigned int cols, const bool isRowMatrix = true, const char* format = "%f ", const char* message = ""){}
};

template<> struct check_data<float4>
{
	static __host__ void checkDeviceMatrix(const float4 *m, const unsigned int rows, const unsigned int cols, const bool isRowMatrix = true, const char* format = "%f %f %f %f ", const char* message = "")
	{
#ifdef __CHECK_DATA_
		printf("\n------------------------- DEVICE MEMORY: %s [%u %u] %s\n", message, rows, cols, (isRowMatrix) ? "row matrix" : "column matrix " );
		float4 *tmp;
		checkCudaErrors(cudaMallocHost((void**)&tmp, rows * cols * sizeof(float4)));
		checkCudaErrors(cudaMemcpy(tmp, m, rows * cols * sizeof(float4), cudaMemcpyDeviceToHost));
		for (unsigned int i=0; i<rows * cols; i++)
		{
			if ((isRowMatrix)&&((i%cols)==0))
				printf("\nRow: ");
			if ((!isRowMatrix)&&((i%rows)==0))
				printf("\nCol: ");
			printf(format, tmp[i].x, tmp[i].y, tmp[i].z, tmp[i].w);
		}
		printf("\n");
		cudaFreeHost(tmp);
#endif
	}
};

template<> struct check_data<uchar4>
{
	static __host__ void checkDeviceMatrix(const uchar4 *m, const unsigned int rows, const unsigned int cols, const bool isRowMatrix = true, const char* format = "%hhu %hhu %hhu %hhu ", const char* message = "")
	{
#ifdef __CHECK_DATA_
		printf("\n------------------------- DEVICE MEMORY: %s [%u %u] %s\n", message, rows, cols, (isRowMatrix) ? "row matrix" : "column matrix " );
		uchar4 *tmp;
		checkCudaErrors(cudaMallocHost((void**)&tmp, rows * cols * sizeof(uchar4)));
		checkCudaErrors(cudaMemcpy(tmp, m, rows * cols * sizeof(uchar4), cudaMemcpyDeviceToHost));
		for (unsigned int i=0; i<rows * cols; i++)
		{
			if ((isRowMatrix)&&((i%cols)==0))
				printf("\nRow: ");
			if ((!isRowMatrix)&&((i%rows)==0))
				printf("\nCol: ");
			printf(format, tmp[i].x, tmp[i].y, tmp[i].z, tmp[i].w);
		}
		printf("\n");
		cudaFreeHost(tmp);
#endif
	}
};




////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Check device matrix. </summary>
///
/// <remarks>	Copies the matrix from device to host and prints its elements. </remarks>
///
/// <param name="m">		[in,out] If non-null, the matrix. </param>
/// <param name="mSize">	The size of matrix. </param>
/// <param name="message">	[in,out] If non-null, the message. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////
template< class T> __host__ void checkDeviceMatrixCUBLAS(const T *m, const unsigned int rows, const unsigned int cols, const bool isRowMatrix = true, const char* format = "%f ", const char* message = "")
{
#ifdef __CHECK_DATA_
	printf("\n------------------------- DEVICE MEMORY: %s [%u %u] %s\n", message, rows, cols, (isRowMatrix) ? "row matrix" : "column matrix " );
	unsigned int tmpSize = rows * cols * sizeof(T);
	T *tmp = (T*)malloc(tmpSize);
	cublasGetVector (rows *cols, sizeof(T), m, 1, tmp, 1);
	for (unsigned int i=0; i<rows * cols; i++)
	{
		if ((isRowMatrix)&&((i%cols)==0))
			printf("\nRow %u: ", i/cols);
		if ((!isRowMatrix)&&((i%rows)==0))
			printf("\nCol %u: ", i/rows);
		printf(format, tmp[i]);
	}
	printf("\n");
	free(tmp);
#endif
}

#pragma endregion

#pragma region INLINE FUNCTIONS

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Check device properties. </summary>
/// <remarks>	Gajdi, 19.11.2010. </remarks>
/// <param name="deviceProp">	[in,out] the device property. </param>
/// <returns>	true if it succeeds, false if it fails. </returns>
////////////////////////////////////////////////////////////////////////////////////////////////////
__forceinline__  __host__ bool checkDeviceProperties()
{	
	cudaDeviceProp deviceProp;
	bool result = true;
	printf("CUDA Device Query (Runtime API) version (CUDART static linking)\n");

	int deviceCount;
	cudaGetDeviceCount(&deviceCount);

	// This function call returns 0 if there are no CUDA capable devices.
	if (deviceCount == 0)
	{	
		printf("There is no device supporting CUDA\n");
		result =  false;
	}

	int dev;
	for (dev = 0; dev < deviceCount; ++dev) 
	{
		cudaGetDeviceProperties(&deviceProp, dev);

		if (dev == 0) 
		{
			// This function call returns 9999 for both major & minor fields, if no CUDA capable devices are present
			if (deviceProp.major == 9999 && deviceProp.minor == 9999)
			{
				printf("There is no device supporting CUDA.\n");
				result = false;
			}
			else if (deviceCount == 1)
				printf("There is 1 device supporting CUDA\n");
			else
				printf("There are %d devices supporting CUDA\n", deviceCount);
		}
		printf("\nDevice %d: \"%s\"\n", dev, deviceProp.name);
	#if CUDART_VERSION >= 2020
		int driverVersion = 0, runtimeVersion = 0;
		cudaDriverGetVersion(&driverVersion);
		printf("  CUDA Driver Version:                           %d.%d\n", driverVersion/1000, driverVersion%100);
		cudaRuntimeGetVersion(&runtimeVersion);
		printf("  CUDA Runtime Version:                          %d.%d\n", runtimeVersion/1000, runtimeVersion%100);
	#endif
		printf("  CUDA Capability Major revision number:         %d\n", deviceProp.major);
		printf("  CUDA Capability Minor revision number:         %d\n", deviceProp.minor);
	#if CUDART_VERSION >= 2000
		printf("  Number of multiprocessors:                     %d\n", deviceProp.multiProcessorCount);
		printf("  Number of cores:                               %d\n", 8 * deviceProp.multiProcessorCount);
	#endif
		printf("  Total amount of global memory:                 %lu bytes\n", deviceProp.totalGlobalMem);
		printf("  Total amount of constant memory:               %lu bytes\n", deviceProp.totalConstMem);
		printf("  Total amount of shared memory per SM:			 %lu bytes\n", deviceProp.sharedMemPerMultiprocessor);
		printf("  Total amount of shared memory per block:       %lu bytes\n", deviceProp.sharedMemPerBlock);
		printf("  Total number of registers available per SM:	 %d\n", deviceProp.regsPerMultiprocessor);
		printf("  Total number of registers available per block: %d\n", deviceProp.regsPerBlock);
		printf("  Warp size:                                     %d\n", deviceProp.warpSize);
		printf("  Maximum number of threads per block:           %d\n", deviceProp.maxThreadsPerBlock);
		printf("  Maximum sizes of each dimension of a block:    %d x %d x %d\n",
			   deviceProp.maxThreadsDim[0],
			   deviceProp.maxThreadsDim[1],
			   deviceProp.maxThreadsDim[2]);
		printf("  Maximum sizes of each dimension of a grid:     %d x %d x %d\n",
			   deviceProp.maxGridSize[0],
			   deviceProp.maxGridSize[1],
			   deviceProp.maxGridSize[2]);
		printf("  Maximum memory pitch:                          %lu bytes\n", deviceProp.memPitch);
		printf("  Texture alignment:                             %lu bytes\n", deviceProp.textureAlignment);
		printf("  Clock rate:                                    %.2f GHz\n", deviceProp.clockRate * 1e-6f);
	#if CUDART_VERSION >= 2000
		printf("  Concurrent copy and execution:                 %s\n", deviceProp.deviceOverlap ? "Yes" : "No");
	#endif
	#if CUDART_VERSION >= 2020
		printf("  Run time limit on kernels:                     %s\n", deviceProp.kernelExecTimeoutEnabled ? "Yes" : "No");
		printf("  Integrated:                                    %s\n", deviceProp.integrated ? "Yes" : "No");
		printf("  Support host page-locked memory mapping:       %s\n", deviceProp.canMapHostMemory ? "Yes" : "No");
		printf("  Compute mode:                                  %s\n", deviceProp.computeMode == cudaComputeModeDefault ?
																		"Default (multiple host threads can use this device simultaneously)" :
																		deviceProp.computeMode == cudaComputeModeExclusive ?
																		"Exclusive (only one host thread at a time can use this device)" :
																		deviceProp.computeMode == cudaComputeModeProhibited ?
																		"Prohibited (no host thread can use this device)" :
																		"Unknown");
	#endif
	}
	printf("\nDevice Test PASSED -----------------------------------------------------\n\n");
	return result;
}

__forceinline__  __host__ void checkError()
{
	cudaError_t err= cudaGetLastError();
	if (cudaGetLastError() != cudaSuccess)
	{
		printf("%s\n", cudaGetErrorString(err));
	}
}


__forceinline__  __host__ unsigned int getNumberOfParts(const unsigned int totalSize, const unsigned int partSize)
{
	return (totalSize+partSize-1)/partSize;
}

__forceinline__  __host__ void prepareKernelSettings(const cudaDeviceProp &deviceProp, const unsigned int initNoChunks, const unsigned int threadsPerBlock, const unsigned int dataLength, KernelSetting &ks)
{
	ks.blockSize = threadsPerBlock;
	ks.dimBlock = dim3(threadsPerBlock,1,1);
	ks.sharedMemSize = (unsigned int)deviceProp.sharedMemPerBlock;

	ks.noChunks = initNoChunks;		//Initial Number of Chunks

	unsigned int noBlocks = getNumberOfParts(dataLength, threadsPerBlock * ks.noChunks);
	if (noBlocks > (unsigned int)deviceProp.maxGridSize[0])
	{
		unsigned int multiplicator = noBlocks / deviceProp.maxGridSize[0];
		if ((noBlocks % deviceProp.maxGridSize[0]) != 0)
			multiplicator++;
		ks.noChunks *= multiplicator;
		ks.dimGrid = getNumberOfParts(dataLength, threadsPerBlock * ks.noChunks);
	}
	else
	{
		ks.dimGrid = dim3(noBlocks, 1,1);
	}
	//ks.print();
}

__forceinline__  __host__ void  initializeCUDA(cudaDeviceProp &deviceProp, int requiredDid = -1)
{
	if (!checkDeviceProperties()) return exit(-1);
	if (requiredDid!=-1)
	{
		int did = requiredDid;
		checkCudaErrors(cudaSetDevice(did));
		cudaGetDeviceProperties(&deviceProp, did);
		printf("SELECTED GPU Device %d: \"%s\" with compute capability %d.%d\n\n", did, deviceProp.name, deviceProp.major, deviceProp.minor);
		if (cudaGetLastError() != cudaSuccess)
			exit(-1);
	}
	else
	{
		int did = 0;
		//cudaGetDevice(&did);
		did = gpuGetMaxGflopsDeviceId();
		checkCudaErrors(cudaSetDevice(did));
		cudaGetDeviceProperties(&deviceProp, did);
		printf("SELECTED GPU Device %d: \"%s\" with compute capability %d.%d\n\n", did, deviceProp.name, deviceProp.major, deviceProp.minor);
		if (cudaGetLastError() != cudaSuccess)
			exit(-1);
	}
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Next pow 2. </summary>
/// <remarks>	Gajdi, 19.11.2010. </remarks>
/// <param name="x">	The number x. </param>
/// <returns>	Returns the number which is close to given number x and its is pow 2. </returns>
////////////////////////////////////////////////////////////////////////////////////////////////////
__forceinline__ __host__ __device__ unsigned int nextPow2(unsigned int x)
{
	--x;
	x |= x >> 1;
	x |= x >> 2;
	x |= x >> 4;
	x |= x >> 8;
	x |= x >> 16;
	return ++x;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Tests if 'x' is pow 2. </summary>
/// <remarks>	Gajdi, 19.11.2010. </remarks>
/// <param name="x">The number x. </param>
/// <returns>	TUE if it is pow 2, otherwise FALSE. </returns>
////////////////////////////////////////////////////////////////////////////////////////////////////
__forceinline__  __host__ __device__ bool isPow2( const unsigned int x)
{
	return ((x&(x-1))==0);
}

/// <summary>	Next multiple of warp. It is supposed that isPow2(multiple)==true </summary>
/// <remarks>	Gajdi, 10. 10. 2014. </remarks>
/// <param name="x">	   	The const unsigned int to process. </param>
/// <param name="multiple">	(Optional) the multiple. </param>
/// <returns>	An int. </returns>
__forceinline__  __host__ __device__ unsigned int nextMultipleOfWarp(const unsigned int x, const unsigned int multiple = WARP_SIZE)
{
	return (x + multiple - 1) & ~(multiple - 1);
	//return ((x + multiple - 1) / multiple) * multiple;
}

/// <summary>	Next multiple of a number.</summary>
/// <remarks>	Gajdi, 10. 10. 2014. </remarks>
/// <param name="x">	   	The const unsigned int to process. </param>
/// <param name="multiple">	(Optional) the multiple. </param>
/// <returns>	An int. </returns>
__forceinline__  __host__ __device__ unsigned int nextMultipleOf(const unsigned int x, const unsigned int multiple)
{
	if (multiple == 0)
	{
		return 0;
	}
	return ((x - 1) / multiple + 1) * multiple;
}

__forceinline__  __host__ void createTimer(cudaEvent_t *startEvent, cudaEvent_t *stopEvent, float *elapsedTime)
{
	cudaEventCreate(startEvent);
	cudaEventCreate(stopEvent);
	*elapsedTime = 0.0f;
}

__forceinline__  __host__ void startTimer(cudaEvent_t &startEvent)
{
	cudaEventRecord(startEvent, 0);
}

__forceinline__  __host__ void stopTimer(cudaEvent_t &startEvent, cudaEvent_t &stopEvent, float &elapsedTime, const bool appendTime = false)
{
	cudaEventRecord(stopEvent, 0);
	cudaEventSynchronize(stopEvent);
	if (!appendTime)
	{
		cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent);
		return;
	}

	float t = 0.0f;
	cudaEventElapsedTime(&t, startEvent, stopEvent);
	elapsedTime += t;
}

__forceinline__  __host__ void destroyTimer(cudaEvent_t &startEvent, cudaEvent_t &stopEvent)
{
	cudaEventDestroy(startEvent);
	cudaEventDestroy(stopEvent);
}


#pragma endregion

#endif