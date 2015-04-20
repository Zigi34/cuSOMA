#include <cudaDefs.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <curand.h>
#include<curand_kernel.h>

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <math.h>

//FUNCTIONS
#define imin(a,b) (a<b?a:b)
#define imax(a,b) (a<b?b:a)

//THREADS, BLOCKS
#define THREADS 64
#define BLOCKS imin(1024, (NUM_VALS+THREADS-1)/THREADS)

//POPULATION
#define NUM_VALS 8192
#define DIMENSION 3

//SOLUTION
#define RANDOM_MAXIMUM 500

//CONFIG
#define CONF_PARMS 2
#define CONF_PATHLENGTH 0
#define CONF_STEP 1

#pragma region SOLUTION
typedef struct __align__(16) individual{
	float val[DIMENSION];
	float fitness;
} individual;
#pragma endregion

typedef struct __align__(16) best_indiv {
	float val[DIMENSION];
	unsigned int index;
} best_indiv;


//LEADER
__constant__ best_indiv leader[1];
__constant__ float config[CONF_PARMS];

//DEBUG ERRORS
#pragma region ERROR FUNCTION
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}
#pragma endregion

#pragma region REDUCTION
__device__ float atomicMaxf(float* address, float val)
{
	int *address_as_int = (int*)address;
	int old = *address_as_int, assumed;
	while (val > __int_as_float(old)) {
		assumed = old;
		old = atomicCAS(address_as_int, assumed,
			__float_as_int(val));
	}
	return __int_as_float(old);
}
__global__ void max_reduce(const individual* const d_array, float* d_max, best_indiv* leader,
	const size_t elements)
{
	__shared__ float shared[THREADS];

	int tid = threadIdx.x;
	int gid = (blockDim.x * blockIdx.x) + tid;
	shared[tid] = 0.0f;

	if (gid < elements)
		shared[tid] = d_array[gid].fitness;
	__syncthreads();

	for (unsigned int s = blockDim.x / 2; s>0; s >>= 1)
	{
		if (tid < s && gid < elements)
			shared[tid] = max(shared[tid], shared[tid + s]);  // 2
		__syncthreads();
	}
	// what to do now?
	// option 1: save block result and launch another kernel
	if (tid == 0)
		d_max[blockIdx.x] = shared[tid]; // 3
	// option 2: use atomics
	if (tid == 0)
		atomicMaxf(d_max, shared[0]);
	__syncthreads();
	
	if (gid < elements && d_max[0] == d_array[gid].fitness) {
		for (int i = 0; i < DIMENSION; i++)
			leader[0].val[i] = d_array[gid].val[i];
		leader[0].index = gid;
	}
}
#pragma endregion

#pragma region RANDOM GENERATOR
__global__ void init_rand(curandState *state, size_t size) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < size)
		curand_init(1337, idx, 0, &state[idx]);
}
__global__ void make_rand(curandState *state, float
	*randArray, const size_t size) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < size)
		randArray[idx] = curand_uniform(&state[idx]);
}
#pragma endregion

#pragma region RANDOM POPULATION
__global__ void random_population(curandState *state, individual* pop, const size_t size) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < size) {
		for (int i = 0; i < DIMENSION; i++)
			pop[idx].val[i] = curand_uniform(&state[idx]) * RANDOM_MAXIMUM;
	}
}
#pragma endregion

#pragma region UTIL FUNCTION
void print_elapsed(clock_t start, clock_t stop)
{
	double elapsed = ((double)(stop - start)) / CLOCKS_PER_SEC;
	printf("Elapsed time: %.3fs\n", elapsed);
}
float random_float()
{
	return (float)(rand() % RANDOM_MAXIMUM) + 1.0;
}
void individual_print(individual *population, const size_t size)
{
	int i;
	for (i = 0; i < NUM_VALS; ++i) {
		printf("(%1.1f, %1.1f, %1.1f) -> %1.1f\n ", population[i].val[0], population[i].val[1], population[i].val[2], population[i].fitness);
	}
	printf("\n");
}
void population_fill(individual *arr, const size_t size)
{
	srand(time(NULL));
	for (int i = 0; i < size; ++i) {
		arr[i].val[0] = random_float();
		arr[i].val[1] = random_float();
		arr[i].val[2] = random_float();
		arr[i].fitness = 0.0;
	}
}
float max_value(individual* values, const size_t size) {
	float vysledek = 0.0;
	for (int i = 0; i<size; i++) {
		if (values[i].fitness > vysledek)
			vysledek = values[i].fitness;
	}
	return vysledek;
}
#pragma endregion

#pragma region SOMA EVOLUTION
__global__ void step(const individual* pop, individual* new_pop, float* prt, size_t size) {
	int lid = threadIdx.x;
	int gid = blockIdx.x * THREADS + threadIdx.x;
	
	//for (float j = config[CONF_STEP]; j < config[CONF_PATHLENGTH]; j += config[CONF_STEP]) {
		#pragma unroll
		for (int i = 0; i < DIMENSION; i++) {
			if (prt[i] > 0.5) {
				new_pop[gid].val[i] = (leader[0].val[i] - pop[gid].val[i]) * config[CONF_STEP] + pop[gid].val[i];
			} else {
				new_pop[gid].val[i] = pop[gid].val[i];
			}
		}
	//}
	__syncthreads();
}
__global__ void evaluate(individual* individuals, const size_t size)
{
	int tid = blockIdx.x * THREADS + threadIdx.x;
	while (tid < size)
	{
		individuals[tid].fitness = (individuals[tid].val[0] * individuals[tid].val[1] * individuals[tid].val[2]);
		tid += THREADS * BLOCKS;
	}
	__syncthreads();
}
void soma(individual *values, const size_t size)
{
	dim3 blocks(BLOCKS, 1);
	dim3 threads(THREADS, 1);
	
	//DEVICE
	float* dev_fitness;
	best_indiv* dev_leader;
	float* dev_prt_vector;
	curandState* dev_states;
	individual* dev_indiv;
	individual* dev_new_pop;

	float host_config[CONF_PARMS];
	host_config[CONF_PATHLENGTH] = 10.0;
	host_config[CONF_STEP] = 1.1;

	//int size = s * (int)(host_config[CONF_PATHLENGTH] / host_config[CONF_STEP]);

	gpuErrchk(cudaMalloc((void**)&dev_fitness, size * sizeof(float)));
	gpuErrchk(cudaMalloc((void**)&dev_leader, sizeof(best_indiv)));
	gpuErrchk(cudaMalloc((void**)&dev_new_pop, size * sizeof(individual)));

	best_indiv* host_leader = (best_indiv*)malloc(sizeof(best_indiv));
	individual* population = (individual*)malloc(size * sizeof(individual));

	#pragma region SOMA PARAMETER INITIALIZE

	cudaMemcpyToSymbol(config, host_config, sizeof(float)* CONF_PARMS);
	#pragma endregion

	#pragma region INITIALIZE
	gpuErrchk(cudaMalloc((void**)&dev_prt_vector, DIMENSION * sizeof(float)));
	gpuErrchk(cudaMalloc((void**)&dev_states, DIMENSION * sizeof(curandState)));

	init_rand << <1, threads >> >(dev_states, DIMENSION);
	make_rand << <1, threads >> >(dev_states, dev_prt_vector, DIMENSION);
	#pragma endregion
	
	#pragma region POPULATION TO GPU
	cudaMalloc((void**)&dev_indiv, size * sizeof(individual));
	cudaMemcpy(dev_indiv, values, size * sizeof(individual), cudaMemcpyHostToDevice);
	#pragma endregion

	for (int iter = 0; iter < 5000; iter++) {
		evaluate << <blocks, threads >> >(dev_indiv, size);
		max_reduce << <blocks, threads >> >(dev_indiv, dev_fitness, dev_leader, size);
		gpuErrchk(cudaMemcpy(host_leader, dev_leader, sizeof(best_indiv), cudaMemcpyDeviceToHost));

		//printf("Leader: (%1.3f, %1.3f, %1.3f)(%d)\n", host_leader[0].val[0], host_leader[0].val[1], host_leader[0].val[2], host_leader[0].index);
		cudaMemcpyToSymbol(leader, host_leader, sizeof(best_indiv));

		make_rand << <1, threads >> >(dev_states, dev_prt_vector, DIMENSION);
		step << <blocks, threads >> >(dev_indiv, dev_new_pop, dev_prt_vector, size);

		gpuErrchk(cudaMemcpy(population, dev_indiv, size * sizeof(individual), cudaMemcpyDeviceToHost));
		//for (int i = 0; i < 10; i++)
			//printf("%1.3f, %1.3f, %1.3f = %1.3f\n", population[i].val[0], population[i].val[1], population[i].val[2], population[i].fitness);
		//printf("\n");

		individual * temp = dev_new_pop;
		dev_new_pop = dev_indiv;
		dev_indiv = temp;
	}
	
	free(host_leader);
	free(population);

	cudaFree(dev_fitness);
	cudaFree(dev_leader);
	cudaFree(dev_prt_vector);
	cudaFree(dev_states);
	cudaFree(dev_indiv);
	cudaFree(dev_new_pop);
}

#pragma endregion


int main(int argc, char *argv[])
{
	//initializeCUDA(deviceProp);
	clock_t start, stop;
	size_t size = NUM_VALS;
	printf("Velikost populace: %d\n",size);

	individual *values = (individual*)malloc(size * sizeof(individual));
	population_fill(values, size);

	start = clock();
	soma(values, size); /* Inplace */
	stop = clock();

	//individual_print(values);
	//printf("MAX= %1.1f\n", max_value(values, NUM_VALS));
	free(values);
	print_elapsed(start, stop);
}

