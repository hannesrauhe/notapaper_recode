/**
 * @file CUDA_MapRecode.cu
 * @date 05.04.2013
 * @author Hannes Rauhe
 *
 */

#include "CUDAHelper.hpp"
#include <assert.h>
#include <omp.h>
#include <iostream>
#include <boost/random.hpp>
#include <time.h>
#include <stdio.h>
#include <cstring>

static const int NUMBER_OF_THREADS = 512;
static const int NUMBER_OF_RUNS = 5;
static const int NUMBER_OF_WARMUP_RUNS = 3;
static const int BUF_SIZE = 128*1024;
//#define SHOW_RUN


void recode_cpu(int *dest, const int* src, const int *map, size_t size) {
    for(uint idx=0;idx<size;++idx) {
        dest[idx] = map[src[idx]];
    }
}
void recode_cpu_parallel(int *dest, const int *src, const int *map, unsigned size) {
#pragma omp parallel for
    for(uint idx=0;idx<size;++idx) {
        dest[idx] = map[src[idx]];
    }
}

__global__ void recode_naive(int *dest, const int *src, const int *map, unsigned size) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx<size) {
        dest[idx] = map[src[idx]];
    }
}

typedef boost::mt19937 RNGType;


float test_cpu_naive(const int* col, const int* map, size_t col_size, size_t map_size) {
    double result_time = 0.0f;

    for(int i = 0; i<(NUMBER_OF_RUNS+NUMBER_OF_WARMUP_RUNS); ++i) {
        int* r = new int[col_size];

        double start = omp_get_wtime();
        recode_cpu(r,col,map,col_size);
        double t = (omp_get_wtime()-start)*1000;
        delete[] r;
#ifdef SHOW_RUN
        std::cout<<i<<"\t"<<t<<std::endl;
#endif
        if(i>=NUMBER_OF_WARMUP_RUNS) {
            result_time=result_time+t;
        }
    }
    result_time=result_time/(NUMBER_OF_RUNS); /*ms*/
    return col_size/result_time;
}

float test_cpu_parallel(const int* col, const int* map, size_t col_size, size_t map_size) {
    double result_time = 0.0f;

    for(int i = 0; i<(NUMBER_OF_RUNS+NUMBER_OF_WARMUP_RUNS); ++i) {
        int* r = new int[col_size];

        double start = omp_get_wtime();
        recode_cpu_parallel(r,col,map,col_size);
        double t = (omp_get_wtime()-start)*1000;
        delete[] r;
#ifdef SHOW_RUN
        std::cout<<map_size<<"\t"<<t/col_size<<std::endl;
#endif
        if(i>=NUMBER_OF_WARMUP_RUNS) {
            result_time=result_time+t;
        }
    }
    result_time=result_time/(NUMBER_OF_RUNS);
    return col_size/result_time; /*ms*/
}

template<bool measure_transfer_only, bool measure_kernel_time_only>
float test_gpu_stream(const int* col, const int* map, size_t col_size, size_t map_size, const int* cmp = NULL) {
//    CUDA_SAFE_CALL( cudaSetDevice(1) );
    int* dev_src[3];
    int* dev_dest[3];
    int* dev_map;
    int* src[3];
    int* result;
    CUDA_SAFE_CALL( cudaMallocHost( (void**)&result, sizeof(int)*col_size) );

    int buf_size = BUF_SIZE;
    if(buf_size>col_size) {
        buf_size=col_size;
    }
    const int nblocks = buf_size / NUMBER_OF_THREADS;


    double result_time = 0.0f;
    float result_kernel_time = 0.0f;

    for(int i = 0; i<(NUMBER_OF_RUNS+NUMBER_OF_WARMUP_RUNS); ++i) {
        double t_start = omp_get_wtime();
        cudaStream_t streams[3];
        CUDA_SAFE_CALL( cudaMalloc(&dev_map,sizeof(int)*map_size) );
        CUDA_SAFE_CALL( cudaMemcpy(dev_map,map,map_size*sizeof(int), cudaMemcpyHostToDevice ) );
        CUDA_SAFE_CALL( cudaMallocHost( (void**)&src[0], sizeof(int)*buf_size) );
        CUDA_SAFE_CALL( cudaMallocHost( (void**)&src[1], sizeof(int)*buf_size) );
        CUDA_SAFE_CALL( cudaMallocHost( (void**)&src[2], sizeof(int)*buf_size) );
        CUDA_SAFE_CALL( cudaMalloc( (void**)&dev_src[0],sizeof(int)*buf_size) );
        CUDA_SAFE_CALL( cudaMalloc( (void**)&dev_src[1],sizeof(int)*buf_size) );
        CUDA_SAFE_CALL( cudaMalloc( (void**)&dev_src[2],sizeof(int)*buf_size) );
        CUDA_SAFE_CALL( cudaMalloc( (void**)&dev_dest[0],sizeof(int)*buf_size) );
        CUDA_SAFE_CALL( cudaMalloc( (void**)&dev_dest[1],sizeof(int)*buf_size) );
        CUDA_SAFE_CALL( cudaMalloc( (void**)&dev_dest[2],sizeof(int)*buf_size) );
        CUDA_SAFE_CALL( cudaStreamCreate(&streams[0]) );
        CUDA_SAFE_CALL( cudaStreamCreate(&streams[1]) );
        CUDA_SAFE_CALL( cudaStreamCreate(&streams[2]) );


        double tk_start = omp_get_wtime();
        int str_runs = col_size/buf_size+2;
        for( int buf_pos=0; buf_pos<str_runs; buf_pos++)
        {
            if(buf_pos<str_runs-1 && buf_pos>0)
                CUDA_SAFE_CALL( cudaMemcpyAsync(dev_src[(buf_pos)%3]           ,src[(buf_pos-1)%3],      buf_size*sizeof(int), cudaMemcpyHostToDevice, streams[0] ) );
            if(buf_pos>2)
                CUDA_SAFE_CALL( cudaMemcpyAsync(result+((buf_pos-3)*buf_size)  ,dev_dest[(buf_pos-2)%3],  buf_size*sizeof(int), cudaMemcpyDeviceToHost, streams[1] ) );

            if(!measure_transfer_only && buf_pos>1 && buf_pos<str_runs-1)
                recode_naive<<<nblocks,NUMBER_OF_THREADS,0,streams[2]>>>(dev_dest[(buf_pos-1)%3],dev_src[(buf_pos-1)%3],dev_map,buf_size);

            if(buf_pos<str_runs-2)
                memcpy(src[(buf_pos)%3],col+(buf_pos*buf_size),buf_size*sizeof(int));

            CUDA_SAFE_CALL( cudaThreadSynchronize());

        }
        double elapsedKernelTime = (omp_get_wtime()-tk_start)*1000;
        CUDA_SAFE_CALL( cudaFree(dev_map) );
        CUDA_SAFE_CALL( cudaFreeHost( src[0]) );
        CUDA_SAFE_CALL( cudaFreeHost( src[1]) );
        CUDA_SAFE_CALL( cudaFreeHost( src[2]) );
        CUDA_SAFE_CALL( cudaFree( dev_src[0]) );
        CUDA_SAFE_CALL( cudaFree( dev_src[1]) );
        CUDA_SAFE_CALL( cudaFree( dev_src[2]) );
        CUDA_SAFE_CALL( cudaFree( dev_dest[0]) );
        CUDA_SAFE_CALL( cudaFree( dev_dest[1]) );
        CUDA_SAFE_CALL( cudaFree( dev_dest[2]) );
        CUDA_SAFE_CALL( cudaStreamDestroy(streams[0]) );
        CUDA_SAFE_CALL( cudaStreamDestroy(streams[1]) );
        CUDA_SAFE_CALL( cudaStreamDestroy(streams[2]) );


        double elapsedTime = (omp_get_wtime()-t_start)*1000;
        if(!i && cmp!=NULL) {
            for(int j=0;j<col_size;++j) {
                assert(cmp[j]==result[j]);
            }
        }
        if(i>=NUMBER_OF_WARMUP_RUNS) {
            result_time=result_time+elapsedTime;
            result_kernel_time+=elapsedKernelTime;
        }
    }

//    CUDA_SAFE_CALL( cudaFreeHost(src) );
    CUDA_SAFE_CALL( cudaFreeHost( result) );
    CUDA_SAFE_CALL( cudaDeviceReset() );

    result_time=result_time/(NUMBER_OF_RUNS);
    result_kernel_time=result_kernel_time/(NUMBER_OF_RUNS);
    if(measure_kernel_time_only)
    	return col_size/result_kernel_time;
    else
    	return col_size/result_time;
}

template float test_gpu_stream<true,false>(const int* col, const int* map, size_t col_size, size_t map_size, const int* cmp);
template float test_gpu_stream<false,false>(const int* col, const int* map, size_t col_size, size_t map_size, const int* cmp);
template float test_gpu_stream<false,true>(const int* col, const int* map, size_t col_size, size_t map_size, const int* cmp);


int main( int argc, char** argv) {
    bool verbose = true;
    if(argc>1 && (strcmp(argv[1],"-h")==0 || strcmp(argv[1],"--help")==0)) {
    	printf("Usage: %s [min size of dict in KB] [max size of dict in KB] [size of column in MB]\n",argv[0]);
    	exit(0);
    }
//    int START_SIZE_OF_VECTOR = 1024*1024*32;
//    int MAX_SIZE_OF_VECTOR = 1024*1024*32; //means number of rows in a column

    //means size of dictionary of one column in KB
    int START_RANGE_OF_VECTOR = argc>1 ? atoi(argv[1]) : 8;
    int MAX_RANGE_OF_VECTOR = argc>2 ? atoi(argv[2]) : START_RANGE_OF_VECTOR*4;
    //size of index vector in MB
	int SIZE_OF_VECTOR = argc>3 ? atoi(argv[3]) : 1024;

	if(verbose) {
		printf("Size of Dictionary: %d - %d KB,Size of Vector in MB: %d\n",START_RANGE_OF_VECTOR,MAX_RANGE_OF_VECTOR,SIZE_OF_VECTOR);
	}
	SIZE_OF_VECTOR*=1024*1024/4; START_RANGE_OF_VECTOR*=1024/4; MAX_RANGE_OF_VECTOR*=1024/4;

//    for(int SIZE_OF_VECTOR = START_SIZE_OF_VECTOR;SIZE_OF_VECTOR<=MAX_SIZE_OF_VECTOR;SIZE_OF_VECTOR<<=1) {
        for(int RANGE_OF_VECTOR=START_RANGE_OF_VECTOR;RANGE_OF_VECTOR<=MAX_RANGE_OF_VECTOR && RANGE_OF_VECTOR<=SIZE_OF_VECTOR;RANGE_OF_VECTOR<<=1) {
            std::cout<<std::fixed<<RANGE_OF_VECTOR*4/1024<<";";
            std::cout.flush();
//            int* cmp_r = new int[SIZE_OF_VECTOR];
            RNGType generator(42u);
            boost::uniform_int<> uni_dist(0,RANGE_OF_VECTOR);
            boost::variate_generator<RNGType, boost::uniform_int<> > uni(generator, uni_dist);

            std::vector<int> v(SIZE_OF_VECTOR);
            for(int i=0;i<SIZE_OF_VECTOR;++i) {
                v[i]=(uni());
            }

            std::vector<int> m_v(RANGE_OF_VECTOR);
            for(int i=0;i<RANGE_OF_VECTOR;++i)
                m_v[i]=i;

            std::random_shuffle( m_v.begin(), m_v.end());

//            recode_cpu(cmp_r,&(v[0]),&(m_v[0]),v.size());
            double c_n = test_cpu_naive(&(v[0]),&(m_v[0]),v.size(),m_v.size());
            std::cout<<c_n*4/1024<<";"; /*MB/s*/
            std::cout.flush();
            double c_p = test_cpu_parallel(&(v[0]),&(m_v[0]),v.size(),m_v.size());
            std::cout<<c_p*4/1024<<";";
            std::cout.flush();
            double g_stream = test_gpu_stream<true,false>(&(v[0]),&(m_v[0]),v.size(),m_v.size()/*,cmp_r*/);
            std::cout<<g_stream*4/1024<<";";
            std::cout.flush();
            g_stream = test_gpu_stream<false,true>(&(v[0]),&(m_v[0]),v.size(),m_v.size()/*,cmp_r*/);
            std::cout<<g_stream*4/1024<<";";
            std::cout.flush();
            g_stream = test_gpu_stream<false,false>(&(v[0]),&(m_v[0]),v.size(),m_v.size()/*,cmp_r*/);
            std::cout<<g_stream*4/1024<<";";
            std::cout.flush();
            std::cout<<std::endl;
//            delete[] cmp_r;
        }
//    }
}
