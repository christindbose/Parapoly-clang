// clang-format off
/************************************************************************************\ 
 *                                                                                  *
 * Copyright � 2014 Advanced Micro Devices, Inc.                                    *
 * Copyright (c) 2015 Mark D. Hill and David A. Wood                                *
 * All rights reserved.                                                             *
 *                                                                                  *
 * Redistribution and use in source and binary forms, with or without               *
 * modification, are permitted provided that the following are met:                 *
 *                                                                                  *
 * You must reproduce the above copyright notice.                                   *
 *                                                                                  *
 * Neither the name of the copyright holder nor the names of its contributors       *
 * may be used to endorse or promote products derived from this software            *
 * without specific, prior, written permission from at least the copyright holder.  *
 *                                                                                  *
 * You must include the following terms in your license and/or other materials      *
 * provided with the software.                                                      *
 *                                                                                  *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"      *
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE        *
 * IMPLIED WARRANTIES OF MERCHANTABILITY, NON-INFRINGEMENT, AND FITNESS FOR A       *
 * PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER        *
 * OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,         *
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT  *
 * OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS      *
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN          *
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING  *
 * IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY   *
 * OF SUCH DAMAGE.                                                                  *
 *                                                                                  *
 * Without limiting the foregoing, the software may implement third party           *
 * technologies for which you must obtain licenses from parties other than AMD.     *
 * You agree that AMD has not obtained or conveyed to you, and that you shall       *
 * be responsible for obtaining the rights to use and/or distribute the applicable  *
 * underlying intellectual property rights related to the third party technologies. *
 * These third party technologies are not licensed hereunder.                       *
 *                                                                                  *
 * If you use the software (in whole or in part), you shall adhere to all           *
 * applicable U.S., European, and other export laws, including but not limited to   *
 * the U.S. Export Administration Regulations ("EAR"�) (15 C.F.R Sections 730-774),  *
 * and E.U. Council Regulation (EC) No 428/2009 of 5 May 2009.  Further, pursuant   *
 * to Section 740.6 of the EAR, you hereby certify that, except pursuant to a       *
 * license granted by the United States Department of Commerce Bureau of Industry   *
 * and Security or as otherwise permitted pursuant to a License Exception under     *
 * the U.S. Export Administration Regulations ("EAR"), you will not (1) export,     *
 * re-export or release to a national of a country in Country Groups D:1, E:1 or    *
 * E:2 any restricted technology, software, or source code you receive hereunder,   *
 * or (2) export to Country Groups D:1, E:1 or E:2 the direct product of such       *
 * technology or software, if such foreign produced direct product is subject to    *
 * national security controls as identified on the Commerce Control List (currently *
 * found in Supplement 1 to Part 774 of EAR).  For the most current Country Group   *
 * listings, or for additional information about the EAR or your obligations under  *
 * those regulations, please refer to the U.S. Bureau of Industry and Security's    *
 * website at http://www.bis.doc.gov/.                                              *
 *                                                                                  *
\************************************************************************************/
// clang-format on

#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
//#include "../../graph_parser/parse.h"
#include <cuda_runtime_api.h>
#include "../../graph_parser/util.h"

#include "../../graph_parser/parse.cpp"
#include "../../graph_parser/util.cpp"

#include "kernel.cu"

// Iteration count
#define ITER 20

void print_vectorf(int *vector, int num);

int main(int argc, char **argv) {
    char *tmpchar;

    int num_nodes;
    int num_edges;
    int file_format = 1;
    bool directed = 0;

    cudaError_t err = cudaSuccess;

    if (argc == 3) {
        tmpchar = argv[1];            // Graph inputfile
        file_format = atoi(argv[2]);  // File format
    } else {
        fprintf(stderr, "You did something wrong!\n");
        exit(1);
    }

    // Allocate the csr structure
    csr_array *csr;

    // Parse graph files into csr structure
    if (file_format == 1) {
        // Metis
        csr = parseMetis(tmpchar, &num_nodes, &num_edges, directed);
    } else if (file_format == 0) {
        // Dimacs9
        csr = parseCOO(tmpchar, &num_nodes, &num_edges, 1);
    } else if (file_format == 2) {
        // Matrix market
        csr = parseMM(tmpchar, &num_nodes, &num_edges, directed, 0);
    } else {
        printf("reserve for future");
        exit(1);
    }

    // Allocate rank_array
    int *rank_array = (int *)malloc(num_nodes * sizeof(int));
    if (!rank_array) {
        fprintf(stderr, "rank array not allocated successfully\n");
        return -1;
    }

    int *row_d;
    int *col_d;
    int *inrow_d;
    int *incol_d;
    int *index_d;

    // Create device-side buffers for the graph
    err = cudaMalloc(&row_d, num_nodes * sizeof(int));
    if (err != cudaSuccess) {
        fprintf(stderr, "ERROR: cudaMalloc row_d (size:%d) => %s\n", num_nodes,
                cudaGetErrorString(err));
        return -1;
    }
    err = cudaMalloc(&col_d, num_edges * sizeof(int));
    if (err != cudaSuccess) {
        fprintf(stderr, "ERROR: cudaMalloc col_d (size:%d) => %s\n", num_edges,
                cudaGetErrorString(err));
        return -1;
    }
    err = cudaMalloc(&inrow_d, num_nodes * sizeof(int));
    if (err != cudaSuccess) {
        fprintf(stderr, "ERROR: cudaMalloc inrow_d (size:%d) => %s\n",
                num_nodes, cudaGetErrorString(err));
        return -1;
    }
    err = cudaMalloc(&incol_d, num_edges * sizeof(int));
    if (err != cudaSuccess) {
        fprintf(stderr, "ERROR: cudaMalloc incol_d (size:%d) => %s\n",
                num_edges, cudaGetErrorString(err));
        return -1;
    }

    // Create buffers for index
    err = cudaMalloc(&index_d, num_nodes * sizeof(int));
    if (err != cudaSuccess) {
        fprintf(stderr, "ERROR: cudaMalloc index_d (size:%d) => %s\n",
                num_nodes, cudaGetErrorString(err));
        return -1;
    }

    double timer1 = gettime();

    // Copy the data to the device-side buffers
    err = cudaMemcpy(row_d, csr->row_array, num_nodes * sizeof(int),
                     cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "ERROR:#endif cudaMemcpy row_d (size:%d) => %s\n",
                num_nodes, cudaGetErrorString(err));
        return -1;
    }

    err = cudaMemcpy(col_d, csr->col_array, num_edges * sizeof(int),
                     cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "ERROR: cudaMemcpy col_d (size:%d) => %s\n", num_nodes,
                cudaGetErrorString(err));
        return -1;
    }
    err = cudaMemcpy(inrow_d, csr->inrow_array, num_nodes * sizeof(int),
                     cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "ERROR:#endif cudaMemcpy inrow_d (size:%d) => %s\n",
                num_nodes, cudaGetErrorString(err));
        return -1;
    }

    err = cudaMemcpy(incol_d, csr->incol_array, num_edges * sizeof(int),
                     cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "ERROR: cudaMemcpy incol_d (size:%d) => %s\n",
                num_nodes, cudaGetErrorString(err));
        return -1;
    }

    // Set up work dimensions
    int block_size = 256;
    int num_blocks = (num_nodes + block_size - 1) / block_size;

    dim3 threads(block_size, 1, 1);
    dim3 grid(num_blocks, 1, 1);

    cudaDeviceSetLimit(cudaLimitMallocHeapSize, 4ULL * 1024 * 1024 * 1024);
    double timer3 = gettime();

    ChiVertex<int, int> **vertex;
    GraphChiContext *context;
    err = cudaMalloc(&vertex, num_nodes * sizeof(ChiVertex<int, int> *));
    if (err != cudaSuccess) {
        fprintf(stderr, "ERROR: cudaMalloc vertex (size:%d) => %s\n", num_edges,
                cudaGetErrorString(err));
        return -1;
    }
    err = cudaMalloc(&context, sizeof(GraphChiContext));
    if (err != cudaSuccess) {
        fprintf(stderr, "ERROR: cudaMalloc context (size:%d) => %s\n",
                num_edges, cudaGetErrorString(err));
        return -1;
    }
    printf("Start initCtx\n");
    initContext<<<1, 1>>>(context, num_nodes, num_edges);
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "ERROR: initCtx failed (%s)\n",
                cudaGetErrorString(err));
        return -1;
    }

    printf("Start initObj\n");
    initObject<<<grid, threads>>>(vertex, context, row_d, col_d, inrow_d,
                                  incol_d);
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "ERROR: initObject failed (%s)\n",
                cudaGetErrorString(err));
        return -1;
    }

    printf("Start initOutEdge\n");
    initOutEdge<<<grid, threads>>>(vertex, context, row_d, col_d);
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "ERROR: initOutEdge failed (%s)\n",
                cudaGetErrorString(err));
        return -1;
    }

    // Run BFS for some iter. TO: convergence determination
    for (int i = 0; i < ITER; i++) {
        printf("Start BFS\n");
        BFS<<<grid, threads>>>(vertex, context, i);
        printf("Finish BFS\n");
        cudaDeviceSynchronize();
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "ERROR: cudaLaunch failed (%s)\n",
                    cudaGetErrorString(err));
            return -1;
        }
    }
    cudaDeviceSynchronize();

    double timer4 = gettime();

    printf("Start Copyback\n");
    copyBack<<<grid, threads>>>(vertex, context, index_d);
    printf("End Copyback\n");
    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "ERROR: cudaLaunch failed (%s)\n",
                cudaGetErrorString(err));
        return -1;
    }
    // Copy the rank buffer back
    err = cudaMemcpy(rank_array, index_d, num_nodes * sizeof(int),
                     cudaMemcpyDeviceToHost);

    if (err != cudaSuccess) {
        fprintf(stderr, "ERROR: cudaMemcpy() failed (%s)\n",
                cudaGetErrorString(err));
        return -1;
    }

    double timer2 = gettime();

    // Report timing characteristics
    printf("kernel time = %lf ms\n", (timer4 - timer3) * 1000);
    printf("kernel + memcpy time = %lf ms\n", (timer2 - timer1) * 1000);

#if 1
    // Print rank array
    print_vectorf(rank_array, num_nodes);
#endif

    // Free the host-side arrays
    free(rank_array);
    csr->freeArrays();
    free(csr);

    // Free the device buffers
    cudaFree(row_d);
    cudaFree(col_d);
    cudaFree(inrow_d);
    cudaFree(incol_d);

    cudaFree(index_d);

    return 0;
}

void print_vectorf(int *vector, int num) {
    FILE *fp = fopen("result.out", "w");
    if (!fp) {
        printf("ERROR: unable to open result.txt\n");
    }

    for (int i = 0; i < num; i++) {
        fprintf(fp, "%d\n", vector[i]);
    }

    fclose(fp);
}
