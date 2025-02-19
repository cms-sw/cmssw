
#ifndef SPARSE_COMPROW_H
#define SPARSE_COMPROW_H

double SparseCompRow_num_flops(int N, int nz, int num_iterations);

void SparseCompRow_matmult( int M, double *y, double *val, int *row,
        int *col, double *x, int NUM_ITERATIONS);

#endif
