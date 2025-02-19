#ifndef KERNEL_H
#define KERNEL_H

double kernel_measureFFT( int FFT_size, double min_time, Random R);
double kernel_measureSOR( int SOR_size, double min_time, Random R);
double kernel_measureMonteCarlo( double min_time, Random R);
double kernel_measureSparseMatMult(int Sparse_size_N,
    int Sparse_size_nz, double min_time, Random R);
double kernel_measureLU( int LU_size, double min_time, Random R);

#endif
