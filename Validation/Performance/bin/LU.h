#ifndef LU_H
#define LU_H

double LU_num_flops(int N);
void LU_copy_matrix(int M, int N, double **lu, double **A);
int LU_factor(int M, int N, double **A, int *pivot);


#endif
