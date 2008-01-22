#include <stdlib.h>
#include <stdio.h>
#include "array.h"

#ifndef NULL
#define NULL 0
#endif


double** new_Array2D_double(int M, int N)
{
    int i=0;
    int failed = 0;

    double **A = (double**) malloc(sizeof(double*)*M);
    if (A == NULL)
        return NULL;

    for (i=0; i<M; i++)
    {
        A[i] = (double*) malloc(N * sizeof(double));
        if (A[i] == NULL)
        {
            failed = 1;
            break;
        }
    }

    /* if we didn't successfully allocate all rows of A      */
    /* clean up any allocated memory (i.e. go back and free  */
    /* previous rows) and return NULL                        */

    if (failed)
    {
        i--;
        for (; i<=0; i--)
            free(A[i]);
        free(A);
        return NULL;
    }
    else
        return A;
}
void Array2D_double_delete(int M, int N, double **A)
{
    int i;
    if (A == NULL) return;

    for (i=0; i<M; i++)
        free(A[i]);

    free(A);
}


  void Array2D_double_copy(int M, int N, double **B, double **A)
  {

        int remainder = N & 3;       /* N mod 4; */
        int i=0;
        int j=0;

        for (i=0; i<M; i++)
        {
            double *Bi = B[i];
            double *Ai = A[i];
            for (j=0; j<remainder; j++)
                Bi[j] = Ai[j];
            for (j=remainder; j<N; j+=4)
            {
                Bi[j] = Ai[j];
                Bi[j+1] = Ai[j+1];
                Bi[j+2] = Ai[j+2];
                Bi[j+3] = Ai[j+3];
            }
        }
  }
