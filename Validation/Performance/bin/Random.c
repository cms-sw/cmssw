

#include <stdlib.h>

#include "Random.h"

#ifndef NULL
#define NULL 0
#endif


  /* static const int mdig = 32; */
#define MDIG 32

  /* static const int one = 1; */
#define ONE 1

  static const int m1 = (ONE << (MDIG-2)) + ((ONE << (MDIG-2) )-ONE);
  static const int m2 = ONE << MDIG/2;

  /* For mdig = 32 : m1 =          2147483647, m2 =      65536
     For mdig = 64 : m1 = 9223372036854775807, m2 = 4294967296 
  */

                                /* move to initialize() because  */
                                /* compiler could not resolve as */
                                /*   a constant.                 */

  static /*const*/ double dm1;  /*  = 1.0 / (double) m1; */


/* private methods (defined below, but not in Random.h */

static void initialize(Random R, int seed);

Random new_Random_seed(int seed)
{
    Random R = (Random) malloc(sizeof(Random_struct));

    initialize(R, seed);
    R->left = 0.0;
    R->right = 1.0;
    R->width = 1.0;
    R->haveRange = 0 /*false*/;

    return R;
}

Random new_Random(int seed, double left, double right) 
{
    Random R = (Random) malloc(sizeof(Random_struct));

    initialize(R, seed);
    R->left = left;
    R->right = right;
    R->width = right - left;
    R->haveRange = 1;          /* true */

    return R;
}

void Random_delete(Random R)
{
    free(R);
}



/* Returns the next random number in the sequence.  */

double Random_nextDouble(Random R) 
{
    int k;

    int I = R->i;
    int J = R->j;
    int *m = R->m;

    k = m[I] - m[J];
    if (k < 0) k += m1;
    R->m[J] = k;

    if (I == 0) 
        I = 16;
    else I--;
    R->i = I;

    if (J == 0) 
        J = 16 ;
    else J--;
    R->j = J;

    if (R->haveRange) 
        return  R->left +  dm1 * (double) k * R->width;
    else
        return dm1 * (double) k;

} 




/*--------------------------------------------------------------------
                           PRIVATE METHODS
  ----------------------------------------------------------------- */

static void initialize(Random R, int seed) 
{

    int jseed, k0, k1, j0, j1, iloop;

    dm1  = 1.0 / (double) m1; 

    R->seed = seed;

    if (seed < 0 ) seed = -seed;            /* seed = abs(seed) */  
    jseed = (seed < m1 ? seed : m1);        /* jseed = min(seed, m1) */
    if (jseed % 2 == 0) --jseed;
    k0 = 9069 % m2;
    k1 = 9069 / m2;
    j0 = jseed % m2;
    j1 = jseed / m2;
    for (iloop = 0; iloop < 17; ++iloop) 
    {
        jseed = j0 * k0;
        j1 = (jseed / m2 + j0 * k1 + j1 * k0) % (m2 / 2);
        j0 = jseed % m2;
        R->m[iloop] = j0 + m2 * j1;
    }
    R->i = 4;
    R->j = 16;

}

double *RandomVector(int N, Random R)
{
    int i;
    double *x = (double *) malloc(sizeof(double)*N);

    for (i=0; i<N; i++)
        x[i] = Random_nextDouble(R);

    return x;
}


double **RandomMatrix(int M, int N, Random R)
{
    int i;
    int j;

    /* allocate matrix */

    double **A = (double **) malloc(sizeof(double*)*M);

    if (A == NULL) return NULL;

    for (i=0; i<M; i++)
    {
        A[i] = (double *) malloc(sizeof(double)*N);
        if (A[i] == NULL) 
        {
            free(A);
            return NULL;
        }
        for (j=0; j<N; j++)
            A[i][j] = Random_nextDouble(R);
    }
    return A;
}



