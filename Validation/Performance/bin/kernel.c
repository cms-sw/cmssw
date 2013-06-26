#include <stdio.h>
#include <stdlib.h>
#include "LU.h"
#include "FFT.h"
#include "SOR.h"
#include "MonteCarlo.h"
#include "LU.h"
#include "Random.h" 
#include "Stopwatch.h"  
#include "SparseCompRow.h"
#include "array.h"


    double kernel_measureFFT(int N, double mintime, Random R)
    {
        /* initialize FFT data as complex (N real/img pairs) */

        int twoN = 2*N;
        double *x = RandomVector(twoN, R);
        long cycles = 1;
        Stopwatch Q = new_Stopwatch();
        int i=0;
        double result = 0.0;

        while(1)
        {
            Stopwatch_start(Q);
            for (i=0; i<cycles; i++)
            {
                FFT_transform(twoN, x);     /* forward transform */
                FFT_inverse(twoN, x);       /* backward transform */
            }
            Stopwatch_stop(Q);
            if (Stopwatch_read(Q) >= mintime)
                break;

            cycles *= 2;

        }
        /* approx Mflops */

        result = FFT_num_flops(N)*cycles/ Stopwatch_read(Q) * 1.0e-6;
        Stopwatch_delete(Q);
        free(x);
        return result;
    }

    double kernel_measureSOR(int N, double min_time, Random R)
    {
        double **G = RandomMatrix(N, N, R);
        double result = 0.0;

        Stopwatch Q = new_Stopwatch();
        int cycles=1;
        while(1)
        {
            Stopwatch_start(Q);
            SOR_execute(N, N, 1.25, G, cycles);
            Stopwatch_stop(Q);

            if (Stopwatch_read(Q) >= min_time) break;

            cycles *= 2;
        }
        /* approx Mflops */

        result = SOR_num_flops(N, N, cycles) / Stopwatch_read(Q) * 1.0e-6;
        Stopwatch_delete(Q);
        Array2D_double_delete(N, N, G);
        return result;

    }



    double kernel_measureMonteCarlo(double min_time, Random R)
    {
        double result = 0.0;
        Stopwatch Q = new_Stopwatch();

        int cycles=1;
        while(1)
        {
            Stopwatch_start(Q);
            MonteCarlo_integrate(cycles);
            Stopwatch_stop(Q);
            if (Stopwatch_read(Q) >= min_time) break;

            cycles *= 2;
        }
        /* approx Mflops */
        result = MonteCarlo_num_flops(cycles) / Stopwatch_read(Q) * 1.0e-6;
        Stopwatch_delete(Q);
        return result;
    }


    double kernel_measureSparseMatMult(int N, int nz, 
            double min_time, Random R)
    {
        /* initialize vector multipliers and storage for result */
        /* y = A*y;  */

        double *x = RandomVector(N, R);
        double *y = (double*) malloc(sizeof(double)*N);

        double result = 0.0;

#if 0
        // initialize square sparse matrix
        //
        // for this test, we create a sparse matrix with M/nz nonzeros
        // per row, with spaced-out evenly between the begining of the
        // row to the main diagonal.  Thus, the resulting pattern looks
        // like
        //             +-----------------+
        //             +*                +
        //             +***              +
        //             +* * *            +
        //             +** *  *          +
        //             +**  *   *        +
        //             +* *   *   *      +
        //             +*  *   *    *    +
        //             +*   *    *    *  + 
        //             +-----------------+
        //
        // (as best reproducible with integer artihmetic)
        // Note that the first nr rows will have elements past
        // the diagonal.
#endif

        int nr = nz/N;      /* average number of nonzeros per row  */
        int anz = nr *N;    /* _actual_ number of nonzeros         */

            
        double *val = RandomVector(anz, R);
        int *col = (int*) malloc(sizeof(int)*nz);
        int *row = (int*) malloc(sizeof(int)*(N+1));
        int r=0;
        int cycles=1;

        Stopwatch Q = new_Stopwatch();

        row[0] = 0; 
        for (r=0; r<N; r++)
        {
            /* initialize elements for row r */

            int rowr = row[r];
            int step = r/ nr;
            int i=0;

            row[r+1] = rowr + nr;
            if (step < 1) step = 1;   /* take at least unit steps */


            for (i=0; i<nr; i++)
                col[rowr+i] = i*step;
                
        }


        while(1)
        {
            Stopwatch_start(Q);
            SparseCompRow_matmult(N, y, val, row, col, x, cycles);
            Stopwatch_stop(Q);
            if (Stopwatch_read(Q) >= min_time) break;

            cycles *= 2;
        }
        /* approx Mflops */
        result = SparseCompRow_num_flops(N, nz, cycles) / 
                        Stopwatch_read(Q) * 1.0e-6;

        Stopwatch_delete(Q);
        free(row);
        free(col);
        free(val);
        free(y);
        free(x);

        return result;
    }


    double kernel_measureLU(int N, double min_time, Random R)
    {

        double **A = NULL;
        double **lu = NULL; 
        int *pivot = NULL;

    

        Stopwatch Q = new_Stopwatch();
        double result = 0.0;
        int i=0;
        int cycles=1;

        if ((A = RandomMatrix(N, N,  R)) == NULL) exit(1);
        if ((lu = new_Array2D_double(N, N)) == NULL) exit(1);
        if ((pivot = (int *) malloc(N * sizeof(int))) == NULL) exit(1);


        while(1)
        {
            Stopwatch_start(Q);
            for (i=0; i<cycles; i++)
            {
                Array2D_double_copy(N, N, lu, A);
                LU_factor(N, N, lu, pivot);
            }
            Stopwatch_stop(Q);
            if (Stopwatch_read(Q) >= min_time) break;

            cycles *= 2;
        }
        /* approx Mflops */
        result = LU_num_flops(N) * cycles / Stopwatch_read(Q) * 1.0e-6;

        Stopwatch_delete(Q);
        free(pivot); 
        Array2D_double_delete(N, N, lu); 
        Array2D_double_delete(N, N, A);

        return result;

    }

