    /* multiple iterations used to make kernel have roughly
        same granulairty as other Scimark kernels. */

    double SparseCompRow_num_flops(int N, int nz, int num_iterations)
    {
        /* Note that if nz does not divide N evenly, then the
           actual number of nonzeros used is adjusted slightly.
        */
        int actual_nz = (nz/N) * N;
        return ((double)actual_nz) * 2.0 * ((double) num_iterations);
    }


    /* computes  a matrix-vector multiply with a sparse matrix
        held in compress-row format.  If the size of the matrix
        in MxN with nz nonzeros, then the val[] is the nz nonzeros,
        with its ith entry in column col[i].  The integer vector row[]
        is of size M+1 and row[i] points to the begining of the
        ith row in col[].  
    */

    void SparseCompRow_matmult( int M, double *y, double *val, int *row,
        int *col, double *x, int NUM_ITERATIONS)
    {
        int reps;
        int r;
        int i;

        for (reps=0; reps<NUM_ITERATIONS; reps++)
        {
        
            for (r=0; r<M; r++)
            {
                double sum = 0.0; 
                int rowR = row[r];
                int rowRp1 = row[r+1];
                for (i=rowR; i<rowRp1; i++)
                    sum += x[ col[i] ] * val[i];
                y[r] = sum;
            }
        }
    }

