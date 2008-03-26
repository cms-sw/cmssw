#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "Random.h"
#include "kernel.h"
#include "constants.h"

void print_banner(void);

int main(int argc, char *argv[])
{
        /* default to the (small) cache-contained version */

        double min_time = RESOLUTION_DEFAULT;

        int FFT_size = FFT_SIZE;
        int SOR_size =  SOR_SIZE;
        int Sparse_size_M = SPARSE_SIZE_M;
        int Sparse_size_nz = SPARSE_SIZE_nz;
        int LU_size = LU_SIZE;


        /* run the benchmark */

        double res[6] = {0.0};
        Random R = new_Random_seed(RANDOM_SEED);


        if (argc > 1)
        {
			int current_arg = 1;

			if (strcmp(argv[1], "-help")==0  ||
					strcmp(argv[1], "-h") == 0)
			{
				fprintf(stderr, "Usage: [-large] [minimum_time]\n");
				exit(0);
			}

			if (strcmp(argv[1], "-large")==0)
			{
				FFT_size = LG_FFT_SIZE;
				SOR_size = LG_SOR_SIZE;
				Sparse_size_M = LG_SPARSE_SIZE_M;
				Sparse_size_nz = LG_SPARSE_SIZE_nz;
				LU_size = LG_LU_SIZE;

				current_arg++;
			}

			if (current_arg < argc)
			{
				min_time = atof(argv[current_arg]);
			}
			
        }

	
		print_banner();
		printf("Using %10.2f seconds min time per kenel.\n", min_time);

        res[1] = kernel_measureFFT( FFT_size, min_time, R);   
        res[2] = kernel_measureSOR( SOR_size, min_time, R);   
        res[3] = kernel_measureMonteCarlo(min_time, R); 
        res[4] = kernel_measureSparseMatMult( Sparse_size_M, 
                Sparse_size_nz, min_time, R);           
        res[5] = kernel_measureLU( LU_size, min_time, R);  



        res[0] = (res[1] + res[2] + res[3] + res[4] + res[5]) / 5;

        /* print out results  */
        printf("Composite Score:        %8.2f\n" ,res[0]);
        printf("FFT             Mflops: %8.2f    (N=%d)\n", res[1], FFT_size);
        printf("SOR             Mflops: %8.2f    (%d x %d)\n", 		
				res[2], SOR_size, SOR_size);
        printf("MonteCarlo:     Mflops: %8.2f\n", res[3]);
        printf("Sparse matmult  Mflops: %8.2f    (N=%d, nz=%d)\n", res[4], 
					Sparse_size_M, Sparse_size_nz);
        printf("LU              Mflops: %8.2f    (M=%d, N=%d)\n", res[5],
				LU_size, LU_size);


        Random_delete(R);

        return 0;
  
}

void print_banner()
{
 printf("**                                                              **\n");
 printf("** SciMark2 Numeric Benchmark, see http://math.nist.gov/scimark **\n");
 printf("** for details. (Results can be submitted to pozo@nist.gov)     **\n");
 printf("**                                                              **\n");
}
