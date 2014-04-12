#include "Random.h"

/**
 Estimate Pi by approximating the area of a circle.

 How: generate N random numbers in the unit square, (0,0) to (1,1)
 and see how are within a radius of 1 or less, i.e.
 <pre>  

 sqrt(x^2 + y^2) < r

 </pre>
  since the radius is 1.0, we can square both sides
  and avoid a sqrt() computation:
  <pre>

    x^2 + y^2 <= 1.0

  </pre>
  this area under the curve is (Pi * r^2)/ 4.0,
  and the area of the unit of square is 1.0,
  so Pi can be approximated by 
  <pre>
                # points with x^2+y^2 < 1
     Pi =~      --------------------------  * 4.0
                     total # points

  </pre>

*/

static const int SEED = 113;


    double MonteCarlo_num_flops(int Num_samples)
    {
        /* 3 flops in x^2+y^2 and 1 flop in random routine */

        return ((double) Num_samples)* 4.0;

    }

    

    double MonteCarlo_integrate(int Num_samples)
    {


        Random R = new_Random_seed(SEED);


        int under_curve = 0;
        int count;

        for (count=0; count<Num_samples; count++)
        {
            double x= Random_nextDouble(R);
            double y= Random_nextDouble(R);

            if ( x*x + y*y <= 1.0)
                 under_curve ++;
            
        }

        Random_delete(R);

        return ((double) under_curve / Num_samples) * 4.0;
    }


