#include <numeric>
#include <cmath>

/** A trivial class computing the mean value of objects in any
 *  STL container.
 */

template< class CONT> 
double stat_mean( const CONT & cont) {
  double sum = accumulate (cont.begin(), cont.end(), 0.);
  return sum / cont.size();
}

/** A simple class computing the R.M.S. of objects in any
 *  STL container.
 */


template< class CONT> 
double stat_RMS( const CONT & cont) {

  typename CONT::const_iterator i;

  int N = cont.size();
  if (N > 1) {
    double sum=0., sum2=0.;
    for (i=cont.begin(); i!=cont.end(); i++) {
      sum  += *i;
      sum2 += (*i) * (*i); 
    }
    return sqrt( std::max( 0., (sum2 - sum*sum/N) / (N-1))) ;
  }
  else return 0.;
}
