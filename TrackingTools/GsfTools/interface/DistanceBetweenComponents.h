#ifndef DistanceBetweenComponents_H
#define DistanceBetweenComponents_H

#include "TrackingTools/GsfTools/interface/SingleGaussianState.h"

/** Base class (abstract) of calculation of distance between
 *  two Gaussian components.
 */

template <unsigned int N>
class DistanceBetweenComponents {
public:
  typedef SingleGaussianState<N> SingleState;
 public:

  virtual double operator() (const SingleState&, 
			     const SingleState&) const = 0;

  virtual DistanceBetweenComponents<N>* clone() const = 0;
  
};  

#endif // DistanceBetweenComponents_H
