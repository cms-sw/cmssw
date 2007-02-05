#ifndef DistanceBetweenComponents_H
#define DistanceBetweenComponents_H

#include "TrackingTools/GsfTools/interface/RCSingleGaussianState.h"

/** Base class (abstract) of calculation of distance between
 *  two Gaussian components.
 */

class DistanceBetweenComponents {

 public:

  virtual double operator() (const RCSingleGaussianState&, 
			     const RCSingleGaussianState&) const = 0;

  virtual DistanceBetweenComponents* clone() const = 0;
  
};  

#endif // DistanceBetweenComponents_H
