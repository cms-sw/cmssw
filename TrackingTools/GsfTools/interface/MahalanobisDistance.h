#ifndef MahalanobisDistance_H
#define MahalanobisDistance_H

#include "TrackingTools/GsfTools/interface/DistanceBetweenComponents.h"

class TrajectoryStateOnSurface;

/** Calculation of Mahalanobis distance between two Gaussian components.
 */

class MahalanobisDistance : public DistanceBetweenComponents {

 public:

  /** Method which calculates the actual Mahalanobis distance.
   */

  virtual double operator() (const RCSingleGaussianState&, 
			     const RCSingleGaussianState&) const;

  virtual MahalanobisDistance* clone() const
  {  
    return new MahalanobisDistance(*this);
  }
  
};  

#endif // MahalanobisDistance_H
