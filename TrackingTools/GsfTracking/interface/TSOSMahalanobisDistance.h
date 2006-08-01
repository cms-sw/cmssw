#ifndef TSOSMahalanobisDistance_H
#define TSOSMahalanobisDistance_H

#include "TrackingTools/GsfTracking/interface/TSOSDistanceBetweenComponents.h"

class TrajectoryStateOnSurface;

/** Calculation of Mahalanobis distance between two Gaussian components.
 */

class TSOSMahalanobisDistance : public TSOSDistanceBetweenComponents {

 public:

  /** Method which calculates the actual Mahalanobis distance.
   */

  virtual double operator() (const TrajectoryStateOnSurface&, 
			     const TrajectoryStateOnSurface&) const;

  virtual TSOSMahalanobisDistance* clone() const
  {  
    return new TSOSMahalanobisDistance(*this);
  }
  
};  

#endif // TSOSMahalanobisDistance_H
