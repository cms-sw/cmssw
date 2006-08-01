#ifndef TSOSKullbackLeiblerDistance_H
#define TSOSKullbackLeiblerDistance_H

#include "TrackingTools/GsfTracking/interface/TSOSDistanceBetweenComponents.h"

class TrajectoryStateOnSurface;

/** Calculation of Kullback-Leibler distance between two Gaussian components.
 */

class TSOSKullbackLeiblerDistance : public TSOSDistanceBetweenComponents {

 public:

  /** Method which calculates the actual Kullback-Leibler distance.
   */

  virtual double operator() (const TrajectoryStateOnSurface&, 
			     const TrajectoryStateOnSurface&) const;

  virtual TSOSKullbackLeiblerDistance* clone() const
  {  
    return new TSOSKullbackLeiblerDistance(*this);
  }

  
};  

#endif // TSOSKullbackLeiblerDistance_H
