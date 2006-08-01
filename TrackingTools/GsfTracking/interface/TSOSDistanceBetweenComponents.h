#ifndef TSOSDistanceBetweenComponents_H
#define TSOSDistanceBetweenComponents_H

#include "TrackingTools/GsfTracking/interface/MultiTrajectoryStateMerger.h"

class TrajectoryStateOnSurface;

/** Base class (abstract) of calculation of distance between
 *  two Gaussian components.
 */

class TSOSDistanceBetweenComponents {

 public:

  virtual double operator() (const TrajectoryStateOnSurface&, 
			     const TrajectoryStateOnSurface&) const = 0;

  virtual TSOSDistanceBetweenComponents* clone() const = 0;
  
};  

#endif // TSOSDistanceBetweenComponents_H
