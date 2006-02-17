#ifndef MeasurementExtractor_H
#define MeasurementExtractor_H
 
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"

 
/** Extracts the subset of TrajectoryState parameters and errors
 *  that correspond to the parameters measured by a RecHit.
 */
 
class MeasurementExtractor {
 public:
  // construct
  MeasurementExtractor(const TrajectoryStateOnSurface& aTSoS) :
    theTSoS(aTSoS) {}
 
  // access
  
  // Following methods can be overloaded against their argument
  // thus allowing one to have different behaviour for different RecHit types
 
  AlgebraicVector measuredParameters(const  TransientTrackingRecHit&);
 
  AlgebraicSymMatrix measuredError(const  TransientTrackingRecHit&);
  
 private:
  const TrajectoryStateOnSurface& theTSoS;
};

#endif
 

