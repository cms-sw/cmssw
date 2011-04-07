#ifndef TkDetLayers_CompatibleDetToGroupAdder_h
#define TkDetLayers_CompatibleDetToGroupAdder_h

#include "TrackingTools/DetLayers/interface/DetGroup.h"
#include "TrackingTools/DetLayers/interface/GeometricSearchDet.h"


#pragma GCC visibility push(hidden)

class TrajectoryStateOnSurface;
class Propagator;
class MeasurementEstimator;

class CompatibleDetToGroupAdder {
public:

  /** Checks the det for compatibility with the tsos propagated by prop and according to est;
   *  if the det is compatible, it is added to result and the method returns true, 
   *  if not result is not modified and the method returns false.
   *  The method for chacking for compatibility used depends on the det:
   *  if the det hasGroups, then groupedCompatibleDets() is used, otherwise
   *  compatible() is used.
   */

  static bool add( const GeometricSearchDet& det,
	    const TrajectoryStateOnSurface& tsos, 
	    const Propagator& prop,
	    const MeasurementEstimator& est,
	    std::vector<DetGroup>& result);
  

  static bool add( const GeomDet& det,
	    const TrajectoryStateOnSurface& tsos, 
	    const Propagator& prop,
	    const MeasurementEstimator& est,
	    std::vector<DetGroup>& result);

};

#pragma GCC visibility pop
#endif
