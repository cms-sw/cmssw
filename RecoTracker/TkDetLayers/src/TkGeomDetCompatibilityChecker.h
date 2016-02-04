#ifndef TkGeomDetCompatibilityChecker_H
#define TkGeomDetCompatibilityChecker_H

#include <utility>

#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"

class GeomDet;
class Propagator;
class MeasurementEstimator;

#pragma GCC visibility push(hidden)
class TkGeomDetCompatibilityChecker {
public:

  std::pair<bool, TrajectoryStateOnSurface>  
  isCompatible(const GeomDet* det,
	       const TrajectoryStateOnSurface& tsos,
	       const Propagator& prop, 
	       const MeasurementEstimator& est) const;


};

#pragma GCC visibility pop
#endif
