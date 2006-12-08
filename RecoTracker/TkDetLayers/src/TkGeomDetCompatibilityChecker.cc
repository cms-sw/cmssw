#include "RecoTracker/TkDetLayers/interface/TkGeomDetCompatibilityChecker.h"
#include "TrackingTools/DetLayers/interface/GeomDetCompatibilityChecker.h"
#include "Geometry/TrackerGeometryBuilder/interface/GluedGeomDet.h"

using namespace std;

pair<bool, TrajectoryStateOnSurface>  
TkGeomDetCompatibilityChecker::isCompatible(const GeomDet* det,
					    const TrajectoryStateOnSurface& tsos,
					    const Propagator& prop, 
					    const MeasurementEstimator& est) const
{
  GeomDetCompatibilityChecker checker;
  const GluedGeomDet* glued = dynamic_cast<const GluedGeomDet*>( det);
  if (glued == 0) return checker.isCompatible( det, tsos, prop, est);
  else {
    pair<bool, TrajectoryStateOnSurface> mono = 
      checker.isCompatible(glued->monoDet(), tsos, prop, est);
    pair<bool, TrajectoryStateOnSurface> stereo = 
      checker.isCompatible(glued->stereoDet(), tsos, prop, est);
    if (mono.first || stereo.first) {
      TrajectoryStateOnSurface gluedDetState = prop.propagate( tsos, det->specificSurface());
      if (gluedDetState.isValid()) {
	return pair<bool, TrajectoryStateOnSurface>( true, gluedDetState);
      }
      else {
	return pair<bool, TrajectoryStateOnSurface>( false, gluedDetState);
      }
    }
    else {
      return pair<bool, TrajectoryStateOnSurface>( false, TrajectoryStateOnSurface());
    }
  }
}
 
