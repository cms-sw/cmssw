#include "TkGeomDetCompatibilityChecker.h"
#include "TrackingTools/DetLayers/interface/GeomDetCompatibilityChecker.h"
#include "Geometry/TrackerGeometryBuilder/interface/GluedGeomDet.h"
#include "DataFormats/SiStripDetId/interface/SiStripDetId.h"
 
using namespace std;

pair<bool, TrajectoryStateOnSurface>  
TkGeomDetCompatibilityChecker::isCompatible(const GeomDet* det,
					    const TrajectoryStateOnSurface& tsos,
					    const Propagator& prop, 
					    const MeasurementEstimator& est) const
{
  GeomDetCompatibilityChecker checker;
  SiStripDetId siStripDetId(det->geographicalId());
  if(!siStripDetId.glued()) return checker.isCompatible( det, tsos, prop, est);
  else {
    const GluedGeomDet* glued = static_cast<const GluedGeomDet*>( det);
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
 
