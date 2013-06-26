#include "TrackingTools/KalmanUpdators/interface/Chi2Strip1DEstimator.h"
#include "Geometry/CommonDetUnit/interface/GeomDetType.h"
#include "Geometry/CommonTopologies/interface/StripTopology.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"
using namespace std;

pair<bool,double>
Chi2Strip1DEstimator::estimate(const TrajectoryStateOnSurface& state,
			       const TransientTrackingRecHit& hit) const {

  if(//hit.isMatched() || 
     hit.detUnit()->type().isTrackerPixel() 
     ) return HitReturnType(false,0.);
  
 
  const StripTopology* topology = 
    dynamic_cast<const StripTopology*>(&(hit.detUnit()->topology())); 
  
  double m = topology->measurementPosition(hit.localPosition()).x();
  
  double x = topology->measurementPosition(state.localPosition()).x();
  
  double V =
    topology->measurementError(hit.localPosition(),
			       hit.localPositionError()).uu();
  double C = 
    topology->measurementError(state.localPosition(),
			       state.localError().positionError()).uu();
  double r = m - x;
  double R = V + C;

  double est = r*r/R;

  return returnIt(est);
}

