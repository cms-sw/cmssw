#include "TrackingTools/KalmanUpdators/interface/Chi2StripEstimator.h"
// #include "CommonDet/BasicDet/interface/Det.h"
// #include "CommonDet/BasicDet/interface/DetUnit.h"
#include "Geometry/CommonDetUnit/interface/GeomDetType.h"
#include "Geometry/CommonTopologies/interface/StripTopology.h"

using namespace std;

pair<bool,double>
Chi2StripEstimator::estimate(const TrajectoryStateOnSurface& state,
			     const TransientTrackingRecHit& hit) const {

  if(hit.dimension()==2 || 
     hit.detUnit()->type().isTrackerPixel()) {
    return HitReturnType(false,0);
  }

  const StripTopology* topology = 
    dynamic_cast<const StripTopology*>(&(hit.detUnit()->topology())); 

  MeasurementPoint mp;
  MeasurementError me;

  AlgebraicVector m(2,0);
  mp = topology->measurementPosition(hit.localPosition());
  m(1) = mp.x();
  m(2) = mp.y();
  
  AlgebraicSymMatrix V(2,0);
  me = topology->measurementError(hit.localPosition(),
				  hit.localPositionError());
 
  V(1,1) = me.uu();
  V(2,1) = me.uv();
  V(2,2) = me.vv();

  AlgebraicVector x(2,0);
  mp = topology->measurementPosition(state.localPosition());
  x(1) = mp.x();
  x(2) = mp.y();
  
  AlgebraicSymMatrix C(2,0);
  me = topology->measurementError(state.localPosition(),
				  state.localError().positionError());
  C(1,1) = me.uu();
  C(2,1) = me.uv();
  C(2,2) = me.vv();

  AlgebraicVector r(m - x);
  AlgebraicSymMatrix R(V + C);
  int ierr; R.invert(ierr); // if (ierr != 0) throw exception;
  double est = max(R.similarity(r), 0.000001); // avoid exact zero

  return returnIt( est);
}

