#include "TrackingTools/KalmanUpdators/interface/Chi2StripEstimator.h"
#include "Geometry/CommonDetUnit/interface/GeomDetType.h"
#include "Geometry/CommonTopologies/interface/StripTopology.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/Math/interface/invertPosDefMatrix.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"
using namespace std;

pair<bool,double>
Chi2StripEstimator::estimate(const TrajectoryStateOnSurface& state,
			     const TransientTrackingRecHit& hit) const {

  if(//hit.isMatched() || 
     hit.detUnit()->type().isTrackerPixel()) {
    return HitReturnType(false,0);
  }

  const StripTopology* topology = 
    dynamic_cast<const StripTopology*>(&(hit.detUnit()->topology())); 

  MeasurementPoint mp;
  MeasurementError me;

  AlgebraicVector2 m;
  mp = topology->measurementPosition(hit.localPosition());
  m[0] = mp.x();
  m[1] = mp.y();
  
  AlgebraicSymMatrix22 V;
  me = topology->measurementError(hit.localPosition(),
				  hit.localPositionError());
 
  V(0,0) = me.uu();
  V(1,0) = me.uv();
  V(1,1) = me.vv();

  AlgebraicVector2 x;
  mp = topology->measurementPosition(state.localPosition());
  m[0] = mp.x();
  m[1] = mp.y();
  
  AlgebraicSymMatrix22 C;
  me = topology->measurementError(state.localPosition(),
				  state.localError().positionError());
  C(0,0) = me.uu();
  C(1,0) = me.uv();
  C(1,1) = me.vv();

  AlgebraicVector2 r(m - x);
  AlgebraicSymMatrix22 R(V+C);
  bool ierr =  !invertPosDefMatrix(R);
  if (ierr) {
    edm::LogError("Chi2StripEstimator")<<" could not invert matrix:\n"<<(V+C);
    return returnIt( 0.0 );
  }

  double est = max(ROOT::Math::Similarity(r, R), 0.000001); // avoid exact zero

  return returnIt( est);
}

