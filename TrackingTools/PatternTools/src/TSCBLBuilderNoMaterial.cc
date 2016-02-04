#include "TrackingTools/PatternTools/interface/TSCBLBuilderNoMaterial.h"
#include "TrackingTools/AnalyticalJacobians/interface/AnalyticalCurvilinearJacobian.h"
#include "TrackingTools/PatternTools/interface/TwoTrackMinimumDistance.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

using namespace std;

TrajectoryStateClosestToBeamLine
TSCBLBuilderNoMaterial::operator()
	(const FreeTrajectoryState& originalFTS,
	 const reco::BeamSpot& beamSpot) const
{
  TwoTrackMinimumDistance ttmd;
  bool status = ttmd.calculate( originalFTS.parameters(), 
  	GlobalTrajectoryParameters(
	   	GlobalPoint(beamSpot.position().x(), beamSpot.position().y(), beamSpot.position().z()), 
		GlobalVector(beamSpot.dxdz(), beamSpot.dydz(), 1.), 
		0, &(originalFTS.parameters().magneticField()) ) );
  if (!status) {
    LogDebug  ("TrackingTools|PatternTools")
      << "TSCBLBuilderNoMaterial: Failure in TTMD when searching for PCA of track to beamline.\n"
      << "TrajectoryStateClosestToBeamLine is now invalid.";
    return TrajectoryStateClosestToBeamLine();
  }

  pair<GlobalPoint, GlobalPoint> points = ttmd.points();

  GlobalPoint xTrack = points.first;
  GlobalVector pTrack = GlobalVector ( GlobalVector::Cylindrical(originalFTS.momentum().perp(), ttmd.firstAngle(), originalFTS.momentum().z()) );

  double s =  ttmd.pathLength().first;

  FreeTrajectoryState theFTS;
  if (originalFTS.hasError()) {
    const AlgebraicSymMatrix55 &errorMatrix = originalFTS.curvilinearError().matrix();
    AnalyticalCurvilinearJacobian curvilinJacobian(originalFTS.parameters(), xTrack,
						   pTrack, s);
    const AlgebraicMatrix55 &jacobian = curvilinJacobian.jacobian();
    CurvilinearTrajectoryError cte( ROOT::Math::Similarity(jacobian, errorMatrix) );
  
    theFTS = FreeTrajectoryState(GlobalTrajectoryParameters(xTrack, pTrack, originalFTS.charge(), 
    					&(originalFTS.parameters().magneticField())),
			        cte);
  }
  else {
    theFTS = FreeTrajectoryState(GlobalTrajectoryParameters(xTrack, pTrack, originalFTS.charge(),
    					&(originalFTS.parameters().magneticField())));
  }
  return TrajectoryStateClosestToBeamLine(theFTS, points.second, beamSpot);
}
