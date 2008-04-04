#include "TrackingTools/PatternTools/interface/TrajectoryStateClosestToBeamLineBuilder.h"
#include "TrackingTools/AnalyticalJacobians/interface/AnalyticalCurvilinearJacobian.h"
#include "TrackingTools/PatternTools/interface/TwoTrackMinimumDistance.h"

TrajectoryStateClosestToBeamLine
TrajectoryStateClosestToBeamLineBuilder::operator()
	(const FreeTrajectoryState& originalFTS,
	 const reco::BeamSpot& beamSpot)
{
  TwoTrackMinimumDistance ttmd;
  pair<GlobalPoint, GlobalPoint> points = ttmd.points( originalFTS.parameters(), 
  	GlobalTrajectoryParameters(
	   	GlobalPoint(beamSpot.position().x(), beamSpot.position().y(), beamSpot.position().z()), 
		GlobalVector(beamSpot.dxdz(), beamSpot.dydz(), 1.), 
		0, &(originalFTS.parameters().magneticField()) ) );
 
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

// void TrajectoryStateClosestToBeamLine::create(const FreeTrajectoryState& originalFTS,
//     const reco::BeamSpot& beamSpot)
// {
//   TwoTrackMinimumDistanceLineLine ttmd;
//   ttmd.calculate( originalFTS.parameters(), 
//   	GlobalTrajectoryParameters(
// 	   	GlobalPoint(beamSpot.position().x(), beamSpot.position().y(), beamSpot.position().z()), 
// 		GlobalVector(beamSpot.dxdz(), beamSpot.dydz(), 1.), 
// 		0, &(originalFTS.parameters().magneticField()) ) );
//   pair<GlobalPoint, GlobalPoint> points = ttmd.points();
//  
//   GlobalPoint xTrack = points.first;
//   GlobalVector pTrack = originalFTS.momentum();
//   double s =  ttmd.pathLength().first;
// 
//   pointOnBeamLine = points.second;
// 
//   if (originalFTS.hasError()) {
//     const AlgebraicSymMatrix55 &errorMatrix = originalFTS.curvilinearError().matrix();
//     AnalyticalCurvilinearJacobian curvilinJacobian(originalFTS.parameters(), xTrack,
// 						   pTrack, s);
//     const AlgebraicMatrix55 &jacobian = curvilinJacobian.jacobian();
//     CurvilinearTrajectoryError cte( ROOT::Math::Similarity(jacobian, errorMatrix) );
//   
//     theFTS = FreeTrajectoryState(GlobalTrajectoryParameters(xTrack, pTrack, originalFTS.charge(), 
//     					&(originalFTS.parameters().magneticField())),
// 			        cte);
//   }
//   else {
//     theFTS = FreeTrajectoryState(GlobalTrajectoryParameters(xTrack, pTrack, originalFTS.charge(),
//     					&(originalFTS.parameters().magneticField())));
//   }
// 
// }
