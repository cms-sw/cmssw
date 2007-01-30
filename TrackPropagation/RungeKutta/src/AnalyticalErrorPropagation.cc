#include "TrackPropagation/RungeKutta/interface/AnalyticalErrorPropagation.h"

#include "TrackingTools/AnalyticalJacobians/interface/AnalyticalCurvilinearJacobian.h"
#include "TrackingTools/TrajectoryState/interface/SurfaceSideDefinition.h"

std::pair<TrajectoryStateOnSurface,double>
AnalyticalErrorPropagation::operator()( const FreeTrajectoryState& startingState, 
					const Surface& surface, SurfaceSide side,
					const GlobalTrajectoryParameters& destParameters, 
					const double& s) const
{
  if (startingState.hasError()) {
    //
    // compute jacobian
    //

    // FIXME: Compute mean B field between startingState and destParameters and pass it to analyticalJacobian
    AnalyticalCurvilinearJacobian analyticalJacobian(startingState.parameters(), 
						     destParameters.position(), 
						     destParameters.momentum(), s);
    AlgebraicMatrix jacobian(analyticalJacobian.jacobian());
    CurvilinearTrajectoryError cte(startingState.curvilinearError().matrix().similarity(jacobian));
    return std::pair<TrajectoryStateOnSurface,double>(TrajectoryStateOnSurface(destParameters,cte,surface,side),s);
  }
  else {
    //
    // return state without errors
    //
    return std::pair<TrajectoryStateOnSurface,double>(TrajectoryStateOnSurface(destParameters,surface,side),s);
  }
}

