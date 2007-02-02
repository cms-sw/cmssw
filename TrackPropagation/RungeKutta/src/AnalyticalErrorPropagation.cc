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
    GlobalPoint xStart = startingState.position();
    GlobalPoint xDest = destParameters.position();
    GlobalVector h1  = destParameters.megneticFieldInInverseGeV(xStart);
    GlobalVector h2  = destParameters.megneticFieldInInverseGeV(xDest);
    GlobalVector h = 0.5*(h1+h2);
    std::cout << "The Fields are: " << h1 << ", " << h2 << ", " << h << std::endl; 
    
  // 
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

