#include "AnalyticalErrorPropagation.h"

#include "TrackingTools/AnalyticalJacobians/interface/AnalyticalCurvilinearJacobian.h"
#include "TrackingTools/TrajectoryState/interface/SurfaceSideDefinition.h"

//#include "FWCore/MessageLogger/interface/MessageLogger.h"


std::pair<TrajectoryStateOnSurface,double>
AnalyticalErrorPropagation::operator()( const FreeTrajectoryState& startingState, 
					const Surface& surface, SurfaceSideDefinition::SurfaceSide side,
					const GlobalTrajectoryParameters& destParameters, 
					const double& s) const
{
  if (startingState.hasError()) {

    //
    // compute jacobian
    //

    // FIXME: Compute mean B field between startingState and destParameters and pass it to analyticalJacobian
    //GlobalPoint xStart = startingState.position();
    //GlobalPoint xDest = destParameters.position();
    //GlobalVector h1  = destParameters.magneticFieldInInverseGeV(xStart);
    //GlobalVector h2  = destParameters.magneticFieldInInverseGeV(xDest);
    //GlobalVector h = 0.5*(h1+h2);
    //LogDebug("RungeKutta") << "AnalyticalErrorPropagation: The Fields are: " << h1 << ", " << h2 << ", " << h ; 
    
    // 
    AnalyticalCurvilinearJacobian analyticalJacobian(startingState.parameters(), 
						     destParameters.position(), 
						     destParameters.momentum(), s);
    AlgebraicMatrix55 jacobian(analyticalJacobian.jacobian());
    CurvilinearTrajectoryError cte( ROOT::Math::Similarity(jacobian, startingState.curvilinearError().matrix()));
    return std::pair<TrajectoryStateOnSurface,double>(TrajectoryStateOnSurface(destParameters,cte,surface,side),s);
  }
  else {
    //
    // return state without errors
    //
    return std::pair<TrajectoryStateOnSurface,double>(TrajectoryStateOnSurface(destParameters,surface,side),s);
  }
}

