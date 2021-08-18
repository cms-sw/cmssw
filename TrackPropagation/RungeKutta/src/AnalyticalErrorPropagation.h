#ifndef AnalyticalErrorPropagation_H
#define AnalyticalErrorPropagation_H

#include "FWCore/Utilities/interface/Visibility.h"
#include "FWCore/Utilities/interface/Likely.h"
#include "TrackingTools/AnalyticalJacobians/interface/AnalyticalCurvilinearJacobian.h"
#include "TrackingTools/TrajectoryParametrization/interface/GlobalTrajectoryParameters.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "TrackingTools/TrajectoryState/interface/SurfaceSideDefinition.h"

class Surface;

inline std::pair<TrajectoryStateOnSurface, double> analyticalErrorPropagation(
    const FreeTrajectoryState& startingState,
    const Surface& surface,
    SurfaceSideDefinition::SurfaceSide side,
    const GlobalTrajectoryParameters& destParameters,
    const double& s) {
  if UNLIKELY (!startingState.hasError())
    // return state without errors
    return std::pair<TrajectoryStateOnSurface, double>(TrajectoryStateOnSurface(destParameters, surface, side), s);

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
  AnalyticalCurvilinearJacobian analyticalJacobian(
      startingState.parameters(), destParameters.position(), destParameters.momentum(), s);
  auto const& jacobian = analyticalJacobian.jacobian();
  return std::pair<TrajectoryStateOnSurface, double>(
      TrajectoryStateOnSurface(
          destParameters, ROOT::Math::Similarity(jacobian, startingState.curvilinearError().matrix()), surface, side),
      s);
}

#endif
