#include "TrackingTools/AnalyticalJacobians/interface/JacobianCartesianToCurvilinear.h"
#include "TrackingTools/TrajectoryParametrization/interface/GlobalTrajectoryParameters.h"
#include "DataFormats/GeometryVector/interface/TrackingJacobians.h"

JacobianCartesianToCurvilinear::
JacobianCartesianToCurvilinear(const GlobalTrajectoryParameters& globalParameters) : 
theJacobian(jacobianCartesianToCurvilinear(globalParameters.momentum(),globalParameters.charge())) {
}
