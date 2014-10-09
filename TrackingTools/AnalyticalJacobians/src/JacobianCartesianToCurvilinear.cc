#include "TrackingTools/AnalyticalJacobians/interface/JacobianCartesianToCurvilinear.h"
#include "TrackingTools/TrajectoryParametrization/interface/GlobalTrajectoryParameters.h"
#include "DataFormats/GeometryVector/interface/jacobians.h"

JacobianCartesianToCurvilinear::
JacobianCartesianToCurvilinear(const GlobalTrajectoryParameters& globalParameters) : 
theJacobian(jacobians::jacobianCartesianToCurvilinear(globalParameters.momentum(),globalParameters.charge())) {
}
