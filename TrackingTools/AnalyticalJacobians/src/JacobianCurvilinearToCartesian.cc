#include "TrackingTools/AnalyticalJacobians/interface/JacobianCurvilinearToCartesian.h"
#include "TrackingTools/TrajectoryParametrization/interface/GlobalTrajectoryParameters.h"
#include "DataFormats/GeometryVector/interface/TrackingJacobians.h"

JacobianCurvilinearToCartesian::
JacobianCurvilinearToCartesian(const GlobalTrajectoryParameters& globalParameters) :
theJacobian(jacobianCurvilinearToCartesian(globalParameters.momentum(),globalParameters.charge())) {}


