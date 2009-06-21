#include "TrackingTools/AnalyticalJacobians/interface/JacobianLocalToCartesian.h"
#include "DataFormats/GeometrySurface/interface/Surface.h"
#include "TrackingTools/TrajectoryParametrization/interface/LocalTrajectoryParameters.h"

#include "DataFormats/GeometrySurface/interface/Plane.h"



#include "FWCore/ Utilities/ interface/ HRRealTime.h"
int main() {

  Surface::RotationType rot;
  Surface::PositionType pos( 0., 0., 0.);

  Plane plane(pos,rot);

  LocalTrajectoryParameters tp(1.1.,1.,0.,0.,1.);

  emd::HRTimeType s= hrRealTime();
  JacobianLocalToCartesia jl2c(plane,tp);
  emd::HRTimeType e = hrRealTime();

  std::cout << e-s << std::endl;

  return 0;

}
