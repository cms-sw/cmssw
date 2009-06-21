#include "TrackingTools/AnalyticalJacobians/interface/JacobianLocalToCartesian.h"
#include "DataFormats/GeometrySurface/interface/Surface.h"
#include "TrackingTools/TrajectoryParametrization/interface/LocalTrajectoryParameters.h"

#include "DataFormats/GeometrySurface/interface/Plane.h"



#include "FWCore/ Utilities/ interface/ HRRealTime.h"
int main() {

  Surface::RotationType rot;
  std::cout << rot << std::endl;


  Surface::PositionType pos( 0., 0., 0.);

  Plane plane(pos,rot);

  GlobalVector g1 = plane.toGlobal(LocalVector(1., 0., 0.));
  GlobalVector g2 = plane.toGlobal(LocalVector(0., 1., 0.));
  GlobalVector g3 = plane.toGlobal(LocalVector(0., 0., 1.));
  AlgebraicMatrix33 Rsub;
  Rsub(0,0) = g1.x(); Rsub(0,1) = g2.x(); Rsub(0,2) = g3.x();
  Rsub(1,0) = g1.y(); Rsub(1,1) = g2.y(); Rsub(1,2) = g3.y();
  Rsub(2,0) = g1.z(); Rsub(2,1) = g2.z(); Rsub(2,2) = g3.z();
  
  std::cout << Rsub << std::endl;
  


  LocalTrajectoryParameters tp(1.1.,1.,0.,0.,1.);

  emd::HRTimeType s= hrRealTime();
  JacobianLocalToCartesia jl2c(plane,tp);
  emd::HRTimeType e = hrRealTime();

  std::cout << e-s << std::endl;

  std::cout << jl2c.jacobian() << std::endl;

  return 0;

}
