#include "TrackingTools/AnalyticalJacobians/interface/JacobianLocalToCartesian.h"
#include "TrackingTools/AnalyticalJacobians/interface/JacobianCartesianToLocal.h"

#include "DataFormats/GeometrySurface/interface/Surface.h"
#include "TrackingTools/TrajectoryParametrization/interface/LocalTrajectoryParameters.h"

#include "DataFormats/GeometrySurface/interface/Plane.h"

#include "MagneticField/Engine/interface/MagneticField.h"

namespace {

  struct M5T : public  MagneticField {
    virtual GlobalVector inTesla (const GlobalPoint&) const {
      return m;
    }

    GloablVector m(0.,0.,5.);
  }

}

#include "FWCore/Utilities/interface/HRRealTime.h"
int main() {


  GlobalVector xx(0.5,1.,1.);
  GlobalVector yy(-1.,0.5,1.);
  
  Surface::RotationType rot(xx,yy);
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
  

  if ( rot.xx() != Rsub(0,0) ||
       rot.xy() != Rsub(1,0) ||
       rot.xz() != Rsub(2,0) ||
       rot.yx() != Rsub(0,1) ||
       rot.yy() != Rsub(1,1) ||
       rot.yz() != Rsub(2,1) ||
       rot.zx() != Rsub(0,2) ||
       rot.zy() != Rsub(1,2) ||
       rot.zz() != Rsub(2,2) )
    std::cout << " wrong assumption!" << std::endl;


  LocalTrajectoryParameters tp(1., 1.,1., 0.,0.,1.);

  {
    edm::HRTimeType s= edm::hrRealTime();
    JacobianLocalToCartesian jl2c(plane,tp);
    edm::HRTimeType e = edm::hrRealTime();
    std::cout << e-s << std::endl;
    std::cout << jl2c.jacobian() << std::endl;
  }

  {
    M5T const m; 
    edm::HRTimeType s= edm::hrRealTime();
    JacobianLocalToCurvilinear jl2c(plane,tp,m);
    edm::HRTimeType e = edm::hrRealTime();
    std::cout << e-s << std::endl;
    std::cout << jl2c.jacobian() << std::endl;
  }


  return 0;

}
