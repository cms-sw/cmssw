#include "TrackingTools/GeomPropagators/interface/HelixForwardPlaneCrossing.h"
#include "TrackingTools/AnalyticalJacobians/interface/AnalyticalCurvilinearJacobian.h"
#include "TrackingTools/TrajectoryParametrization/interface/GlobalTrajectoryParameters.h"



#include "DataFormats/GeometrySurface/interface/Surface.h"
#include "TrackingTools/TrajectoryParametrization/interface/LocalTrajectoryParameters.h"

#include "DataFormats/GeometrySurface/interface/Plane.h"

#include "MagneticField/Engine/interface/MagneticField.h"

namespace {

  struct M5T : public  MagneticField {
    M5T() :  m(0.,0.,5.){}
    virtual GlobalVector inTesla (const GlobalPoint&) const {
      return m;
    }

    GlobalVector m;
  };

}

#include "FWCore/Utilities/interface/HRRealTime.h"
void st(){}
void en(){}

#include<iostream>

int main() {

  M5T const m;

  // GlobalVector xx(0.5,1.,1.);
  // GlobalVector yy(-1.,0.5,1.);

  Basic3DVector<float>  axis(0.5,1.,1);
  
  Surface::RotationType rot(axis,0.5*M_PI);
  std::cout << rot << std::endl;

  Surface::PositionType pos( 0., 0., 0.);

  Plane plane(pos,rot);

  LocalTrajectoryParameters tpl(1., 1.,1., 0.,0.,1.);
  GlobalVector mg = plane.toGlobal(tpl.momentum());
  GlobalTrajectoryParameters tpg(pos,mg,1., &m);
  std::cout << tpl.position() << " " << tpl.momentum() << std::endl;
  std::cout << tpg.position() << " " << tpg.momentum() << std::endl;

  double curv =   tpg.transverseCurvature();

  AlgebraicMatrix55 fullJacobian(AlgebraicMatrixID());
  AlgebraicMatrix55 deltaJacobian(AlgebraicMatrixID());
  GlobalTrajectoryParameters tpg0(pos,mg,1., &m);
  for (int i=0; i<10;++i) {
    HelixForwardPlaneCrossing prop(HelixForwardPlaneCrossing::PositionType(tpg.position()), 
                                 HelixForwardPlaneCrossing::DirectionType(tpg.momentum()), curv);
    GlobalVector h = tpg.magneticFieldInInverseGeV(tpg.position());
    double s = 0.1;
    GlobalPoint x(prop.position(s));
    GlobalVector p(prop.direction(s));
    GlobalTrajectoryParameters tpg2(x,p, curv, 0, &m);
    std::cout << tpg2.position() << " " << tpg2.momentum() << std::endl;
    AnalyticalCurvilinearJacobian full;
    AnalyticalCurvilinearJacobian delta;
    full.computeFullJacobian(tpg,x,p,h,s);
    std::cout <<  full.jacobian() << std::endl;
    std::cout << std::endl;
    delta.computeInfinitesimalJacobian(tpg,x,p,h,s);
    std::cout << delta.jacobian() << std::endl;
    std::cout << std::endl;
    tpg = tpg2;
    fullJacobian *=full.jacobian();
    deltaJacobian *=delta.jacobian();
  }
  std::cout << std::endl;
  std::cout << std::endl;
  std::cout <<  fullJacobian << std::endl;
  std::cout << std::endl;
  std::cout << deltaJacobian << std::endl;
  std::cout << std::endl;
  AnalyticalCurvilinearJacobian full;
  full.computeFullJacobian(tpg0,tpg.position(),tpg.momentum(),h,1.);
  std::cout <<  full.jacobian() << std::endl;
  std::cout << std::endl;



  return 0;
}
