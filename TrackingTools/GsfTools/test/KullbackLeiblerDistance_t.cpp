#include "TrackingTools/GsfTools/interface/KullbackLeiblerDistance.h"
#include "TrackingTools/GsfTools/interface/DistanceBetweenComponents.h"

#include "TrackingTools/AnalyticalJacobians/interface/JacobianLocalToCartesian.h"
#include "TrackingTools/AnalyticalJacobians/interface/JacobianCartesianToLocal.h"
#include "TrackingTools/AnalyticalJacobians/interface/JacobianLocalToCurvilinear.h"


#include "DataFormats/GeometrySurface/interface/Surface.h"
#include "TrackingTools/TrajectoryParametrization/interface/LocalTrajectoryParameters.h"

#include "DataFormats/GeometrySurface/interface/Plane.h"


#include "FWCore/Utilities/interface/HRRealTime.h"
#include<iostream>

void st(){}
void en(){}


  typedef DistanceBetweenComponents<5> Distance; 
  typedef KullbackLeiblerDistance<5> KDistance;
  typedef SingleGaussianState<5> GS;
  typedef GS::Vector Vector;
  typedef GS::Matrix Matrix;

  Distance const & distance() {
      static Distance * d = new KDistance;
      return *d;
  }

Matrix buildCovariance() {

  // build a resonable covariance matrix as JIJ

  Basic3DVector<float>  axis(0.5,1.,1);
  
  Surface::RotationType rot(axis,0.5*M_PI);

  Surface::PositionType pos( 0., 0., 0.);

  Plane plane(pos,rot);
  JacobianLocalToCartesian jl2c(plane,tp);
  return  jl2c.jacobian().transpose()* jl2c.jacobian();

}

int main() {

  Distance const & d = distance();

  Matrix cov = buildCovariance();

  GS * gs1 = new GS(Vector(1., 1.,1., 1.,1.),Matrix(ROOT::Math::SMatrixIdentity()));

  GS * gs0 = new GS(Vector(1., 1.,1., 0.,0.),Matrix(ROOT::Math::SMatrixIdentity()));
  GS * gsP = new GS(Vector(1., 1.,1., 10.,10.),Matrix(ROOT::Math::SMatrixIdentity()));


  GS * gs2 = new GS(Vector(2., 2., 2., 2.,2.),cov);

  // make sure we load all code...
  edm::HRTimeType s0= edm::hrRealTime();
  double res = d(*gs0,*gsP);
  edm::HRTimeType e0 = edm::hrRealTime();
  std::cout << e0-s0 << std::endl;

  st();	
  edm::HRTimeType s= edm::hrRealTime();
  double res2 = d(*gs1,*gs2);
  edm::HRTimeType e = edm::hrRealTime();
  en();
  std::cout << e-s << std::endl;
 

  std:: cout << res << " " << res2 << std::endl;

  return 0;

}
