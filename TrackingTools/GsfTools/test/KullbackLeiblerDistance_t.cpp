#include "TrackingTools/GsfTools/interface/KullbackLeiblerDistance.h"
#include "TrackingTools/GsfTools/interface/DistanceBetweenComponents.h"

#include "FWCore/Utilities/interface/HRRealTime.h"
#include<iostream>

int main() {
  typedef KullbackLeiblerDistance<5> Distance;
  typedef SingleGaussianState<5> GS;
  typedef GS::Vector Vector;
  typedef GS::Matrix Matrix;

  Distance d;

  GS gs0(Vector(1., 1.,1., 0.,0.),Matrix(ROOT::Math::SMatrixIdentity()));
  GS gsP(Vector(1., 1.,1., 10.,10.),Matrix(ROOT::Math::SMatrixIdentity()));

  GS gs1(Vector(1., 1.,1., 1.,1.),Matrix(ROOT::Math::SMatrixIdentity()));

  GS gs2(Vector(2., 2., 2., 2.,2.),ROOT::Math::SMatrixIdentity());

  // make sure we load all code...
  double res = d(gs0,gsP);


  edm::HRTimeType s= edm::hrRealTime();
  double res2 = d(gs1,gs2);
  edm::HRTimeType e = edm::hrRealTime();
  std::cout << e-s << std::endl;
 

  std:: cout << res << " " << res2 << std::endl;

  return 0;

}
