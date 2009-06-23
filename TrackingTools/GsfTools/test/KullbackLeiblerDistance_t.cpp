#include "TrackingTools/GsfTools/interface/KullbackLeiblerDistance.h"
#include "TrackingTools/GsfTools/interface/DistanceBetweenComponents.h"

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

int main() {

  Distance const & d = distance();

  GS gs0(Vector(1., 1.,1., 0.,0.),Matrix(ROOT::Math::SMatrixIdentity()));
  GS gsP(Vector(1., 1.,1., 10.,10.),Matrix(ROOT::Math::SMatrixIdentity()));

  GS gs1(Vector(1., 1.,1., 1.,1.),Matrix(ROOT::Math::SMatrixIdentity()));

  GS gs2(Vector(2., 2., 2., 2.,2.),ROOT::Math::SMatrixIdentity());

  // make sure we load all code...
  edm::HRTimeType s0= edm::hrRealTime();
  double res = d(gs0,gsP);
  edm::HRTimeType e0 = edm::hrRealTime();
  std::cout << e0-s0 << std::endl;

  st();	
  edm::HRTimeType s= edm::hrRealTime();
  double res2 = d(gs1,gs2);
  edm::HRTimeType e = edm::hrRealTime();
  en();
  std::cout << e-s << std::endl;
 

  std:: cout << res << " " << res2 << std::endl;

  return 0;

}
