#include "TrackingTools/GsfTools/interface/KullbackLeiblerDistance.h"
#include "TrackingTools/GsfTools/interface/DistanceBetweenComponents.h"


int main() {
  typedef KullbackLeiblerDistance<5> Distance;
  typedef SingleGaussianState<5> GS;
  typedef GS::Vector Vector;
  typedef GS::Matrix Matrix;

  Distance d;

  GS gs1(Vector(1., 1.,1., 1.,1.),Matrix(ROOT::Math::SMatrixIdentity()));

  GS gs2(Vector(2., 2., 2., 2.,2.),ROOT::Math::SMatrixIdentity());

  return 0;

}
