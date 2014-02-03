#include <cmath>
#include "RecoTracker/TkTrackingRegions/interface/OuterHitPhiPrediction.h"

OuterHitPhiPrediction::Range 
    OuterHitPhiPrediction::operator()(float radius) const 
{

  if( std::max(fabs(theCurvature.min()), fabs(theCurvature.max())) > 1./radius) 
      return Range(-M_PI,M_PI); 
  
  float Phi_r = asin(radius*theCurvature.max()/2 + theOriginRBound/radius);
  float curv0 = theCurvature.mean();

  if (curv0 == 0.) {
    return Range( thePhiAtVertex.min() - Phi_r - theTolerance.left(),
                  thePhiAtVertex.max() + Phi_r + theTolerance.right());
  } 
  else {
    float Phi_0 = asin(radius*curv0/2);
    float Phi_m = asin(radius*theCurvature.min()/2-theOriginRBound/radius);
    return Range( thePhiAtVertex.min() + Phi_0 + Phi_m - theTolerance.left(),
                  thePhiAtVertex.max() + Phi_0 + Phi_r + theTolerance.right());
  } 
}
