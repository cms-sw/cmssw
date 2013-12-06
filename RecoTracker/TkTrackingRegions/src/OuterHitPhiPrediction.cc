#include <cmath>
#include "OuterHitPhiPrediction.h"

OuterHitPhiPrediction::Range 
    OuterHitPhiPrediction::operator()(float radius) const 
{

  auto invr = 1.f/radius;
  if( std::max(std::abs(theCurvature.min()), std::abs(theCurvature.max())) > invr) 
      return Range(-M_PI,M_PI); 
  
  float Phi_r = std::asin(radius*theCurvature.max()*0.5f + theOriginRBound*invr);

  if (theCurvature.max()  == -theCurvature.min())
    return Range( thePhiAtVertex.min() - Phi_r - theTolerance.left(),
                  thePhiAtVertex.max() + Phi_r + theTolerance.right());
  
  float curv0 = theCurvature.mean();
  float Phi_0 = std::asin(radius*curv0*0.5f);
  float Phi_m = std::asin(radius*theCurvature.min()*0.5f-theOriginRBound*invr);
  return Range( thePhiAtVertex.min() + Phi_0 + Phi_m - theTolerance.left(),
		thePhiAtVertex.max() + Phi_0 + Phi_r + theTolerance.right());

}
