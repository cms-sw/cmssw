#include "OuterHitPhiPrediction.h"
#include "DataFormats/Math/interface/approx_asin.h"

OuterHitPhiPrediction::Range 
    OuterHitPhiPrediction::sym(float radius) const 
{
  auto arc = radius*theCurvature.max()*0.5f + theOriginRBound/radius;
  
  auto Phi_r = unsafe_asin07<5>(arc);
  return Range( thePhiAtVertex.min() - Phi_r - theTolerance,
                thePhiAtVertex.max() + Phi_r + theTolerance);
}


// in case somebody comes with a RELEVANT use case...
OuterHitPhiPrediction::Range
    OuterHitPhiPrediction::asym(float radius) const
{

  auto invr = 1.f/radius;
  if( std::max(std::abs(theCurvature.min()), std::abs(theCurvature.max())) > invr)
      return Range(-M_PI,M_PI);

  float Phi_r = std::asin(radius*theCurvature.max()*0.5f + theOriginRBound*invr);

  if (theCurvature.max()  == -theCurvature.min())
    return Range( thePhiAtVertex.min() - Phi_r - theTolerance,
                  thePhiAtVertex.max() + Phi_r + theTolerance);
  
  float curv0 = theCurvature.mean();
  float Phi_0 = std::asin(radius*curv0*0.5f);
  float Phi_m = std::asin(radius*theCurvature.min()*0.5f-theOriginRBound*invr);
  return Range( thePhiAtVertex.min() + Phi_0 + Phi_m - theTolerance,
                thePhiAtVertex.max() + Phi_0 + Phi_r + theTolerance);

}

