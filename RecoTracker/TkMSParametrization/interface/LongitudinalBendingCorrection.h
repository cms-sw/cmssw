#ifndef PixelRecoUtilities_LongitudinalBendingCorrection_H
#define PixelRecoUtilities_LongitudinalBendingCorrection_H
#include "RecoTracker/TkMSParametrization/interface/PixelRecoUtilities.h"


namespace pixelrecoutilities {
  class LongitudinalBendingCorrection {
  public:
    LongitudinalBendingCorrection(): coeff(0){}
    LongitudinalBendingCorrection(float pt, const edm::EventSetup& es) {
      init(pt,es);
    }
    void init(float pt, const edm::EventSetup& es) {
      auto theInvCurv =  pt*PixelRecoUtilities::fieldInInvGev(es);
      coeff = 1.f/(4.f*6.f*theInvCurv*theInvCurv);
    }
    
    inline float operator()(float radius) const {
      return  radius*radius*radius*coeff; 
    }
  private:
    float coeff;
  };
}

#endif
