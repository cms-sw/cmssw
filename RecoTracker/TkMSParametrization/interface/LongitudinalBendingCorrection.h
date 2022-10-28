#ifndef PixelRecoUtilities_LongitudinalBendingCorrection_H
#define PixelRecoUtilities_LongitudinalBendingCorrection_H
#include "MagneticField/Engine/interface/MagneticField.h"

namespace pixelrecoutilities {
  class LongitudinalBendingCorrection {
  public:
    LongitudinalBendingCorrection() : coeff(0) {}
    LongitudinalBendingCorrection(float pt, const MagneticField& field) { init(pt, field); }
    void init(float pt, const MagneticField& field) {
      auto theInvCurv = pt * field.inverseBzAtOriginInGeV();
      coeff = 1.f / (4.f * 6.f * theInvCurv * theInvCurv);
    }

    inline float operator()(float radius) const { return radius * radius * radius * coeff; }

  private:
    float coeff;
  };
}  // namespace pixelrecoutilities

#endif
