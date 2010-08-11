#ifndef PixelRecoUtilities_LongitudinalBendingCorrection_H
#define PixelRecoUtilities_LongitudinalBendingCorrection_H

namespace edm {class EventSetup;}

namespace pixelrecoutilities {
class LongitudinalBendingCorrection {
public:
  LongitudinalBendingCorrection(float pt, const edm::EventSetup& es);
  inline float operator()(double radius) const {
    return  radius*radius*radius*coeff; 
  }
private:
  float theInvCurv;
  float coeff;
};
}

#endif
