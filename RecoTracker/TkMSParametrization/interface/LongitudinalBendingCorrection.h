#ifndef PixelRecoUtilities_LongitudinalBendingCorrection_H
#define PixelRecoUtilities_LongitudinalBendingCorrection_H

namespace edm {class EventSetup;}

namespace pixelrecoutilities {
class LongitudinalBendingCorrection {
public:
  LongitudinalBendingCorrection(float pt, const edm::EventSetup& es);
  float operator()(double radius) const {
    return  radius/6. * radius*radius/(2.*theInvCurv*2.*theInvCurv); 
  }
private:
  float theInvCurv;
};
}

#endif
