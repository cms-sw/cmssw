#include "RecoTracker/TkMSParametrization/interface/LongitudinalBendingCorrection.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"

#include "RecoTracker/TkMSParametrization/interface/PixelRecoUtilities.h"

namespace  pixelrecoutilities {
  void LongitudinalBendingCorrection::init(float pt, const edm::EventSetup& es)
  {
    theInvCurv =  pt*PixelRecoUtilities::fieldInInvGev(es);
    coeff = 1.f/(4.f*6.f*theInvCurv*theInvCurv);
  }
  
}
