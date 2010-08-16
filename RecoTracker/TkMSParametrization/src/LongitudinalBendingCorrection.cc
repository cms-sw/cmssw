#include "RecoTracker/TkMSParametrization/interface/LongitudinalBendingCorrection.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"

namespace {
  struct FieldAt0 {
    FieldAt0(const edm::EventSetup& es) {
      edm::ESHandle<MagneticField> pSetup;
      es.get<IdealMagneticFieldRecord>().get(pSetup);
      fieldInInvGev = 1.f/std::abs(pSetup->inTesla(GlobalPoint(0,0,0)).z()  *2.99792458e-3f);
    }
    float fieldInInvGev;
  };
}

namespace  pixelrecoutilities {
  void LongitudinalBendingCorrection::init(float pt, const edm::EventSetup& es)
  {
    static  FieldAt0 fieldAt0(es);
    theInvCurv =  pt*fieldAt0.fieldInInvGev;
    coeff = 1.f/(4.f*6.f*theInvCurv*theInvCurv);
  }
  
}
