#include "RecoTracker/TkMSParametrization/interface/LongitudinalBendingCorrection.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"

using pixelrecoutilities::LongitudinalBendingCorrection;
LongitudinalBendingCorrection::LongitudinalBendingCorrection(float pt, const edm::EventSetup& es)
{
  edm::ESHandle<MagneticField> pSetup;
  es.get<IdealMagneticFieldRecord>().get(pSetup);
  float fieldInInvGev = 1.f/std::abs(pSetup->inTesla(GlobalPoint(0,0,0)).z()  *2.99792458e-3f);
  theInvCurv =  pt*fieldInInvGev;
  coeff = 1.f/(4.f*6.f*theInvCurv*theInvCurv);
}
