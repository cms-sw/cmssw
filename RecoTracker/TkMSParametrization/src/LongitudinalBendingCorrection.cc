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
  static float fieldInInvGev = 1./fabs(pSetup->inTesla(GlobalPoint(0,0,0)).z()  *2.99792458e-3);
  theInvCurv =  pt*fieldInInvGev;
}
