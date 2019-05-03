#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "RecoTracker/TkMSParametrization/interface/PixelRecoUtilities.h"

namespace PixelRecoUtilities {
FieldAt0::FieldAt0(const edm::EventSetup &es) {
  edm::ESHandle<MagneticField> pSetup;
  es.get<IdealMagneticFieldRecord>().get(pSetup);
  fieldInInvGev = 1.f / std::abs(pSetup->inTesla(GlobalPoint(0, 0, 0)).z() *
                                 2.99792458e-3f);
}

} // namespace PixelRecoUtilities
