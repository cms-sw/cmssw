#include "FWCore/Framework/interface/MakerMacros.h"
#include "CommonTools/UtilAlgos/interface/ObjectSelectorStream.h"
#include "Validation/RecoHI/plugins/HitPixelLayersTPSelector.h"

namespace reco {
  namespace modules {

    // define your producer name
    typedef ObjectSelectorStream<HitPixelLayersTPSelector, TrackingParticleRefVector> HitPixelLayersTPSelection;

    // declare the module as plugin
    DEFINE_FWK_MODULE(HitPixelLayersTPSelection);
  }  // namespace modules
}  // namespace reco
