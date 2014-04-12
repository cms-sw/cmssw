#include "FWCore/Framework/interface/MakerMacros.h"
#include "CommonTools/UtilAlgos/interface/ObjectSelector.h"
#include "Validation/RecoHI/plugins/HitPixelLayersTPSelector.h"

namespace reco
{
  namespace modules
  {
    
    // define your producer name
    typedef ObjectSelector<HitPixelLayersTPSelector> HitPixelLayersTPSelection;
    
    // declare the module as plugin
    DEFINE_FWK_MODULE( HitPixelLayersTPSelection );
  }
}

