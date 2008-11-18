#include "FWCore/Framework/interface/MakerMacros.h"

#include "PhysicsTools/UtilAlgos/interface/ObjectSelector.h"
#include "SimTracker/TrackHistory/plugins/BTrackingParticleSelector.h"

namespace reco{
  namespace modules {

    // define your producer name
    typedef ObjectSelector<BTrackingParticleSelector> BTrackingParticleSelection;
    
    // declare the module as plugin
    DEFINE_FWK_MODULE( BTrackingParticleSelection );
  }
}
