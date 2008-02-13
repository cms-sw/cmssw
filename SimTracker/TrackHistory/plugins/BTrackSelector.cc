#include "PluginManager/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "PhysicsTools/UtilAlgos/interface/ObjectSelector.h"
#include "SimTracker/TrackHistory/plugins/BTrackSelector.h"

namespace reco{
  namespace modules {

    // define your producer name
    typedef ObjectSelector<BTrackSelector> BTrackSelection;
    
    // declare the module as plugin
    DEFINE_FWK_MODULE( BTrackSelection );
  }
}
