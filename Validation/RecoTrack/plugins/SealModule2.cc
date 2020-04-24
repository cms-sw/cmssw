#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "CommonTools/UtilAlgos/interface/StringCutObjectSelector.h"
#include "CommonTools/UtilAlgos/interface/SingleObjectSelector.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

typedef SingleObjectSelector<
            std::vector<reco::Track>,
            StringCutObjectSelector<reco::Track>
        > TrackSelectorForValidation;

DEFINE_FWK_MODULE(TrackSelectorForValidation);
