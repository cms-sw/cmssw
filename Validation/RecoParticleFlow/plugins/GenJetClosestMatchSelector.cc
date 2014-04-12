#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "CommonTools/UtilAlgos/interface/ObjectSelector.h"
#include "Validation/RecoParticleFlow/plugins/GenJetClosestMatchSelectorDefinition.h"

typedef ObjectSelector<GenJetClosestMatchSelectorDefinition> GenJetClosestMatchSelector;

DEFINE_FWK_MODULE(GenJetClosestMatchSelector);

