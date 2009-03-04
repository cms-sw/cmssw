#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "PhysicsTools/UtilAlgos/interface/ObjectSelector.h"
#include "Validation/RecoParticleFlow/plugins/GenJetSelectorDefinition.h"

typedef ObjectSelector<GenJetSelectorDefinition> GenJetSelector;

DEFINE_FWK_MODULE(GenJetSelector);

