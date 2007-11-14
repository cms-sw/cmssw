#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "Validation/RecoJets/interface/CaloJetTester.h"
#include "Validation/RecoJets/src/PFJetTester.h"

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE( CaloJetTester );
DEFINE_ANOTHER_FWK_MODULE( PFJetTester );
