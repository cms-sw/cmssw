#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "CaloJetTester.h"
#include "PFJetTester.h"

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE( CaloJetTester );
DEFINE_ANOTHER_FWK_MODULE( PFJetTester );
