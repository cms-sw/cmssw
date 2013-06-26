#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "PFJetTester.h"
#include "PFJetTesterUnCorr.h"
#include "CaloJetTester.h"
#include "CaloJetTesterUnCorr.h"
#include "JPTJetTester.h"
#include "JPTJetTesterUnCorr.h"
#include "JetFileSaver.h"

DEFINE_FWK_MODULE( PFJetTester );
DEFINE_FWK_MODULE( CaloJetTester );
DEFINE_FWK_MODULE( JPTJetTester );
DEFINE_FWK_MODULE( PFJetTesterUnCorr );
DEFINE_FWK_MODULE( CaloJetTesterUnCorr );
DEFINE_FWK_MODULE( JPTJetTesterUnCorr );
DEFINE_FWK_MODULE( JetFileSaver );
