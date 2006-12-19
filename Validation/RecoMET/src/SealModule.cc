#include "PluginManager/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "Validation/RecoMET/interface/CaloTowerMETAnalyzer.h"
#include "Validation/RecoMET/interface/ECALRecHitAnalyzer.h"
#include "Validation/RecoMET/interface/HCALRecHitAnalyzer.h"
#include "Validation/RecoMET/interface/METTester.h"
#include "Validation/RecoMET/interface/DumpEvent.h"

DEFINE_SEAL_MODULE();

DEFINE_ANOTHER_FWK_MODULE (METTester) ;
DEFINE_ANOTHER_FWK_MODULE (CaloTowerMETAnalyzer) ;
DEFINE_ANOTHER_FWK_MODULE (ECALRecHitAnalyzer) ;
DEFINE_ANOTHER_FWK_MODULE (HCALRecHitAnalyzer) ;
DEFINE_ANOTHER_FWK_MODULE (DumpEvent) ;
