#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
//#include "Validation/RecoMET/interface/CaloTowerAnalyzer.h"
//#include "Validation/RecoMET/interface/ECALRecHitAnalyzer.h"
//#include "Validation/RecoMET/interface/HCALRecHitAnalyzer.h"
#include "Validation/RecoMET/interface/METTester.h"
#include "Validation/RecoMET/interface/METFileSaver.h"



DEFINE_FWK_MODULE (METFileSaver) ;
DEFINE_FWK_MODULE (METTester) ;
//DEFINE_FWK_MODULE (CaloTowerAnalyzer) ;
//DEFINE_FWK_MODULE (ECALRecHitAnalyzer) ;
//DEFINE_FWK_MODULE (HCALRecHitAnalyzer) ;

