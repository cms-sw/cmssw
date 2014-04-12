#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "Validation/EcalClusters/interface/EgammaBasicClusters.h"
#include "Validation/EcalClusters/interface/EgammaSuperClusters.h"
#include "Validation/EcalClusters/interface/ContainmentCorrectionAnalyzer.h"



DEFINE_FWK_MODULE(EgammaBasicClusters);
DEFINE_FWK_MODULE(EgammaSuperClusters);
DEFINE_FWK_MODULE(ContainmentCorrectionAnalyzer);

