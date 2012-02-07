#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "TauAnalysis/MCEmbeddingTools/plugins/CaloCleanerFactory.h"

#include "TauAnalysis/MCEmbeddingTools/plugins/CaloCleanerAllCrossed.h"
#include "TauAnalysis/MCEmbeddingTools/plugins/CaloCleanerMVA.h"

DEFINE_EDM_PLUGIN(CaloCleanerFactory, CaloCleanerAllCrossed, "CaloCleanerAllCrossed");
DEFINE_EDM_PLUGIN(CaloCleanerFactory, CaloCleanerMVA,        "CaloCleanerMVA");
