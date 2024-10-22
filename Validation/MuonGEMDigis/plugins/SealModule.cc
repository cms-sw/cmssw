#include "Validation/MuonGEMDigis/plugins/GEMStripDigiValidation.h"
#include "Validation/MuonGEMDigis/plugins/GEMPadDigiValidation.h"
#include "Validation/MuonGEMDigis/plugins/GEMPadDigiClusterValidation.h"
#include "Validation/MuonGEMDigis/plugins/GEMCoPadDigiValidation.h"
#include "Validation/MuonGEMDigis/plugins/GEMCheckGeometry.h"
#include "Validation/MuonGEMDigis/plugins/MuonGEMDigisHarvestor.h"
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(GEMStripDigiValidation);
DEFINE_FWK_MODULE(GEMPadDigiValidation);
DEFINE_FWK_MODULE(GEMPadDigiClusterValidation);
DEFINE_FWK_MODULE(GEMCoPadDigiValidation);
DEFINE_FWK_MODULE(GEMCheckGeometry);
DEFINE_FWK_MODULE(MuonGEMDigisHarvestor);
