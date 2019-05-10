#include "FWCore/PluginManager/interface/ModuleDef.h"

#include "FWCore/Framework/interface/MakerMacros.h"

#include "SimMuon/DTDigitizer/src/DTDigiReader.cc"
#include "SimMuon/DTDigitizer/src/DTDigitizer.h"

DEFINE_FWK_MODULE(DTDigitizer);
DEFINE_FWK_MODULE(DTDigiReader);

#include "SimMuon/DTDigitizer/src/DTDigiSyncFromTable.h"
#include "SimMuon/DTDigitizer/src/DTDigiSyncTOFCorr.h"

#include "SimMuon/DTDigitizer/interface/DTDigiSyncFactory.h"
DEFINE_EDM_PLUGIN(DTDigiSyncFactory, DTDigiSyncFromTable, "DTDigiSyncFromTable");
DEFINE_EDM_PLUGIN(DTDigiSyncFactory, DTDigiSyncTOFCorr, "DTDigiSyncTOFCorr");

#include "SimMuon/DTDigitizer/src/DTNeutronWriter.h"
DEFINE_FWK_MODULE(DTNeutronWriter);
