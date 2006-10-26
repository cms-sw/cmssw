#include "PluginManager/ModuleDef.h"

#include "FWCore/Framework/interface/MakerMacros.h"

#include "SimMuon/DTDigitizer/src/DTDigitizer.h"
#include "SimMuon/DTDigitizer/src/DTDigiReader.cc"

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(DTDigitizer);
DEFINE_ANOTHER_FWK_MODULE(DTDigiReader);

#include "SimMuon/DTDigitizer/src/DTDigiSyncFromTable.h"
#include "SimMuon/DTDigitizer/src/DTDigiSyncTOFCorr.h"

#include "SimMuon/DTDigitizer/interface/DTDigiSyncFactory.h"
DEFINE_SEAL_PLUGIN (DTDigiSyncFactory, DTDigiSyncFromTable, "DTDigiSyncFromTable");
DEFINE_SEAL_PLUGIN (DTDigiSyncFactory, DTDigiSyncTOFCorr, "DTDigiSyncTOFCorr");

