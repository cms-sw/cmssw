#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "SimMuon/CSCDigitizer/src/CSCDigiProducer.h"
#include "SimMuon/CSCDigitizer/src/CSCDigiDump.h"
#include "SimMuon/CSCDigitizer/src/CSCNeutronWriter.h"
#include "SimMuon/CSCDigitizer/src/CSCDigiSuppressor.h"

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(CSCDigiProducer);
DEFINE_ANOTHER_FWK_MODULE(CSCDigiDump);
DEFINE_ANOTHER_FWK_MODULE(CSCNeutronWriter);
DEFINE_ANOTHER_FWK_MODULE(CSCDigiSuppressor);
