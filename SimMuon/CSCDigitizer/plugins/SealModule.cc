#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "SimMuon/CSCDigitizer/plugins/CSCDigiProducer.h"
#include "SimMuon/CSCDigitizer/plugins/CSCDigiDump.h"
#include "SimMuon/CSCDigitizer/plugins/CSCNeutronWriter.h"

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(CSCDigiProducer);
DEFINE_ANOTHER_FWK_MODULE(CSCDigiDump);
DEFINE_ANOTHER_FWK_MODULE(CSCNeutronWriter);

