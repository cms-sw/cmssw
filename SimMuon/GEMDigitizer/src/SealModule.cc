#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "SimMuon/GEMDigitizer/src/GEMDigiProducer.h"
#include "SimMuon/GEMDigitizer/src/GEMSimFactory.h"

#include "SimMuon/GEMDigitizer/src/GEMSimTriv.h"
#include "SimMuon/GEMDigitizer/src/GEMSimAverage.h"

DEFINE_FWK_MODULE(GEMDigiProducer);

DEFINE_EDM_PLUGIN(GEMSimFactory, GEMSimTriv, "GEMSimTriv");
DEFINE_EDM_PLUGIN(GEMSimFactory, GEMSimAverage,"GEMSimAverage");

#include "SimMuon/GEMDigitizer/src/GEMCSCPadDigiProducer.h"

DEFINE_FWK_MODULE(GEMCSCPadDigiProducer);

