#include "SimGeneral/PreMixingModule/interface/PreMixingMuonWorker.h"
#include "SimGeneral/PreMixingModule/interface/PreMixingWorkerFactory.h"

#include "DataFormats/RPCDigi/interface/RPCDigiCollection.h"

using PreMixingRPCWorker = PreMixingMuonWorker<RPCDigiCollection>;

DEFINE_EDM_PLUGIN(PreMixingWorkerFactory, PreMixingRPCWorker, "PreMixingRPCWorker");
