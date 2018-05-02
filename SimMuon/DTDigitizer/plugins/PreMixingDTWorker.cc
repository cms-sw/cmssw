#include "SimGeneral/PreMixingModule/interface/PreMixingMuonWorker.h"
#include "SimGeneral/PreMixingModule/interface/PreMixingWorkerFactory.h"
#include "DataFormats/DTDigi/interface/DTDigiCollection.h"

using PreMixingDTWorker = PreMixingMuonWorker<DTDigiCollection>;

DEFINE_EDM_PLUGIN(PreMixingWorkerFactory, PreMixingDTWorker , "PreMixingDTWorker");
