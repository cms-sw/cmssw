#include "DataFormats/DTDigi/interface/DTDigiCollection.h"
#include "SimGeneral/PreMixingModule/interface/PreMixingMuonWorker.h"
#include "SimGeneral/PreMixingModule/interface/PreMixingWorkerFactory.h"

using PreMixingDTWorker = PreMixingMuonWorker<DTDigiCollection>;

DEFINE_PREMIXING_WORKER(PreMixingDTWorker);
