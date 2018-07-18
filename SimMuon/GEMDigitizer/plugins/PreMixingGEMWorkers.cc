#include "SimGeneral/PreMixingModule/interface/PreMixingMuonWorker.h"
#include "SimGeneral/PreMixingModule/interface/PreMixingWorkerFactory.h"
#include "DataFormats/GEMDigi/interface/GEMDigiCollection.h"
#include "DataFormats/GEMDigi/interface/ME0DigiCollection.h"

using PreMixingGEMWorker = PreMixingMuonWorker<GEMDigiCollection>;
using PreMixingME0Worker = PreMixingMuonWorker<ME0DigiCollection>;

DEFINE_PREMIXING_WORKER(PreMixingGEMWorker);
DEFINE_PREMIXING_WORKER(PreMixingME0Worker);
