#include "SimGeneral/PreMixingModule/interface/PreMixingMuonWorker.h"
#include "SimGeneral/PreMixingModule/interface/PreMixingWorkerFactory.h"
#include "DataFormats/GEMDigi/interface/GEMDigiCollection.h"
#include "DataFormats/GEMDigi/interface/ME0DigiCollection.h"

using PreMixingGEMWorker = PreMixingMuonWorker<GEMDigiCollection>;
using PreMixingME0Worker = PreMixingMuonWorker<ME0DigiCollection>;

DEFINE_EDM_PLUGIN(PreMixingWorkerFactory, PreMixingGEMWorker, "PreMixingGEMWorker");
DEFINE_EDM_PLUGIN(PreMixingWorkerFactory, PreMixingME0Worker, "PreMixingME0Worker");
