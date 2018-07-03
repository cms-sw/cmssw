#include "SimGeneral/PreMixingModule/interface/PreMixingDigiSimLinkWorker.h"
#include "SimDataFormats/GEMDigiSimLink/interface/GEMDigiSimLink.h"
#include "SimDataFormats/GEMDigiSimLink/interface/ME0DigiSimLink.h"
#include "DataFormats/Common/interface/DetSetVector.h"

using PreMixingGEMDigiSimLinkWorker = PreMixingDigiSimLinkWorker<edm::DetSetVector<GEMDigiSimLink> >;
using PreMixingME0DigiSimLinkWorker = PreMixingDigiSimLinkWorker<edm::DetSetVector<ME0DigiSimLink> >;

// register plugins
#include "SimGeneral/PreMixingModule/interface/PreMixingWorkerFactory.h"
DEFINE_EDM_PLUGIN(PreMixingWorkerFactory, PreMixingGEMDigiSimLinkWorker , "PreMixingGEMDigiSimLinkWorker");
DEFINE_EDM_PLUGIN(PreMixingWorkerFactory, PreMixingME0DigiSimLinkWorker , "PreMixingME0DigiSimLinkWorker");
