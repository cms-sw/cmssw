#include "SimGeneral/PreMixingModule/interface/PreMixingDigiSimLinkWorker.h"
#include "SimGeneral/PreMixingModule/interface/PreMixingWorkerFactory.h"
#include "SimDataFormats/TrackerDigiSimLink/interface/StripDigiSimLink.h"
#include "SimDataFormats/TrackerDigiSimLink/interface/PixelDigiSimLink.h"
#include "DataFormats/Common/interface/DetSetVector.h"

using PreMixingPixelDigiSimLinkWorker = PreMixingDigiSimLinkWorker<edm::DetSetVector<PixelDigiSimLink> >;
using PreMixingStripDigiSimLinkWorker = PreMixingDigiSimLinkWorker<edm::DetSetVector<StripDigiSimLink> >;

DEFINE_EDM_PLUGIN(PreMixingWorkerFactory, PreMixingPixelDigiSimLinkWorker , "PreMixingPixelDigiSimLinkWorker");
DEFINE_EDM_PLUGIN(PreMixingWorkerFactory, PreMixingStripDigiSimLinkWorker , "PreMixingStripDigiSimLinkWorker");
