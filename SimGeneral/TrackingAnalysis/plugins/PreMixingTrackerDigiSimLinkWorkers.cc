#include "DataFormats/Common/interface/DetSetVector.h"
#include "SimDataFormats/TrackerDigiSimLink/interface/PixelDigiSimLink.h"
#include "SimDataFormats/TrackerDigiSimLink/interface/StripDigiSimLink.h"
#include "SimGeneral/PreMixingModule/interface/PreMixingDigiSimLinkWorker.h"
#include "SimGeneral/PreMixingModule/interface/PreMixingWorkerFactory.h"

using PreMixingPixelDigiSimLinkWorker = PreMixingDigiSimLinkWorker<edm::DetSetVector<PixelDigiSimLink>>;
using PreMixingStripDigiSimLinkWorker = PreMixingDigiSimLinkWorker<edm::DetSetVector<StripDigiSimLink>>;

DEFINE_PREMIXING_WORKER(PreMixingPixelDigiSimLinkWorker);
DEFINE_PREMIXING_WORKER(PreMixingStripDigiSimLinkWorker);
