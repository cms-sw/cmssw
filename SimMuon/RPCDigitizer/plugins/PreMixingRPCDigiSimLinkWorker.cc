#include "SimGeneral/PreMixingModule/interface/PreMixingDigiSimLinkWorker.h"
#include "SimGeneral/PreMixingModule/interface/PreMixingWorkerFactory.h"
#include "SimDataFormats/RPCDigiSimLink/interface/RPCDigiSimLink.h"
#include "DataFormats/Common/interface/DetSetVector.h"

using PreMixingRPCDigiSimLinkWorker = PreMixingDigiSimLinkWorker<edm::DetSetVector<RPCDigiSimLink> >;

DEFINE_PREMIXING_WORKER(PreMixingRPCDigiSimLinkWorker);
