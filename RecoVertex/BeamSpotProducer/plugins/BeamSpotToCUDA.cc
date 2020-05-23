#include "CUDADataFormats/Common/interface/Product.h"
#include "CUDADataFormats/BeamSpot/interface/BeamSpotCUDA.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "HeterogeneousCore/CUDACore/interface/ScopedContext.h"
#include "HeterogeneousCore/CUDAServices/interface/CUDAService.h"
#include "HeterogeneousCore/CUDAUtilities/interface/host_noncached_unique_ptr.h"

#include <cuda_runtime.h>

namespace {
  class BSHost {
  public:
    BSHost() : bs{cms::cuda::make_host_noncached_unique<BeamSpotCUDA::Data>(cudaHostAllocWriteCombined)} {}
    BeamSpotCUDA::Data* get() { return bs.get(); }

  private:
    cms::cuda::host::noncached::unique_ptr<BeamSpotCUDA::Data> bs;
  };
}  // namespace

class BeamSpotToCUDA : public edm::global::EDProducer<edm::StreamCache<BSHost>> {
public:
  explicit BeamSpotToCUDA(const edm::ParameterSet& iConfig);
  ~BeamSpotToCUDA() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  std::unique_ptr<BSHost> beginStream(edm::StreamID) const override {
    edm::Service<CUDAService> cs;
    if (cs->enabled()) {
      return std::make_unique<BSHost>();
    } else {
      return nullptr;
    }
  }
  void produce(edm::StreamID streamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const override;

private:
  edm::EDGetTokenT<reco::BeamSpot> bsGetToken_;
  edm::EDPutTokenT<cms::cuda::Product<BeamSpotCUDA>> bsPutToken_;
};

BeamSpotToCUDA::BeamSpotToCUDA(const edm::ParameterSet& iConfig)
    : bsGetToken_{consumes<reco::BeamSpot>(iConfig.getParameter<edm::InputTag>("src"))},
      bsPutToken_{produces<cms::cuda::Product<BeamSpotCUDA>>()} {}

void BeamSpotToCUDA::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("src", edm::InputTag("offlineBeamSpot"));
  descriptions.add("offlineBeamSpotCUDA", desc);
}

void BeamSpotToCUDA::produce(edm::StreamID streamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  cms::cuda::ScopedContextProduce ctx{streamID};

  const reco::BeamSpot& bs = iEvent.get(bsGetToken_);

  BeamSpotCUDA::Data* bsHost = streamCache(streamID)->get();

  bsHost->x = bs.x0();
  bsHost->y = bs.y0();
  bsHost->z = bs.z0();

  bsHost->sigmaZ = bs.sigmaZ();
  bsHost->beamWidthX = bs.BeamWidthX();
  bsHost->beamWidthY = bs.BeamWidthY();
  bsHost->dxdz = bs.dxdz();
  bsHost->dydz = bs.dydz();
  bsHost->emittanceX = bs.emittanceX();
  bsHost->emittanceY = bs.emittanceY();
  bsHost->betaStar = bs.betaStar();

  ctx.emplace(iEvent, bsPutToken_, bsHost, ctx.stream());
}

DEFINE_FWK_MODULE(BeamSpotToCUDA);
