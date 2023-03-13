#include <cuda_runtime.h>

#include "CUDADataFormats/Common/interface/Product.h"
#include "CUDADataFormats/Common/interface/HostProduct.h"
#include "CUDADataFormats/Vertex/interface/ZVertexSoAHeterogeneousHost.h"
#include "CUDADataFormats/Vertex/interface/ZVertexSoAHeterogeneousDevice.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "HeterogeneousCore/CUDACore/interface/ScopedContext.h"

class PixelVertexSoAFromCUDA : public edm::stream::EDProducer<edm::ExternalWork> {
public:
  explicit PixelVertexSoAFromCUDA(const edm::ParameterSet& iConfig);
  ~PixelVertexSoAFromCUDA() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void acquire(edm::Event const& iEvent,
               edm::EventSetup const& iSetup,
               edm::WaitingTaskWithArenaHolder waitingTaskHolder) override;
  void produce(edm::Event& iEvent, edm::EventSetup const& iSetup) override;

  edm::EDGetTokenT<cms::cuda::Product<ZVertexSoADevice>> tokenCUDA_;
  edm::EDPutTokenT<ZVertexSoAHost> tokenSOA_;

  ZVertexSoAHost zvertex_h;
};

PixelVertexSoAFromCUDA::PixelVertexSoAFromCUDA(const edm::ParameterSet& iConfig)
    : tokenCUDA_(consumes<cms::cuda::Product<ZVertexSoADevice>>(iConfig.getParameter<edm::InputTag>("src"))),
      tokenSOA_(produces<ZVertexSoAHost>()) {}

void PixelVertexSoAFromCUDA::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  desc.add<edm::InputTag>("src", edm::InputTag("pixelVerticesCUDA"));
  descriptions.add("pixelVerticesSoA", desc);
}

void PixelVertexSoAFromCUDA::acquire(edm::Event const& iEvent,
                                     edm::EventSetup const& iSetup,
                                     edm::WaitingTaskWithArenaHolder waitingTaskHolder) {
  cms::cuda::Product<ZVertexSoADevice> const& inputDataWrapped = iEvent.get(tokenCUDA_);
  cms::cuda::ScopedContextAcquire ctx{inputDataWrapped, std::move(waitingTaskHolder)};
  auto const& zvertex_d = ctx.get(inputDataWrapped);  // Tracks on device
  zvertex_h = ZVertexSoAHost(ctx.stream());           // Create an instance of Tracks on Host, using the stream
  cudaCheck(cudaMemcpyAsync(zvertex_h.buffer().get(),
                            zvertex_d.const_buffer().get(),
                            zvertex_d.bufferSize(),
                            cudaMemcpyDeviceToHost,
                            ctx.stream()));  // Copy data from Device to Host
}

void PixelVertexSoAFromCUDA::produce(edm::Event& iEvent, edm::EventSetup const& iSetup) {
  // No copies....
  iEvent.emplace(tokenSOA_, std::move(zvertex_h));
}

DEFINE_FWK_MODULE(PixelVertexSoAFromCUDA);
