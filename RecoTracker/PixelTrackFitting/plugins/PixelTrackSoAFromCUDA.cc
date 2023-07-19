#include <cuda_runtime.h>
#include <Eigen/Core>  // needed here by soa layout

#include "CUDADataFormats/Common/interface/Product.h"
#include "CUDADataFormats/Common/interface/HostProduct.h"
#include "CUDADataFormats/Track/interface/TrackSoAHeterogeneousHost.h"
#include "CUDADataFormats/Track/interface/TrackSoAHeterogeneousDevice.h"
#include "CUDADataFormats/Track/interface/PixelTrackUtilities.h"
#include "DataFormats/Common/interface/Handle.h"
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

// Switch on to enable checks and printout for found tracks
// #define PIXEL_DEBUG_PRODUCE

template <typename TrackerTraits>
class PixelTrackSoAFromCUDAT : public edm::stream::EDProducer<edm::ExternalWork> {
  using TrackSoAHost = TrackSoAHeterogeneousHost<TrackerTraits>;
  using TrackSoADevice = TrackSoAHeterogeneousDevice<TrackerTraits>;

public:
  explicit PixelTrackSoAFromCUDAT(const edm::ParameterSet& iConfig);
  ~PixelTrackSoAFromCUDAT() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void acquire(edm::Event const& iEvent,
               edm::EventSetup const& iSetup,
               edm::WaitingTaskWithArenaHolder waitingTaskHolder) override;
  void produce(edm::Event& iEvent, edm::EventSetup const& iSetup) override;

  edm::EDGetTokenT<cms::cuda::Product<TrackSoADevice>> tokenCUDA_;
  edm::EDPutTokenT<TrackSoAHost> tokenSOA_;

  TrackSoAHost tracks_h_;
};

template <typename TrackerTraits>
PixelTrackSoAFromCUDAT<TrackerTraits>::PixelTrackSoAFromCUDAT(const edm::ParameterSet& iConfig)
    : tokenCUDA_(consumes(iConfig.getParameter<edm::InputTag>("src"))), tokenSOA_(produces<TrackSoAHost>()) {}

template <typename TrackerTraits>
void PixelTrackSoAFromCUDAT<TrackerTraits>::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  desc.add<edm::InputTag>("src", edm::InputTag("pixelTracksCUDA"));
  descriptions.addWithDefaultLabel(desc);
}

template <typename TrackerTraits>
void PixelTrackSoAFromCUDAT<TrackerTraits>::acquire(edm::Event const& iEvent,
                                                    edm::EventSetup const& iSetup,
                                                    edm::WaitingTaskWithArenaHolder waitingTaskHolder) {
  cms::cuda::Product<TrackSoADevice> const& inputDataWrapped = iEvent.get(tokenCUDA_);
  cms::cuda::ScopedContextAcquire ctx{inputDataWrapped, std::move(waitingTaskHolder)};
  auto const& tracks_d = ctx.get(inputDataWrapped);  // Tracks on device
  tracks_h_ = TrackSoAHost(ctx.stream());            // Create an instance of Tracks on Host, using the stream
  cudaCheck(cudaMemcpyAsync(tracks_h_.buffer().get(),
                            tracks_d.const_buffer().get(),
                            tracks_d.bufferSize(),
                            cudaMemcpyDeviceToHost,
                            ctx.stream()));  // Copy data from Device to Host
}

template <typename TrackerTraits>
void PixelTrackSoAFromCUDAT<TrackerTraits>::produce(edm::Event& iEvent, edm::EventSetup const& iSetup) {
  auto maxTracks = tracks_h_.view().metadata().size();
  auto nTracks = tracks_h_.view().nTracks();

  assert(nTracks < maxTracks);
  if (nTracks == maxTracks - 1) {
    edm::LogWarning("PixelTracks") << "Unsorted reconstructed pixel tracks truncated to " << maxTracks - 1
                                   << " candidates";
  }

#ifdef PIXEL_DEBUG_PRODUCE
  std::cout << "size of SoA " << sizeof(tsoa) << " stride " << maxTracks << std::endl;
  std::cout << "found " << nTracks << " tracks in cpu SoA at " << &tsoa << std::endl;

  int32_t nt = 0;
  for (int32_t it = 0; it < maxTracks; ++it) {
    auto nHits = TracksUtilities<TrackerTraits>::nHits(tracks_h_.view(), it);
    assert(nHits == int(tracks_h_.view().hitIndices().size(it)));
    if (nHits == 0)
      break;  // this is a guard: maybe we need to move to nTracks...
    nt++;
  }
  assert(nTracks == nt);
#endif

  // DO NOT  make a copy  (actually TWO....)
  iEvent.emplace(tokenSOA_, std::move(tracks_h_));
  assert(!tracks_h_.buffer());
}

using PixelTrackSoAFromCUDAPhase1 = PixelTrackSoAFromCUDAT<pixelTopology::Phase1>;
DEFINE_FWK_MODULE(PixelTrackSoAFromCUDAPhase1);

using PixelTrackSoAFromCUDAPhase2 = PixelTrackSoAFromCUDAT<pixelTopology::Phase2>;
DEFINE_FWK_MODULE(PixelTrackSoAFromCUDAPhase2);

using PixelTrackSoAFromCUDAHIonPhase1 = PixelTrackSoAFromCUDAT<pixelTopology::HIonPhase1>;
DEFINE_FWK_MODULE(PixelTrackSoAFromCUDAHIonPhase1);
