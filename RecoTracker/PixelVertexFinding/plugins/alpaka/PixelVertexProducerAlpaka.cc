#include <alpaka/alpaka.hpp>

#include "Geometry/CommonTopologies/interface/SimplePixelTopology.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Utilities/interface/StreamID.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "HeterogeneousCore/AlpakaCore/interface/module_backend_config.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDPutToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ESGetToken.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/Event.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EventSetup.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/global/EDProducer.h"

#include "DataFormats/TrackSoA/interface/alpaka/TrackSoACollection.h"
#include "DataFormats/TrackSoA/interface/TrackSoADevice.h"
#include "DataFormats/Vertex/interface/alpaka/ZVertexSoACollection.h"
#include "DataFormats/Vertex/interface/ZVertexSoADevice.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"

#include "vertexFinder.h"

#undef PIXVERTEX_DEBUG_PRODUCE

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  using namespace cms::alpakatools;

  template <typename TrackerTraits>
  class PixelVertexProducerAlpaka : public global::EDProducer<> {
    using TkSoADevice = TrackSoACollection<TrackerTraits>;
    using GPUAlgo = vertexFinder::Producer<TrackerTraits>;

  public:
    explicit PixelVertexProducerAlpaka(const edm::ParameterSet& iConfig);
    ~PixelVertexProducerAlpaka() override = default;

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  private:
    void produceOnGPU(edm::StreamID streamID,  // maybe even remove this and leave only produce?
                      device::Event& iEvent,
                      const device::EventSetup& iSetup) const;
    void produce(edm::StreamID streamID, device::Event& iEvent, const device::EventSetup& iSetup) const override;

    device::EDGetToken<TkSoADevice> tokenDeviceTrack_;
    device::EDPutToken<ZVertexCollection> tokenDeviceVertex_;

    const GPUAlgo gpuAlgo_;

    // Tracking cuts before sending tracks to vertex algo
    const float ptMin_;
    const float ptMax_;
  };

  template <typename TrackerTraits>
  PixelVertexProducerAlpaka<TrackerTraits>::PixelVertexProducerAlpaka(const edm::ParameterSet& conf)
      : gpuAlgo_(conf.getParameter<bool>("oneKernel"),
                 conf.getParameter<bool>("useDensity"),
                 conf.getParameter<bool>("useDBSCAN"),
                 conf.getParameter<bool>("useIterative"),
                 conf.getParameter<bool>("doSplitting"),
                 conf.getParameter<int>("minT"),
                 conf.getParameter<double>("eps"),
                 conf.getParameter<double>("errmax"),
                 conf.getParameter<double>("chi2max")),
        ptMin_(conf.getParameter<double>("PtMin")),  // 0.5 GeV
        ptMax_(conf.getParameter<double>("PtMax"))   // 75. GeV
  {
    tokenDeviceTrack_ = consumes(conf.getParameter<edm::InputTag>("pixelTrackSrc"));
    tokenDeviceVertex_ = produces();
  }

  template <typename TrackerTraits>
  void PixelVertexProducerAlpaka<TrackerTraits>::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;

    // Only one of these three algos can be used at once.
    // Maybe this should become a Plugin Factory
    desc.add<bool>("oneKernel", true);
    desc.add<bool>("useDensity", true);
    desc.add<bool>("useDBSCAN", false);
    desc.add<bool>("useIterative", false);
    desc.add<bool>("doSplitting", true);

    desc.add<int>("minT", 2);          // min number of neighbours to be "core"
    desc.add<double>("eps", 0.07);     // max absolute distance to cluster
    desc.add<double>("errmax", 0.01);  // max error to be "seed"
    desc.add<double>("chi2max", 9.);   // max normalized distance to cluster

    desc.add<double>("PtMin", 0.5);
    desc.add<double>("PtMax", 75.);
    desc.add<edm::InputTag>("pixelTrackSrc", edm::InputTag("pixelTracksAlpaka"));

    descriptions.addWithDefaultLabel(desc);
  }

  template <typename TrackerTraits>
  void PixelVertexProducerAlpaka<TrackerTraits>::produceOnGPU(edm::StreamID streamID,
                                                              device::Event& iEvent,
                                                              const device::EventSetup& iSetup) const {
    auto const& hTracks = iEvent.get(tokenDeviceTrack_);

    iEvent.emplace(tokenDeviceVertex_, gpuAlgo_.makeAsync(iEvent.queue(), hTracks.view(), ptMin_, ptMax_));
  }

  template <typename TrackerTraits>
  void PixelVertexProducerAlpaka<TrackerTraits>::produce(edm::StreamID streamID,
                                                         device::Event& iEvent,
                                                         const device::EventSetup& iSetup) const {
    produceOnGPU(streamID, iEvent, iSetup);
  }

  using PixelVertexProducerAlpakaPhase1 = PixelVertexProducerAlpaka<pixelTopology::Phase1>;
  using PixelVertexProducerAlpakaPhase2 = PixelVertexProducerAlpaka<pixelTopology::Phase2>;
  using PixelVertexProducerAlpakaHIonPhase1 = PixelVertexProducerAlpaka<pixelTopology::HIonPhase1>;

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

DEFINE_FWK_ALPAKA_MODULE(PixelVertexProducerAlpakaPhase1);
DEFINE_FWK_ALPAKA_MODULE(PixelVertexProducerAlpakaPhase2);
DEFINE_FWK_ALPAKA_MODULE(PixelVertexProducerAlpakaHIonPhase1);
