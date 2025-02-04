#include <alpaka/alpaka.hpp>

#include "DataFormats/TrackSoA/interface/TracksHost.h"
#include "DataFormats/TrackSoA/interface/alpaka/TracksSoACollection.h"
#include "DataFormats/TrackSoA/interface/TracksDevice.h"
#include "DataFormats/TrackingRecHitSoA/interface/alpaka/TrackingRecHitsSoACollection.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/RunningAverage.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDGetToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDPutToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/Event.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EventSetup.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/stream/EDProducer.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "RecoTracker/TkMSParametrization/interface/PixelRecoUtilities.h"
#include "RecoLocalTracker/Records/interface/PixelCPEFastParamsRecord.h"
#include "RecoLocalTracker/SiPixelRecHits/interface/alpaka/PixelCPEFastParamsCollection.h"

#include "CAHitNtupletGenerator.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  template <typename TrackerTraits>
  class CAHitNtupletAlpaka : public stream::EDProducer<> {
    using HitsConstView = TrackingRecHitSoAConstView<TrackerTraits>;
    using HitsOnDevice = TrackingRecHitsSoACollection<TrackerTraits>;
    using HitsOnHost = TrackingRecHitHost<TrackerTraits>;

    using TkSoAHost = TracksHost<TrackerTraits>;
    using TkSoADevice = TracksSoACollection<TrackerTraits>;

    using Algo = CAHitNtupletGenerator<TrackerTraits>;

  public:
    explicit CAHitNtupletAlpaka(const edm::ParameterSet& iConfig);
    ~CAHitNtupletAlpaka() override = default;
    void produce(device::Event& iEvent, const device::EventSetup& es) override;
    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  private:
    const edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> tokenField_;
    const device::ESGetToken<PixelCPEFastParams<TrackerTraits>, PixelCPEFastParamsRecord> cpeToken_;
    const device::EDGetToken<HitsOnDevice> tokenHit_;
    const device::EDPutToken<TkSoADevice> tokenTrack_;

    Algo deviceAlgo_;
  };

  template <typename TrackerTraits>
  CAHitNtupletAlpaka<TrackerTraits>::CAHitNtupletAlpaka(const edm::ParameterSet& iConfig)
      : EDProducer(iConfig),
        tokenField_(esConsumes()),
        cpeToken_(esConsumes(edm::ESInputTag("", iConfig.getParameter<std::string>("CPE")))),
        tokenHit_(consumes(iConfig.getParameter<edm::InputTag>("pixelRecHitSrc"))),
        tokenTrack_(produces()),
        deviceAlgo_(iConfig) {}

  template <typename TrackerTraits>
  void CAHitNtupletAlpaka<TrackerTraits>::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;

    desc.add<edm::InputTag>("pixelRecHitSrc", edm::InputTag("siPixelRecHitsPreSplittingAlpaka"));

    std::string cpe = "PixelCPEFastParams";
    cpe += TrackerTraits::nameModifier;
    desc.add<std::string>("CPE", cpe);

    Algo::fillPSetDescription(desc);
    descriptions.addWithDefaultLabel(desc);
  }

  template <typename TrackerTraits>
  void CAHitNtupletAlpaka<TrackerTraits>::produce(device::Event& iEvent, const device::EventSetup& es) {
    auto bf = 1. / es.getData(tokenField_).inverseBzAtOriginInGeV();

    auto& fcpe = es.getData(cpeToken_);

    auto const& hits = iEvent.get(tokenHit_);

    iEvent.emplace(tokenTrack_, deviceAlgo_.makeTuplesAsync(hits, fcpe.const_buffer().data(), bf, iEvent.queue()));
  }

  using CAHitNtupletAlpakaPhase1 = CAHitNtupletAlpaka<pixelTopology::Phase1>;
  using CAHitNtupletAlpakaHIonPhase1 = CAHitNtupletAlpaka<pixelTopology::HIonPhase1>;
  using CAHitNtupletAlpakaPhase2 = CAHitNtupletAlpaka<pixelTopology::Phase2>;
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"

DEFINE_FWK_ALPAKA_MODULE(CAHitNtupletAlpakaPhase1);
DEFINE_FWK_ALPAKA_MODULE(CAHitNtupletAlpakaHIonPhase1);
DEFINE_FWK_ALPAKA_MODULE(CAHitNtupletAlpakaPhase2);
