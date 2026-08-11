#include <alpaka/alpaka.hpp>

#include "DataFormats/TrackSoA/interface/TracksHost.h"
#include "DataFormats/TrackSoA/interface/alpaka/TracksSoACollection.h"
#include "DataFormats/TrackSoA/interface/TracksDevice.h"
#include "DataFormats/TrackingRecHitSoA/interface/alpaka/TrackingRecHitsSoACollection.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/global/EDProducer.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDGetToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDPutToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/Event.h"

#include "CAHitNtupletGenerator.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  class PixelTracksMaskingSoA : public global::EDProducer<> {
    using Algo = CAHitMaskingAndMerger;

  public:
    explicit PixelTracksMaskingSoA(const edm::ParameterSet& iConfig);
    ~PixelTracksMaskingSoA() override = default;

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  private:
    void produce(edm::StreamID streamID, device::Event& iEvent, const device::EventSetup& iSetup) const override;

    uint32_t const iterationIndex_;
    pixelTrack::Quality const minQuality_;

    const device::EDGetToken<reco::TrackingRecHitsMaskingCollection> inputRecHitsMaskToken_;
    const device::EDGetToken<reco::TracksSoACollection> inputTrackSoAToken_;

    const device::EDPutToken<reco::TrackingRecHitsMaskingCollection> outputRecHitsMaskToken_;

    Algo deviceAlgo_;
  };

  PixelTracksMaskingSoA::PixelTracksMaskingSoA(const edm::ParameterSet& iConfig)
      : EDProducer(iConfig),
        iterationIndex_(iConfig.getParameter<uint32_t>("iterationIndex")),
        minQuality_(pixelTrack::qualityByName(iConfig.getParameter<std::string>("minQuality"))),
        inputRecHitsMaskToken_(consumes(iConfig.getParameter<edm::InputTag>("recHitsMaskSoASrc"))),
        inputTrackSoAToken_(consumes(iConfig.getParameter<edm::InputTag>("tracksSoASrc"))),
        outputRecHitsMaskToken_(produces()) {
    if (minQuality_ == pixelTrack::Quality::notQuality) {
      throw cms::Exception("PixelTrackConfiguration")
          << iConfig.getParameter<std::string>("minQuality") + " is not a pixelTrack::Quality";
    }
    if (minQuality_ < pixelTrack::Quality::dup) {
      throw cms::Exception("PixelTrackConfiguration")
          << iConfig.getParameter<std::string>("minQuality") + " not supported";
    }
  }

  void PixelTracksMaskingSoA::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;

    desc.add<edm::InputTag>(
        "recHitsMaskSoASrc",
        edm::InputTag("siPixelRecHitsExtendedPreSplittingAlpaka"));  // has to be changed for each iteration
    desc.add<edm::InputTag>("tracksSoASrc",
                            edm::InputTag("pixelTracksHighPtAlpaka"));  // has to be changed for each iteration
    desc.add<std::string>("minQuality", "highPurity");
    desc.add<uint32_t>("iterationIndex", 0);

    descriptions.addWithDefaultLabel(desc);
  }

  void PixelTracksMaskingSoA::produce(edm::StreamID streamID,
                                      device::Event& iEvent,
                                      const device::EventSetup& es) const {
    // get both Pixel and Tracker SoA collections
    auto queue = iEvent.queue();
    const auto& inpMaskColl = iEvent.get(inputRecHitsMaskToken_);
    const auto& inpTkColl = iEvent.get(inputTrackSoAToken_);

    iEvent.emplace(outputRecHitsMaskToken_,
                   deviceAlgo_.makeMaskingAsync(inpMaskColl, inpTkColl, minQuality_, iterationIndex_, iEvent.queue()));
  }
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
DEFINE_FWK_ALPAKA_MODULE(PixelTracksMaskingSoA);
