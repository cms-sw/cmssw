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

#include "CAMaskKernels.h"

// #define GPU_DEBUG

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  class CAMaskSoAProducer : public global::EDProducer<> {
    using HitsOnDevice = reco::TrackingRecHitsSoACollection;
    using MapToHit = reco::TrackingRecHitsMaskingSoACollection;

    using HitsConstView = ::reco::TrackingRecHitConstView;
    using MapToHitConstView = MapToHit::ConstView;

  public:
    explicit CAMaskSoAProducer(const edm::ParameterSet& iConfig);
    ~CAMaskSoAProducer() override = default;

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  private:
    void produce(edm::StreamID streamID, device::Event& iEvent, const device::EventSetup& iSetup) const override;

    pixelTrack::Quality const minQuality_;
    const bool useOldMask_ = false;
    const bool useHits_ = false;

    // Need one of the two input tokens: either the old mask or the hits on device
    device::EDGetToken<MapToHit> inputRecHitsMaskToken_;
    std::vector<device::EDGetToken<HitsOnDevice>> inputHitsOnDeviceToken_;

    const device::EDGetToken<reco::TracksSoACollection> inputTrackSoAToken_;

    const device::EDPutToken<reco::TrackingRecHitsMaskingSoACollection> outputRecHitsMaskToken_;
  };

  CAMaskSoAProducer::CAMaskSoAProducer(const edm::ParameterSet& iConfig)
      : EDProducer(iConfig),
        minQuality_(pixelTrack::qualityByName(iConfig.getParameter<std::string>("minQuality"))),
        useOldMask_(not iConfig.getParameter<edm::InputTag>("oldMask").label().empty()),
        useHits_(!iConfig.getParameter<std::vector<edm::InputTag>>("hitSoAs").empty()),
        inputTrackSoAToken_(consumes(iConfig.getParameter<edm::InputTag>("trackSoA"))),
        outputRecHitsMaskToken_(produces()) {
    if (minQuality_ == pixelTrack::Quality::notQuality) {
      throw cms::Exception("PixelTrackConfiguration")
          << iConfig.getParameter<std::string>("minQuality") + " is not a pixelTrack::Quality";
    }
    if (minQuality_ < pixelTrack::Quality::dup) {
      throw cms::Exception("PixelTrackConfiguration")
          << iConfig.getParameter<std::string>("minQuality") + " not supported";
    }

    if (useOldMask_) {
      inputRecHitsMaskToken_ = device::EDGetToken<MapToHit>(consumes(iConfig.getParameter<edm::InputTag>("oldMask")));
    }
    if (useHits_) {
      for (const auto& it : iConfig.getParameter<std::vector<edm::InputTag>>("hitSoAs")) {
        inputHitsOnDeviceToken_.push_back(consumes(it));
      }
    }
    if (not useOldMask_ and not useHits_) {
      throw cms::Exception("PixelTrackConfiguration") << "Either recHitsMaskSoASrc or hitsOnDeviceSrc must be provided";
    }
    if (useOldMask_ and useHits_) {
      throw cms::Exception("PixelTrackConfiguration")
          << "Only one of recHitsMaskSoASrc or hitsOnDeviceSrc should be provided";
    }
  }

  void CAMaskSoAProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;

    desc.add<edm::InputTag>("oldMask", edm::InputTag(""));
    desc.add<std::vector<edm::InputTag>>("hitSoAs", {});
    desc.add<edm::InputTag>("trackSoA",
                            edm::InputTag("pixelTracksHighPtAlpaka"));  // has to be changed for each iteration
    desc.add<std::string>("minQuality", "highPurity");

    descriptions.addWithDefaultLabel(desc);
  }

  void CAMaskSoAProducer::produce(edm::StreamID streamID, device::Event& iEvent, const device::EventSetup& es) const {
    // get both Pixel and Tracker SoA collections
    auto& queue = iEvent.queue();
    const auto& inpTkColl = iEvent.get(inputTrackSoAToken_);

    MapToHitConstView maskView;
    HitsConstView hitsView;

    int maskSize = 0;
    if (useOldMask_) {
      maskView = iEvent.get(inputRecHitsMaskToken_).view();
      maskSize = maskView.metadata().size();
    }
    if (useHits_) {
      for (const auto& token : inputHitsOnDeviceToken_) {
        hitsView = iEvent.get(token).view().trackingHits();
        maskSize += hitsView.metadata().size();
      }
    }

#ifdef GPU_DEBUG
    if (useHits_)
      printf("CAMaskSoAProducer::maskSize (inside useHits_): %d\n", maskSize);
    else if (useOldMask_)
      printf("CAMaskSoAProducer::maskSize (inside useOldMask_): %d\n", maskSize);
#endif

    reco::TrackingRecHitsMaskingSoACollection outMask(queue, maskSize);
    if (useOldMask_) {
      auto outMaskColumn = cms::alpakatools::make_device_view(queue, outMask.view().recHitMask());
      auto inMaskColumn = cms::alpakatools::make_device_view(queue, maskView.recHitMask());
      alpaka::memcpy(queue, outMaskColumn, inMaskColumn);
    } else {
      auto outMaskColumn = cms::alpakatools::make_device_view(queue, outMask.view().recHitMask());
      alpaka::memset(queue, outMaskColumn, 0);
    }

    caMasking::makeMaskingAsync(queue, outMask, inpTkColl, minQuality_);

    iEvent.emplace(outputRecHitsMaskToken_, std::move(outMask));
  }
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
DEFINE_FWK_ALPAKA_MODULE(CAMaskSoAProducer);
