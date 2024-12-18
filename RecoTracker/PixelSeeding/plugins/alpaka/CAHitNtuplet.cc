#include <alpaka/alpaka.hpp>

#include <TFormula.h>

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

#include "RecoTracker/Record/interface/TrackerRecoGeometryRecord.h"
#include "RecoTracker/PixelSeeding/interface/alpaka/CAGeometrySoACollection.h"

#include "CAHitNtupletGenerator.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  template <typename TrackerTraits>
  class CAHitNtupletAlpaka : public stream::EDProducer<> {
    using HitsConstView = ::reco::TrackingRecHitConstView;
    using HitsOnDevice = reco::TrackingRecHitsSoACollection;
    using HitsOnHost = ::reco::TrackingRecHitHost;

    using TkSoAHost = ::reco::TracksHost;
    using TkSoADevice = reco::TracksSoACollection;

    using Algo = CAHitNtupletGenerator<TrackerTraits>;

  public:
    explicit CAHitNtupletAlpaka(const edm::ParameterSet& iConfig);
    ~CAHitNtupletAlpaka() override = default;
    void produce(device::Event& iEvent, const device::EventSetup& es) override;
    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  private:
    const edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> tokenField_;
    const device::ESGetToken<reco::CAGeometrySoACollection, TrackerRecoGeometryRecord> geometrySoA_;  
    const device::EDGetToken<HitsOnDevice> tokenHit_;
    const device::EDPutToken<TkSoADevice> tokenTrack_;

    const TFormula maxNumberOfDoublets_;
    const TFormula maxNumberOfTuples_;
    Algo deviceAlgo_;
  };

  template <typename TrackerTraits>
  CAHitNtupletAlpaka<TrackerTraits>::CAHitNtupletAlpaka(const edm::ParameterSet& iConfig)
      : tokenField_(esConsumes()),
        geometrySoA_(esConsumes(edm::ESInputTag("", iConfig.getParameter<std::string>("caGeometry")))),
        tokenHit_(consumes(iConfig.getParameter<edm::InputTag>("pixelRecHitSrc"))),
        tokenTrack_(produces()),
        maxNumberOfDoublets_(TFormula("doubletsHitsDependecy", iConfig.getParameter<std::string>("maxNumberOfDoublets").data())),
        maxNumberOfTuples_(TFormula("tracksHitsDependency",iConfig.getParameter<std::string>("maxNumberOfTuples").data())),
        deviceAlgo_(iConfig) {

        }

  template <typename TrackerTraits>
  void CAHitNtupletAlpaka<TrackerTraits>::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;

    desc.add<edm::InputTag>("pixelRecHitSrc", edm::InputTag("siPixelRecHitsPreSplittingAlpaka"));
    desc.add<std::string>("caGeometry", std::string("caGeometry"));

    Algo::fillPSetDescription(desc);
    descriptions.addWithDefaultLabel(desc);
  }

  template <typename TrackerTraits>
  void CAHitNtupletAlpaka<TrackerTraits>::produce(device::Event& iEvent, const device::EventSetup& es) {
    auto bf = 1. / es.getData(tokenField_).inverseBzAtOriginInGeV();

    auto const& geometry = es.getData(geometrySoA_);

    auto const& hits = iEvent.get(tokenHit_);

    uint32_t const maxTuples = maxNumberOfTuples_.Eval(hits.nHits());
    uint32_t const maxDoublets = maxNumberOfDoublets_.Eval(hits.nHits());
    // std::cout << "maxDoublets " << maxDoublets << std::endl;
    iEvent.emplace(tokenTrack_, deviceAlgo_.makeTuplesAsync(hits, geometry, bf, maxDoublets, maxTuples, iEvent.queue()));
  }

  using CAHitNtupletAlpakaPhase1 = CAHitNtupletAlpaka<pixelTopology::Phase1>;
  using CAHitNtupletAlpakaHIonPhase1 = CAHitNtupletAlpaka<pixelTopology::HIonPhase1>;
  using CAHitNtupletAlpakaPhase2 = CAHitNtupletAlpaka<pixelTopology::Phase2>;
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"

DEFINE_FWK_ALPAKA_MODULE(CAHitNtupletAlpakaPhase1);
DEFINE_FWK_ALPAKA_MODULE(CAHitNtupletAlpakaHIonPhase1);
DEFINE_FWK_ALPAKA_MODULE(CAHitNtupletAlpakaPhase2);
