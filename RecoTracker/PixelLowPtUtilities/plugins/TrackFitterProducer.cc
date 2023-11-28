#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "RecoTracker/PixelTrackFitting/interface/PixelFitter.h"
#include "RecoTracker/PixelLowPtUtilities/interface/TrackFitter.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "TrackingTools/Records/interface/TransientRecHitRecord.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"

class TrackFitterProducer : public edm::global::EDProducer<> {
public:
  explicit TrackFitterProducer(const edm::ParameterSet& iConfig);
  ~TrackFitterProducer() override {}

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const override;

  const edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> geomToken_;
  const edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> magFieldToken_;
  const edm::ESGetToken<TransientTrackingRecHitBuilder, TransientRecHitRecord> ttrhToken_;
  edm::EDGetTokenT<reco::BeamSpot> theBeamSpotToken;
};

TrackFitterProducer::TrackFitterProducer(const edm::ParameterSet& iConfig)
    : geomToken_(esConsumes()),
      magFieldToken_(esConsumes()),
      ttrhToken_(esConsumes(edm::ESInputTag("", iConfig.getParameter<std::string>("TTRHBuilder")))) {
  produces<PixelFitter>();
}

void TrackFitterProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  desc.add<std::string>("TTRHBuilder", "");

  descriptions.add("trackFitter", desc);
}

void TrackFitterProducer::produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  const auto& tracker = &iSetup.getData(geomToken_);
  const auto& field = &iSetup.getData(magFieldToken_);
  const auto& ttrh = &iSetup.getData(ttrhToken_);

  auto impl = std::make_unique<TrackFitter>(tracker, field, ttrh);
  auto prod = std::make_unique<PixelFitter>(std::move(impl));
  iEvent.put(std::move(prod));
}

DEFINE_FWK_MODULE(TrackFitterProducer);
