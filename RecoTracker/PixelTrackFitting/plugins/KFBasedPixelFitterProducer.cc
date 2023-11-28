#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "RecoTracker/PixelTrackFitting/interface/PixelFitter.h"
#include "RecoTracker/PixelTrackFitting/interface/KFBasedPixelFitter.h"

#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "TrackingTools/Records/interface/TransientRecHitRecord.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"

class KFBasedPixelFitterProducer : public edm::global::EDProducer<> {
public:
  explicit KFBasedPixelFitterProducer(const edm::ParameterSet& iConfig);
  ~KFBasedPixelFitterProducer() override {}

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const override;

  edm::EDGetTokenT<reco::BeamSpot> theBeamSpotToken;
  const edm::ESGetToken<TransientTrackingRecHitBuilder, TransientRecHitRecord> theTTRHBuilderToken;
  const edm::ESGetToken<Propagator, TrackingComponentsRecord> thePropagatorToken;
  const edm::ESGetToken<Propagator, TrackingComponentsRecord> thePropagatorOppositeToken;
  const edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> theTrackerToken;
  const edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> theFieldToken;
  const edm::EDPutTokenT<PixelFitter> thePutToken;
};

KFBasedPixelFitterProducer::KFBasedPixelFitterProducer(const edm::ParameterSet& iConfig)
    : theTTRHBuilderToken(esConsumes(edm::ESInputTag("", iConfig.getParameter<std::string>("TTRHBuilder")))),
      thePropagatorToken(esConsumes(edm::ESInputTag("", iConfig.getParameter<std::string>("propagator")))),
      thePropagatorOppositeToken(esConsumes(edm::ESInputTag("", iConfig.getParameter<std::string>("propagator")))),
      theTrackerToken(esConsumes()),
      theFieldToken(esConsumes()),
      thePutToken(produces()) {
  if (iConfig.getParameter<bool>("useBeamSpotConstraint")) {
    theBeamSpotToken = consumes<reco::BeamSpot>(iConfig.getParameter<edm::InputTag>("beamSpotConstraint"));
  }
}

void KFBasedPixelFitterProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  desc.add<bool>("useBeamSpotConstraint", true);
  desc.add<edm::InputTag>("beamSpotConstraint", edm::InputTag("offlineBeamSpot"));
  desc.add<std::string>("propagator", "PropagatorWithMaterial");
  desc.add<std::string>("propagatorOpposite", "PropagatorWithMaterialOpposite");
  desc.add<std::string>("TTRHBuilder", "PixelTTRHBuilderWithoutAngle");

  descriptions.add("kfBasedPixelFitter", desc);
}

void KFBasedPixelFitterProducer::produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  const reco::BeamSpot* beamspot = nullptr;
  if (!theBeamSpotToken.isUninitialized()) {
    edm::Handle<reco::BeamSpot> hbs;
    iEvent.getByToken(theBeamSpotToken, hbs);
    beamspot = hbs.product();
  }

  iEvent.emplace(thePutToken,
                 std::make_unique<KFBasedPixelFitter>(&iSetup.getData(thePropagatorToken),
                                                      &iSetup.getData(thePropagatorOppositeToken),
                                                      &iSetup.getData(theTTRHBuilderToken),
                                                      &iSetup.getData(theTrackerToken),
                                                      &iSetup.getData(theFieldToken),
                                                      beamspot));
}

DEFINE_FWK_MODULE(KFBasedPixelFitterProducer);
