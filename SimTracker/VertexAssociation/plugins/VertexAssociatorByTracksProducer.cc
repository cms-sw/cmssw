#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "DataFormats/Common/interface/Handle.h"

#include "SimTracker/Common/interface/TrackingParticleSelector.h"

#include "SimTracker/VertexAssociation/interface/VertexAssociatorByTracks.h"
#include "SimDataFormats/Associations/interface/VertexToTrackingVertexAssociator.h"

class VertexAssociatorByTracksProducer: public edm::global::EDProducer<> {
public:
  explicit VertexAssociatorByTracksProducer(const edm::ParameterSet&);
  ~VertexAssociatorByTracksProducer();

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  virtual void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

  // ----------member data ---------------------------
  const double R2SMatchedSimRatio_;
  const double R2SMatchedRecoRatio_;
  const double S2RMatchedSimRatio_;
  const double S2RMatchedRecoRatio_;

  const TrackingParticleSelector selector_;
  const reco::TrackBase::TrackQuality trackQuality_;

  edm::EDGetTokenT<reco::RecoToSimCollection> trackRecoToSimAssociationToken_;
  edm::EDGetTokenT<reco::SimToRecoCollection> trackSimToRecoAssociationToken_;
};

namespace {
  TrackingParticleSelector makeSelector(const edm::ParameterSet& param) {
    return TrackingParticleSelector(
                    param.getParameter<double>("ptMinTP"),
                    param.getParameter<double>("minRapidityTP"),
                    param.getParameter<double>("maxRapidityTP"),
                    param.getParameter<double>("tipTP"),
                    param.getParameter<double>("lipTP"),
                    param.getParameter<int>("minHitTP"),
                    param.getParameter<bool>("signalOnlyTP"),
                    param.getParameter<bool>("intimeOnlyTP"),
                    param.getParameter<bool>("chargedOnlyTP"),
		    param.getParameter<bool>("stableOnlyTP"),
                    param.getParameter<std::vector<int> >("pdgIdTP")
                );
  }
}

VertexAssociatorByTracksProducer::VertexAssociatorByTracksProducer(const edm::ParameterSet& config):
  R2SMatchedSimRatio_(config.getParameter<double>("R2SMatchedSimRatio")),
  R2SMatchedRecoRatio_(config.getParameter<double>("R2SMatchedRecoRatio")),
  S2RMatchedSimRatio_(config.getParameter<double>("S2RMatchedSimRatio")),
  S2RMatchedRecoRatio_(config.getParameter<double>("S2RMatchedRecoRatio")),
  selector_(makeSelector(config.getParameter<edm::ParameterSet>("trackingParticleSelector"))),
  trackQuality_(reco::TrackBase::qualityByName(config.getParameter<std::string>("trackQuality"))),
  trackRecoToSimAssociationToken_(consumes<reco::RecoToSimCollection>(config.getParameter<edm::InputTag>("trackAssociation"))),
  trackSimToRecoAssociationToken_(consumes<reco::SimToRecoCollection>(config.getParameter<edm::InputTag>("trackAssociation")))
{
  produces<reco::VertexToTrackingVertexAssociator>();
}

VertexAssociatorByTracksProducer::~VertexAssociatorByTracksProducer() {}

void VertexAssociatorByTracksProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  // Matching conditions
  desc.add<double>("R2SMatchedSimRatio", 0.3);
  desc.add<double>("R2SMatchedRecoRatio", 0.0);
  desc.add<double>("S2RMatchedSimRatio", 0.0);
  desc.add<double>("S2RMatchedRecoRatio", 0.3);

  //RecoTrack selection
  desc.add<std::string>("trackQuality", "highPurity");

  // TrackingParticle selection
  edm::ParameterSetDescription descTp;

  descTp.add<double>("lipTP", 30.0);
  descTp.add<bool>("chargedOnlyTP", true);
  descTp.add<std::vector<int>>("pdgIdTP",  std::vector<int>());
  descTp.add<bool>("signalOnlyTP", true);
  descTp.add<double>("minRapidityTP", -2.4);
  descTp.add<int>("minHitTP", 0);
  descTp.add<double>("ptMinTP", 0.9);
  descTp.add<double>("maxRapidityTP", 2.4);
  descTp.add<double>("tipTP", 3.5);
  desc.add<edm::ParameterSetDescription>("trackingParticleSelector", descTp);

  // Track-TrackingParticle association
  desc.add<edm::InputTag>("trackAssociation", edm::InputTag("trackingParticleRecoTrackAsssociation"));

  descriptions.add("VertexAssociatorByTracks", desc);
}

void VertexAssociatorByTracksProducer::produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup&) const {
  edm::Handle<reco::RecoToSimCollection > recotosimCollectionH;
  iEvent.getByToken(trackRecoToSimAssociationToken_, recotosimCollectionH);

  edm::Handle<reco::SimToRecoCollection > simtorecoCollectionH;
  iEvent.getByToken(trackSimToRecoAssociationToken_, simtorecoCollectionH);

  auto impl = std::make_unique<VertexAssociatorByTracks>(&(iEvent.productGetter()),
                                                         R2SMatchedSimRatio_,
                                                         R2SMatchedRecoRatio_,
                                                         S2RMatchedSimRatio_,
                                                         S2RMatchedRecoRatio_,
                                                         &selector_,
                                                         trackQuality_,
                                                         recotosimCollectionH.product(),
                                                         simtorecoCollectionH.product());

  auto toPut = std::make_unique<reco::VertexToTrackingVertexAssociator>(std::move(impl));
  iEvent.put(std::move(toPut));
}

DEFINE_FWK_MODULE(VertexAssociatorByTracksProducer);
