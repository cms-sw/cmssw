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

#include "SimTracker/VertexAssociation/interface/VertexAssociatorByPositionAndTracks.h"
#include "SimDataFormats/Associations/interface/VertexToTrackingVertexAssociator.h"

class VertexAssociatorByPositionAndTracksProducer: public edm::global::EDProducer<> {
public:
  explicit VertexAssociatorByPositionAndTracksProducer(const edm::ParameterSet&);
  ~VertexAssociatorByPositionAndTracksProducer();

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);      

private:
  virtual void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;
      
  // ----------member data ---------------------------
  const double absZ_;
  const double sigmaZ_;
  const double maxRecoZ_;
  const double sharedTrackFraction_;

  edm::EDGetTokenT<reco::RecoToSimCollection> trackRecoToSimAssociationToken_;
  edm::EDGetTokenT<reco::SimToRecoCollection> trackSimToRecoAssociationToken_;
};

VertexAssociatorByPositionAndTracksProducer::VertexAssociatorByPositionAndTracksProducer(const edm::ParameterSet& config):
  absZ_(config.getParameter<double>("absZ")),
  sigmaZ_(config.getParameter<double>("sigmaZ")),
  maxRecoZ_(config.getParameter<double>("maxRecoZ")),
  sharedTrackFraction_(config.getParameter<double>("sharedTrackFraction")),
  trackRecoToSimAssociationToken_(consumes<reco::RecoToSimCollection>(config.getParameter<edm::InputTag>("trackAssociation"))),
  trackSimToRecoAssociationToken_(consumes<reco::SimToRecoCollection>(config.getParameter<edm::InputTag>("trackAssociation")))
{
  produces<reco::VertexToTrackingVertexAssociator>();
}

VertexAssociatorByPositionAndTracksProducer::~VertexAssociatorByPositionAndTracksProducer() {}

void VertexAssociatorByPositionAndTracksProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  
  // Matching conditions
  desc.add<double>("absZ", 0.1);
  desc.add<double>("sigmaZ", 3.0);
  desc.add<double>("maxRecoZ", 1000.0);
  desc.add<double>("sharedTrackFraction", -1.0);

  // Track-TrackingParticle association
  desc.add<edm::InputTag>("trackAssociation", edm::InputTag("trackingParticleRecoTrackAsssociation"));

  descriptions.add("VertexAssociatorByPositionAndTracks", desc);
}

void VertexAssociatorByPositionAndTracksProducer::produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup&) const {
  edm::Handle<reco::RecoToSimCollection > recotosimCollectionH;
  iEvent.getByToken(trackRecoToSimAssociationToken_, recotosimCollectionH);

  edm::Handle<reco::SimToRecoCollection > simtorecoCollectionH;
  iEvent.getByToken(trackSimToRecoAssociationToken_, simtorecoCollectionH);

  auto impl = std::make_unique<VertexAssociatorByPositionAndTracks>(&(iEvent.productGetter()),
                                                                    absZ_,
                                                                    sigmaZ_,
                                                                    maxRecoZ_,
                                                                    sharedTrackFraction_,
                                                                    recotosimCollectionH.product(),
                                                                    simtorecoCollectionH.product());

  auto toPut = std::make_unique<reco::VertexToTrackingVertexAssociator>(std::move(impl));
  iEvent.put(std::move(toPut));
}

DEFINE_FWK_MODULE(VertexAssociatorByPositionAndTracksProducer);
