// system include files
#include <memory>
#include <string>

// user include files
#include "FWCore/Framework/interface/global/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "SimDataFormats/Associations/interface/MtdSimLayerClusterToTPAssociator.h"

#include "FWCore/Utilities/interface/EDGetToken.h"

//
// class decleration
//

class MtdSimLayerClusterToTPAssociatorEDProducer : public edm::global::EDProducer<> {
public:
  explicit MtdSimLayerClusterToTPAssociatorEDProducer(const edm::ParameterSet &);
  ~MtdSimLayerClusterToTPAssociatorEDProducer() override;

  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

private:
  void produce(edm::StreamID, edm::Event &, const edm::EventSetup &) const override;

  edm::EDGetTokenT<MtdSimLayerClusterCollection> simClustersToken_;
  edm::EDGetTokenT<TrackingParticleCollection> tpToken_;
  edm::EDGetTokenT<reco::MtdSimLayerClusterToTPAssociator> associatorToken_;
};

MtdSimLayerClusterToTPAssociatorEDProducer::MtdSimLayerClusterToTPAssociatorEDProducer(const edm::ParameterSet &pset) {
  produces<reco::SimToTPCollectionMtd>();
  produces<reco::TPToSimCollectionMtd>();

  simClustersToken_ = consumes<MtdSimLayerClusterCollection>(pset.getParameter<edm::InputTag>("mtdSimClustersTag"));
  tpToken_ = consumes<TrackingParticleCollection>(pset.getParameter<edm::InputTag>("trackingParticlesTag"));
  associatorToken_ = consumes<reco::MtdSimLayerClusterToTPAssociator>(pset.getParameter<edm::InputTag>("associator"));
}

MtdSimLayerClusterToTPAssociatorEDProducer::~MtdSimLayerClusterToTPAssociatorEDProducer() {}

//
// member functions
//

// ------------ method called to produce the data  ------------
void MtdSimLayerClusterToTPAssociatorEDProducer::produce(edm::StreamID,
                                                         edm::Event &iEvent,
                                                         const edm::EventSetup &iSetup) const {
  using namespace edm;

  edm::Handle<reco::MtdSimLayerClusterToTPAssociator> theAssociator;
  iEvent.getByToken(associatorToken_, theAssociator);

  edm::Handle<MtdSimLayerClusterCollection> simClusters;
  iEvent.getByToken(simClustersToken_, simClusters);

  edm::Handle<TrackingParticleCollection> trackingParticles;
  iEvent.getByToken(tpToken_, trackingParticles);

  reco::SimToTPCollectionMtd simToTPColl = theAssociator->associateSimToTP(simClusters, trackingParticles);
  reco::TPToSimCollectionMtd tpToSimColl = theAssociator->associateTPToSim(simClusters, trackingParticles);

  auto s2tp = std::make_unique<reco::SimToTPCollectionMtd>(simToTPColl);
  auto tp2s = std::make_unique<reco::TPToSimCollectionMtd>(tpToSimColl);

  iEvent.put(std::move(s2tp));
  iEvent.put(std::move(tp2s));
}

void MtdSimLayerClusterToTPAssociatorEDProducer::fillDescriptions(edm::ConfigurationDescriptions &cfg) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("associator", edm::InputTag("mtdSimLayerClusterToTPAssociatorByTrackId"));
  desc.add<edm::InputTag>("mtdSimClustersTag", edm::InputTag("mix", "MergedMtdTruthLC"));
  desc.add<edm::InputTag>("trackingParticlesTag", edm::InputTag("mix", "MergedTrackTruth"));

  cfg.add("mtdSimLayerClusterToTPAssociationDefault", desc);
}

// define this as a plug-in
DEFINE_FWK_MODULE(MtdSimLayerClusterToTPAssociatorEDProducer);
