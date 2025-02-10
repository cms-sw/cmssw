// Author: Felice Pantaleo, felice.pantaleo@cern.ch 06/2024

// user include files
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "DataFormats/HGCalReco/interface/Trackster.h"
#include "DataFormats/CaloRecHit/interface/CaloCluster.h"
#include "SimDataFormats/Associations/interface/TICLAssociationMap.h"
#include "DataFormats/Provenance/interface/ProductID.h"
#include "LCToTSAssociatorProducer.h"

LCToTSAssociatorProducer::LCToTSAssociatorProducer(const edm::ParameterSet &pset)
    : LCCollectionToken_(consumes<std::vector<reco::CaloCluster>>(pset.getParameter<edm::InputTag>("layer_clusters"))),
      tracksterCollectionToken_(
          consumes<std::vector<ticl::Trackster>>(pset.getParameter<edm::InputTag>("tracksters"))) {
  produces<
      ticl::AssociationMap<ticl::mapWithSharedEnergy, std::vector<reco::CaloCluster>, std::vector<ticl::Trackster>>>();
}

LCToTSAssociatorProducer::~LCToTSAssociatorProducer() {}

//
// member functions
//

// ------------ method called to produce the data  ------------
void LCToTSAssociatorProducer::produce(edm::StreamID, edm::Event &iEvent, const edm::EventSetup &iSetup) const {
  using namespace edm;

  Handle<std::vector<reco::CaloCluster>> layer_clusters;
  iEvent.getByToken(LCCollectionToken_, layer_clusters);

  Handle<std::vector<ticl::Trackster>> tracksters;
  iEvent.getByToken(tracksterCollectionToken_, tracksters);

  // Create association map
  auto lcToTracksterMap = std::make_unique<
      ticl::AssociationMap<ticl::mapWithSharedEnergy, std::vector<reco::CaloCluster>, std::vector<ticl::Trackster>>>(
      layer_clusters, tracksters, iEvent);

  // Loop over tracksters
  for (unsigned int tracksterId = 0; tracksterId < tracksters->size(); ++tracksterId) {
    const auto &trackster = (*tracksters)[tracksterId];
    // Loop over vertices in trackster
    for (unsigned int i = 0; i < trackster.vertices().size(); ++i) {
      // Get layerCluster
      const auto &lc = (*layer_clusters)[trackster.vertices()[i]];
      float sharedEnergy = lc.energy() / trackster.vertex_multiplicity()[i];
      edm::Ref<std::vector<reco::CaloCluster>> lcRef(layer_clusters, trackster.vertices()[i]);
      edm::Ref<std::vector<ticl::Trackster>> tracksterRef(tracksters, tracksterId);
      lcToTracksterMap->insert(lcRef, tracksterRef, sharedEnergy);
    }
  }
  iEvent.put(std::move(lcToTracksterMap));
}

void LCToTSAssociatorProducer::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("layer_clusters", edm::InputTag("hgcalMergeLayerClusters"));
  desc.add<edm::InputTag>("tracksters", edm::InputTag("ticlTracksters"));
  descriptions.add("LCToTSAssociatorProducer", desc);
}

DEFINE_FWK_MODULE(LCToTSAssociatorProducer);