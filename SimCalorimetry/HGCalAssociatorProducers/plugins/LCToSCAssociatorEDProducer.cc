//
// Original Author:  Leonardo Cristella
//         Created:  Thu Dec  3 10:52:11 CET 2020
//
//

// system include files
#include <memory>
#include <string>

// user include files
#include "DataFormats/CaloRecHit/interface/CaloClusterFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFClusterFwd.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/allowedValues.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "SimDataFormats/Associations/interface/LayerClusterToSimClusterAssociator.h"

#include <iostream>

//
// class declaration
//

template <typename CLUSTER>
class LCToSCAssociatorEDProducerT : public edm::global::EDProducer<> {
public:
  explicit LCToSCAssociatorEDProducerT(const edm::ParameterSet &);
  ~LCToSCAssociatorEDProducerT() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

private:
  void produce(edm::StreamID, edm::Event &, const edm::EventSetup &) const override;

  edm::InputTag label_lcl_;
  edm::InputTag label_scl_;
  std::vector<DetId::Detector> detIds_;

  edm::EDGetTokenT<SimClusterCollection> SCCollectionToken_;
  edm::EDGetTokenT<CLUSTER> LCCollectionToken_;
  edm::EDGetTokenT<ticl::LayerClusterToSimClusterAssociatorT<CLUSTER>> associatorToken_;
};

template <typename CLUSTER>
LCToSCAssociatorEDProducerT<CLUSTER>::LCToSCAssociatorEDProducerT(const edm::ParameterSet &pset) {
  produces<ticl::SimToRecoCollectionWithSimClustersT<CLUSTER>>();
  produces<ticl::RecoToSimCollectionWithSimClustersT<CLUSTER>>();

  label_lcl_ = pset.getParameter<edm::InputTag>("label_lcl");
  label_scl_ = pset.getParameter<edm::InputTag>("label_scl");

  std::vector<std::string> filter_sim_hits = pset.getParameter<std::vector<std::string>>("filter_sim_hits");
  simcluster_utils::check_detids(filter_sim_hits);
  detIds_ = simcluster_utils::join_detids(filter_sim_hits);

  LCCollectionToken_ = consumes<CLUSTER>(label_lcl_);
  SCCollectionToken_ = consumes<SimClusterCollection>(label_scl_);
  associatorToken_ =
      consumes<ticl::LayerClusterToSimClusterAssociatorT<CLUSTER>>(pset.getParameter<edm::InputTag>("associator"));
}

//
// member functions
//

// ------------ method called to produce the data  ------------
template <typename CLUSTER>
void LCToSCAssociatorEDProducerT<CLUSTER>::produce(edm::StreamID,
                                                   edm::Event &iEvent,
                                                   const edm::EventSetup &iSetup) const {
  edm::Handle<ticl::LayerClusterToSimClusterAssociatorT<CLUSTER>> theAssociator;
  iEvent.getByToken(associatorToken_, theAssociator);

  if (!theAssociator.isValid()) {
    edm::LogWarning("LCToSCAssociatorEDProducerT") << "Associator is unavailable.";
    return;
  }

  edm::Handle<SimClusterCollection> SCCollection;
  iEvent.getByToken(SCCollectionToken_, SCCollection);

  if (!SCCollection.isValid()) {
    edm::LogWarning("LCToSCAssociatorEDProducerT")
        << "SimCluster collection is unavailable. Producing empty associations.";

    return;
  }

  edm::Handle<CLUSTER> LCCollection;
  iEvent.getByToken(LCCollectionToken_, LCCollection);

  // Protections
  if (!SCCollection.isValid()) {
    edm::LogWarning("LCToSCAssociatorEDProducerT")
        << "CaloCluster collection with label " << label_scl_ << " is unavailable. Producing empty associations.";
  }
  if (!LCCollection.isValid()) {
    edm::LogWarning("LCToSCAssociatorEDProducer")
        << "CaloCluster collection with label " << label_lcl_ << " is unavailable. Producing empty associations.";

    // Return empty collections
    auto emptyRecSimColl = std::make_unique<ticl::RecoToSimCollectionWithSimClustersT<CLUSTER>>();
    auto emptySimRecColl = std::make_unique<ticl::SimToRecoCollectionWithSimClustersT<CLUSTER>>();

    iEvent.put(std::move(emptyRecSimColl));
    iEvent.put(std::move(emptySimRecColl));
    return;
  }

  // associate LC and SC
  LogTrace("AssociatorValidator") << "Calling associateRecoToSim method\n";
  ticl::RecoToSimCollectionWithSimClustersT<CLUSTER> recSimColl =
      theAssociator->associateRecoToSim(LCCollection, SCCollection, detIds_);

  LogTrace("AssociatorValidator") << "Calling associateSimToReco method\n";
  ticl::SimToRecoCollectionWithSimClustersT<CLUSTER> simRecColl =
      theAssociator->associateSimToReco(LCCollection, SCCollection, detIds_);

  auto rts = std::make_unique<ticl::RecoToSimCollectionWithSimClustersT<CLUSTER>>(recSimColl);
  auto str = std::make_unique<ticl::SimToRecoCollectionWithSimClustersT<CLUSTER>>(simRecColl);

  iEvent.put(std::move(rts));
  iEvent.put(std::move(str));
}

template <typename CLUSTER>
void LCToSCAssociatorEDProducerT<CLUSTER>::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("label_scl", edm::InputTag("mix", "MergedCaloTruth"));
  desc.add<std::vector<std::string>>("filter_sim_hits", {""});
  desc.add<edm::InputTag>("label_lcl", edm::InputTag("hgcalMergeLayerClusters"));
  desc.add<edm::InputTag>("associator", edm::InputTag("scAssocByEnergyScoreProducer"));
  descriptions.addWithDefaultLabel(desc);
}

// define this as a plug-in
using LCToSCAssociatorEDProducer = LCToSCAssociatorEDProducerT<reco::CaloClusterCollection>;
DEFINE_FWK_MODULE(LCToSCAssociatorEDProducer);
using PCToSCAssociatorEDProducer = LCToSCAssociatorEDProducerT<reco::PFClusterCollection>;
DEFINE_FWK_MODULE(PCToSCAssociatorEDProducer);
