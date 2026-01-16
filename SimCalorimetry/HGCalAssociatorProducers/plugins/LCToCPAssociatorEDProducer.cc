//
// Original Author:  Leonardo Cristella
//         Created:  Thu Dec  3 10:52:11 CET 2020
//
//

// system include files
#include <memory>
#include <string>

// user include files
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "SimDataFormats/Associations/interface/LayerClusterToCaloParticleAssociator.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "SimDataFormats/CaloAnalysis/interface/CaloParticleFwd.h"
#include "SimDataFormats/CaloAnalysis/interface/CaloParticle.h"
#include "DataFormats/CaloRecHit/interface/CaloClusterFwd.h"
#include "DataFormats/CaloRecHit/interface/CaloCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFClusterFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "DataFormats/HGCRecHit/interface/HGCRecHitCollections.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHitFwd.h"
#include "FWCore/Utilities/interface/EDGetToken.h"

//
// class declaration
//

template <typename CLUSTER>
class LCToCPAssociatorEDProducerT : public edm::global::EDProducer<> {
public:
  explicit LCToCPAssociatorEDProducerT(const edm::ParameterSet &);
  ~LCToCPAssociatorEDProducerT() override = default;

  // static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

private:
  void produce(edm::StreamID, edm::Event &, const edm::EventSetup &) const override;

  edm::InputTag label_lc;

  edm::EDGetTokenT<CaloParticleCollection> CPCollectionToken_;
  edm::EDGetTokenT<CLUSTER> LCCollectionToken_;
  edm::EDGetTokenT<ticl::LayerClusterToCaloParticleAssociatorT<CLUSTER>> associatorToken_;
};

template <typename CLUSTER>
LCToCPAssociatorEDProducerT<CLUSTER>::LCToCPAssociatorEDProducerT(const edm::ParameterSet &pset) {
  produces<ticl::SimToRecoCollectionT<CLUSTER>>();
  produces<ticl::RecoToSimCollectionT<CLUSTER>>();

  label_lc = pset.getParameter<edm::InputTag>("label_lc");

  CPCollectionToken_ = consumes<CaloParticleCollection>(pset.getParameter<edm::InputTag>("label_cp"));
  LCCollectionToken_ = consumes<CLUSTER>(label_lc);
  associatorToken_ =
      consumes<ticl::LayerClusterToCaloParticleAssociatorT<CLUSTER>>(pset.getParameter<edm::InputTag>("associator"));
}

//
// member functions
//

// ------------ method called to produce the data  ------------
template <typename CLUSTER>
void LCToCPAssociatorEDProducerT<CLUSTER>::produce(edm::StreamID,
                                                   edm::Event &iEvent,
                                                   const edm::EventSetup &iSetup) const {
  using namespace edm;

  edm::Handle<ticl::LayerClusterToCaloParticleAssociatorT<CLUSTER>> theAssociator;
  iEvent.getByToken(associatorToken_, theAssociator);

  if (!theAssociator.isValid()) {
    edm::LogWarning("LCToCPAssociatorEDProducerT") << "Associator is unavailable.";
    return;
  }

  Handle<CaloParticleCollection> CPCollection;
  iEvent.getByToken(CPCollectionToken_, CPCollection);
  if (!CPCollection.isValid()) {
    edm::LogWarning("LCToCPAssociatorEDProducerT") << "CaloParticle collection is unavailable.";
    return;
  }

  Handle<CLUSTER> LCCollection;
  iEvent.getByToken(LCCollectionToken_, LCCollection);

  // Protection against missing cluster collection
  if (!LCCollection.isValid()) {
    edm::LogWarning("LCToCPAssociatorEDProducerT")
        << "CaloCluster collection with label " << label_lc << " is unavailable. Producing empty associations.";

    // Return empty collections
    auto emptyRecSimColl = std::make_unique<ticl::RecoToSimCollectionT<CLUSTER>>();
    auto emptySimRecColl = std::make_unique<ticl::SimToRecoCollectionT<CLUSTER>>();

    iEvent.put(std::move(emptyRecSimColl));
    iEvent.put(std::move(emptySimRecColl));
    return;
  }

  // associate LC and CP
  LogTrace("AssociatorValidator") << "Calling associateRecoToSim method\n";
  ticl::RecoToSimCollectionT<CLUSTER> recSimColl = theAssociator->associateRecoToSim(LCCollection, CPCollection);

  LogTrace("AssociatorValidator") << "Calling associateSimToReco method\n";
  ticl::SimToRecoCollectionT<CLUSTER> simRecColl = theAssociator->associateSimToReco(LCCollection, CPCollection);

  auto rts = std::make_unique<ticl::RecoToSimCollectionT<CLUSTER>>(recSimColl);
  auto str = std::make_unique<ticl::SimToRecoCollectionT<CLUSTER>>(simRecColl);

  iEvent.put(std::move(rts));
  iEvent.put(std::move(str));
}

template <typename CLUSTER>
void LCToCPAssociatorEDProducerT<CLUSTER>::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("label_cp", edm::InputTag("mix", "MergedCaloTruth"));
  desc.add<edm::InputTag>("label_lc", edm::InputTag("hgcalMergeLayerClusters"));
  desc.add<edm::InputTag>("associator", edm::InputTag("lcAssocByEnergyScoreProducer"));
  descriptions.addWithDefaultLabel(desc);
}

// define this as a plug-in
using LCToCPAssociatorEDProducer = LCToCPAssociatorEDProducerT<reco::CaloClusterCollection>;
DEFINE_FWK_MODULE(LCToCPAssociatorEDProducer);
using PCToCPAssociatorEDProducer = LCToCPAssociatorEDProducerT<reco::PFClusterCollection>;
DEFINE_FWK_MODULE(PCToCPAssociatorEDProducer);
