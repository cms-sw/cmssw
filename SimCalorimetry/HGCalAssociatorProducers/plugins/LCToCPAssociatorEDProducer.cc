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
#include "DataFormats/CaloRecHit/interface/CaloClusterFwd.h"
#include "DataFormats/HGCRecHit/interface/HGCRecHitCollections.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHitFwd.h"
#include "FWCore/Utilities/interface/EDGetToken.h"

//
// class declaration
//

class LCToCPAssociatorEDProducer : public edm::global::EDProducer<> {
public:
  explicit LCToCPAssociatorEDProducer(const edm::ParameterSet &);
  ~LCToCPAssociatorEDProducer() override;

  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);
  
private:
  void produce(edm::StreamID, edm::Event &, const edm::EventSetup &) const override;

  edm::InputTag label_lc;
  
  edm::EDGetTokenT<CaloParticleCollection> CPCollectionToken_;
  edm::EDGetTokenT<reco::CaloClusterCollection> LCCollectionToken_;
  edm::EDGetTokenT<ticl::LayerClusterToCaloParticleAssociator> associatorToken_;
};

LCToCPAssociatorEDProducer::LCToCPAssociatorEDProducer(const edm::ParameterSet &pset) {
  produces<ticl::SimToRecoCollection>();
  produces<ticl::RecoToSimCollection>();

  label_lc = pset.getParameter<edm::InputTag>("label_lc");
	
  CPCollectionToken_ = consumes<CaloParticleCollection>(pset.getParameter<edm::InputTag>("label_cp"));
  LCCollectionToken_ = consumes<reco::CaloClusterCollection>(label_lc);
  associatorToken_   = consumes<ticl::LayerClusterToCaloParticleAssociator>(pset.getParameter<edm::InputTag>("associator"));
}

LCToCPAssociatorEDProducer::~LCToCPAssociatorEDProducer() {}

//
// member functions
//

// ------------ method called to produce the data  ------------
void LCToCPAssociatorEDProducer::produce(edm::StreamID, edm::Event &iEvent, const edm::EventSetup &iSetup) const {
  using namespace edm;

  edm::Handle<ticl::LayerClusterToCaloParticleAssociator> theAssociator;
  iEvent.getByToken(associatorToken_, theAssociator);

  Handle<CaloParticleCollection> CPCollection;
  iEvent.getByToken(CPCollectionToken_, CPCollection);

  Handle<reco::CaloClusterCollection> LCCollection;
  iEvent.getByToken(LCCollectionToken_, LCCollection);

  // Protection against missing CaloCluster collection
  if (!LCCollection.isValid()) {
    edm::LogWarning("LCToCPAssociatorEDProducer")
        << "CaloCluster collection with label\n    "
		<< label_lc
		<< "\nis unavailable. Producing empty associations.";

    // Return empty collections
    auto emptyRecSimColl = std::make_unique<ticl::RecoToSimCollection>();
    auto emptySimRecColl = std::make_unique<ticl::SimToRecoCollection>();

    iEvent.put(std::move(emptyRecSimColl));
    iEvent.put(std::move(emptySimRecColl));
    return;
  }

  // associate LC and CP
  LogTrace("AssociatorValidator") << "Calling associateRecoToSim method\n";
  ticl::RecoToSimCollection recSimColl = theAssociator->associateRecoToSim(LCCollection, CPCollection);

  LogTrace("AssociatorValidator") << "Calling associateSimToReco method\n";
  ticl::SimToRecoCollection simRecColl = theAssociator->associateSimToReco(LCCollection, CPCollection);

  auto rts = std::make_unique<ticl::RecoToSimCollection>(recSimColl);
  auto str = std::make_unique<ticl::SimToRecoCollection>(simRecColl);

  iEvent.put(std::move(rts));
  iEvent.put(std::move(str));
}

void LCToCPAssociatorEDProducer::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("label_cp", edm::InputTag("mix", "MergedCaloTruth"));
  desc.add<edm::InputTag>("label_lc", edm::InputTag("hgcalMergeLayerClusters"));
  desc.add<edm::InputTag>("associator", edm::InputTag("lcAssocByEnergyScoreProducer"));
  descriptions.addWithDefaultLabel(desc);
}

// define this as a plug-in
DEFINE_FWK_MODULE(LCToCPAssociatorEDProducer);
