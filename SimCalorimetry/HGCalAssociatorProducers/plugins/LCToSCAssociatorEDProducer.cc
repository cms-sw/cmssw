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
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "SimDataFormats/Associations/interface/LayerClusterToSimClusterAssociator.h"

//
// class decleration
//

class LCToSCAssociatorEDProducer : public edm::global::EDProducer<> {
public:
  explicit LCToSCAssociatorEDProducer(const edm::ParameterSet &);
  ~LCToSCAssociatorEDProducer() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

private:
  void produce(edm::StreamID, edm::Event &, const edm::EventSetup &) const override;

  edm::EDGetTokenT<SimClusterCollection> SCCollectionToken_;
  edm::EDGetTokenT<reco::CaloClusterCollection> LCCollectionToken_;
  edm::EDGetTokenT<ticl::LayerClusterToSimClusterAssociator> associatorToken_;
};

LCToSCAssociatorEDProducer::LCToSCAssociatorEDProducer(const edm::ParameterSet &pset) {
  produces<ticl::SimToRecoCollectionWithSimClusters>();
  produces<ticl::RecoToSimCollectionWithSimClusters>();

  SCCollectionToken_ = consumes<SimClusterCollection>(pset.getParameter<edm::InputTag>("label_scl"));
  LCCollectionToken_ = consumes<reco::CaloClusterCollection>(pset.getParameter<edm::InputTag>("label_lcl"));
  associatorToken_ = consumes<ticl::LayerClusterToSimClusterAssociator>(pset.getParameter<edm::InputTag>("associator"));
}

//
// member functions
//

// ------------ method called to produce the data  ------------
void LCToSCAssociatorEDProducer::produce(edm::StreamID, edm::Event &iEvent, const edm::EventSetup &iSetup) const {
  using namespace edm;

  edm::Handle<ticl::LayerClusterToSimClusterAssociator> theAssociator;
  iEvent.getByToken(associatorToken_, theAssociator);

  Handle<SimClusterCollection> SCCollection;
  iEvent.getByToken(SCCollectionToken_, SCCollection);

  Handle<reco::CaloClusterCollection> LCCollection;
  iEvent.getByToken(LCCollectionToken_, LCCollection);

  // Protection against missing CaloCluster collection
  if (!LCCollection.isValid()) {
    edm::LogWarning("LCToSCAssociatorEDProducer")
        << "CaloCluster collection is unavailable. Producing empty associations.";

    // Return empty collections
    auto emptyRecSimColl = std::make_unique<ticl::RecoToSimCollectionWithSimClusters>();
    auto emptySimRecColl = std::make_unique<ticl::SimToRecoCollectionWithSimClusters>();

    iEvent.put(std::move(emptyRecSimColl));
    iEvent.put(std::move(emptySimRecColl));
    return;
  }

  // associate LC and SC
  LogTrace("AssociatorValidator") << "Calling associateRecoToSim method\n";
  ticl::RecoToSimCollectionWithSimClusters recSimColl = theAssociator->associateRecoToSim(LCCollection, SCCollection);

  LogTrace("AssociatorValidator") << "Calling associateSimToReco method\n";
  ticl::SimToRecoCollectionWithSimClusters simRecColl = theAssociator->associateSimToReco(LCCollection, SCCollection);

  auto rts = std::make_unique<ticl::RecoToSimCollectionWithSimClusters>(recSimColl);
  auto str = std::make_unique<ticl::SimToRecoCollectionWithSimClusters>(simRecColl);

  iEvent.put(std::move(rts));
  iEvent.put(std::move(str));
}

void LCToSCAssociatorEDProducer::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("label_scl", edm::InputTag("scAssocByEnergyScoreProducer"));
  desc.add<edm::InputTag>("label_lcl", edm::InputTag("mix", "MergedCaloTruth"));
  desc.add<edm::InputTag>("associator", edm::InputTag("hgcalMergeLayerClusters"));
  descriptions.addWithDefaultLabel(desc);
}

// define this as a plug-in
DEFINE_FWK_MODULE(LCToSCAssociatorEDProducer);
