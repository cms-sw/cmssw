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

#include "SimDataFormats/Associations/interface/LayerClusterToSimClusterAssociator.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/CaloRecHit/interface/CaloClusterFwd.h"

#include "FWCore/Utilities/interface/EDGetToken.h"

//
// class decleration
//

class LCToSCAssociatorEDProducer : public edm::global::EDProducer<> {
public:
  explicit LCToSCAssociatorEDProducer(const edm::ParameterSet &);
  ~LCToSCAssociatorEDProducer() override;

private:
  void produce(edm::StreamID, edm::Event &, const edm::EventSetup &) const override;

  edm::EDGetTokenT<SimClusterCollection> SCCollectionToken_;
  edm::EDGetTokenT<reco::CaloClusterCollection> LCCollectionToken_;
  edm::EDGetTokenT<hgcal::LayerClusterToSimClusterAssociator> associatorToken_;
};

LCToSCAssociatorEDProducer::LCToSCAssociatorEDProducer(const edm::ParameterSet &pset) {
  produces<hgcal::SimToRecoCollectionWithSimClusters>();
  produces<hgcal::RecoToSimCollectionWithSimClusters>();

  SCCollectionToken_ = consumes<SimClusterCollection>(pset.getParameter<edm::InputTag>("label_scl"));
  LCCollectionToken_ = consumes<reco::CaloClusterCollection>(pset.getParameter<edm::InputTag>("label_lcl"));
  associatorToken_ =
      consumes<hgcal::LayerClusterToSimClusterAssociator>(pset.getParameter<edm::InputTag>("associator"));
}

LCToSCAssociatorEDProducer::~LCToSCAssociatorEDProducer() {}

//
// member functions
//

// ------------ method called to produce the data  ------------
void LCToSCAssociatorEDProducer::produce(edm::StreamID, edm::Event &iEvent, const edm::EventSetup &iSetup) const {
  using namespace edm;

  edm::Handle<hgcal::LayerClusterToSimClusterAssociator> theAssociator;
  iEvent.getByToken(associatorToken_, theAssociator);

  Handle<SimClusterCollection> SCCollection;
  iEvent.getByToken(SCCollectionToken_, SCCollection);

  Handle<reco::CaloClusterCollection> LCCollection;
  iEvent.getByToken(LCCollectionToken_, LCCollection);

  // associate LC and SC
  LogTrace("AssociatorValidator") << "Calling associateRecoToSim method"
                                  << "\n";
  hgcal::RecoToSimCollectionWithSimClusters recSimColl = theAssociator->associateRecoToSim(LCCollection, SCCollection);

  LogTrace("AssociatorValidator") << "Calling associateSimToReco method"
                                  << "\n";
  hgcal::SimToRecoCollectionWithSimClusters simRecColl = theAssociator->associateSimToReco(LCCollection, SCCollection);

  auto rts = std::make_unique<hgcal::RecoToSimCollectionWithSimClusters>(recSimColl);
  auto str = std::make_unique<hgcal::SimToRecoCollectionWithSimClusters>(simRecColl);

  iEvent.put(std::move(rts));
  iEvent.put(std::move(str));
}

// define this as a plug-in
DEFINE_FWK_MODULE(LCToSCAssociatorEDProducer);
