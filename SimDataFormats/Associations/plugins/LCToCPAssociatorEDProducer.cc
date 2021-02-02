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
#include "SimDataFormats/CaloAnalysis/interface/CaloParticleFwd.h"
#include "DataFormats/CaloRecHit/interface/CaloClusterFwd.h"

#include "FWCore/Utilities/interface/EDGetToken.h"

//
// class decleration
//

class LCToCPAssociatorEDProducer : public edm::global::EDProducer<> {
public:
  explicit LCToCPAssociatorEDProducer(const edm::ParameterSet &);
  ~LCToCPAssociatorEDProducer() override;

private:
  void produce(edm::StreamID, edm::Event &, const edm::EventSetup &) const override;

  edm::EDGetTokenT<CaloParticleCollection> CPCollectionToken_;
  edm::EDGetTokenT<reco::CaloClusterCollection> LCCollectionToken_;
  edm::EDGetTokenT<hgcal::LayerClusterToCaloParticleAssociator> associatorToken_;
};

LCToCPAssociatorEDProducer::LCToCPAssociatorEDProducer(const edm::ParameterSet &pset) {
  produces<hgcal::SimToRecoCollection>();
  produces<hgcal::RecoToSimCollection>();

  CPCollectionToken_ = consumes<CaloParticleCollection>(pset.getParameter<edm::InputTag>("label_cp"));
  LCCollectionToken_ = consumes<reco::CaloClusterCollection>(pset.getParameter<edm::InputTag>("label_lc"));
  associatorToken_ =
      consumes<hgcal::LayerClusterToCaloParticleAssociator>(pset.getParameter<edm::InputTag>("associator"));
}

LCToCPAssociatorEDProducer::~LCToCPAssociatorEDProducer() {}

//
// member functions
//

// ------------ method called to produce the data  ------------
void LCToCPAssociatorEDProducer::produce(edm::StreamID, edm::Event &iEvent, const edm::EventSetup &iSetup) const {
  using namespace edm;

  edm::Handle<hgcal::LayerClusterToCaloParticleAssociator> theAssociator;
  iEvent.getByToken(associatorToken_, theAssociator);

  Handle<CaloParticleCollection> CPCollection;
  iEvent.getByToken(CPCollectionToken_, CPCollection);

  Handle<reco::CaloClusterCollection> LCCollection;
  iEvent.getByToken(LCCollectionToken_, LCCollection);

  // associate LC and CP
  LogTrace("AssociatorValidator") << "Calling associateRecoToSim method"
                                  << "\n";
  hgcal::RecoToSimCollection recSimColl = theAssociator->associateRecoToSim(LCCollection, CPCollection);

  LogTrace("AssociatorValidator") << "Calling associateSimToReco method"
                                  << "\n";
  hgcal::SimToRecoCollection simRecColl = theAssociator->associateSimToReco(LCCollection, CPCollection);

  auto rts = std::make_unique<hgcal::RecoToSimCollection>(recSimColl);
  auto str = std::make_unique<hgcal::SimToRecoCollection>(simRecColl);

  iEvent.put(std::move(rts));
  iEvent.put(std::move(str));
}

// define this as a plug-in
DEFINE_FWK_MODULE(LCToCPAssociatorEDProducer);
