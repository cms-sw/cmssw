//
// Original Author:  Leonardo Cristella
//         Created:  Tue Feb  2 10:52:11 CET 2021
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

#include "SimDataFormats/Associations/interface/MultiClusterToCaloParticleAssociator.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "SimDataFormats/CaloAnalysis/interface/CaloParticleFwd.h"
#include "DataFormats/ParticleFlowReco/interface/HGCalMultiCluster.h"

#include "FWCore/Utilities/interface/EDGetToken.h"

//
// class decleration
//

class MCToCPAssociatorEDProducer : public edm::global::EDProducer<> {
public:
  explicit MCToCPAssociatorEDProducer(const edm::ParameterSet &);
  ~MCToCPAssociatorEDProducer() override;

private:
  void produce(edm::StreamID, edm::Event &, const edm::EventSetup &) const override;

  edm::EDGetTokenT<CaloParticleCollection> CPCollectionToken_;
  edm::EDGetTokenT<reco::HGCalMultiClusterCollection> MCCollectionToken_;
  edm::EDGetTokenT<hgcal::MultiClusterToCaloParticleAssociator> associatorToken_;
};

MCToCPAssociatorEDProducer::MCToCPAssociatorEDProducer(const edm::ParameterSet &pset) {
  produces<hgcal::SimToRecoCollectionWithMultiClusters>();
  produces<hgcal::RecoToSimCollectionWithMultiClusters>();

  CPCollectionToken_ = consumes<CaloParticleCollection>(pset.getParameter<edm::InputTag>("label_cp"));
  MCCollectionToken_ = consumes<reco::HGCalMultiClusterCollection>(pset.getParameter<edm::InputTag>("label_mcl"));
  associatorToken_ =
      consumes<hgcal::MultiClusterToCaloParticleAssociator>(pset.getParameter<edm::InputTag>("associator"));
}

MCToCPAssociatorEDProducer::~MCToCPAssociatorEDProducer() {}

//
// member functions
//

// ------------ method called to produce the data  ------------
void MCToCPAssociatorEDProducer::produce(edm::StreamID, edm::Event &iEvent, const edm::EventSetup &iSetup) const {
  using namespace edm;

  edm::Handle<hgcal::MultiClusterToCaloParticleAssociator> theAssociator;
  iEvent.getByToken(associatorToken_, theAssociator);

  Handle<CaloParticleCollection> CPCollection;
  iEvent.getByToken(CPCollectionToken_, CPCollection);

  Handle<reco::HGCalMultiClusterCollection> MCCollection;
  iEvent.getByToken(MCCollectionToken_, MCCollection);

  // associate MutiCluster and CP
  LogTrace("AssociatorValidator") << "Calling associateRecoToSim method"
                                  << "\n";
  hgcal::RecoToSimCollectionWithMultiClusters recSimColl =
      theAssociator->associateRecoToSim(MCCollection, CPCollection);

  LogTrace("AssociatorValidator") << "Calling associateSimToReco method"
                                  << "\n";
  hgcal::SimToRecoCollectionWithMultiClusters simRecColl =
      theAssociator->associateSimToReco(MCCollection, CPCollection);

  auto rts = std::make_unique<hgcal::RecoToSimCollectionWithMultiClusters>(recSimColl);
  auto str = std::make_unique<hgcal::SimToRecoCollectionWithMultiClusters>(simRecColl);

  iEvent.put(std::move(rts));
  iEvent.put(std::move(str));
}

// define this as a plug-in
DEFINE_FWK_MODULE(MCToCPAssociatorEDProducer);
