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

#include "SimDataFormats/Associations/interface/LayerClusterToSimTracksterAssociator.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "SimDataFormats/CaloAnalysis/interface/CaloParticleFwd.h"
#include "DataFormats/CaloRecHit/interface/CaloClusterFwd.h"

#include "FWCore/Utilities/interface/EDGetToken.h"

//
// class decleration
//

class LCToSimTSAssociatorEDProducer : public edm::global::EDProducer<> {
public:
  explicit LCToSimTSAssociatorEDProducer(const edm::ParameterSet &);
  ~LCToSimTSAssociatorEDProducer() override;

private:
  void produce(edm::StreamID, edm::Event &, const edm::EventSetup &) const override;

  edm::EDGetTokenT<reco::CaloClusterCollection> LCCollectionToken_;
  edm::EDGetTokenT<ticl::TracksterCollection> SimTSCollectionToken_;
  edm::EDGetTokenT<ticl::LayerClusterToSimTracksterAssociator> associatorToken_;

  edm::EDGetTokenT<CaloParticleCollection> CPCollectionToken_;
  edm::InputTag associatorCP_;
  edm::EDGetTokenT<ticl::RecoToSimCollection> associationMapLCToCPToken_;
  edm::EDGetTokenT<ticl::SimToRecoCollection> associationMapCPToLCToken_;

  edm::EDGetTokenT<SimClusterCollection> SCCollectionToken_;
  edm::InputTag associatorSC_;
  edm::EDGetTokenT<ticl::RecoToSimCollectionWithSimClusters> associationMapLCToSCToken_;
  edm::EDGetTokenT<ticl::SimToRecoCollectionWithSimClusters> associationMapSCToLCToken_;
};

LCToSimTSAssociatorEDProducer::LCToSimTSAssociatorEDProducer(const edm::ParameterSet &pset)
    : LCCollectionToken_(consumes<reco::CaloClusterCollection>(pset.getParameter<edm::InputTag>("label_lc"))),
      SimTSCollectionToken_(consumes<ticl::TracksterCollection>(pset.getParameter<edm::InputTag>("label_simTst"))),
      associatorToken_(
          consumes<ticl::LayerClusterToSimTracksterAssociator>(pset.getParameter<edm::InputTag>("associator"))),
      CPCollectionToken_(consumes<CaloParticleCollection>(pset.getParameter<edm::InputTag>("label_cp"))),
      associatorCP_(pset.getParameter<edm::InputTag>("associator_cp")),
      associationMapLCToCPToken_(consumes<ticl::RecoToSimCollection>(associatorCP_)),
      associationMapCPToLCToken_(consumes<ticl::SimToRecoCollection>(associatorCP_)),
      SCCollectionToken_(consumes<SimClusterCollection>(pset.getParameter<edm::InputTag>("label_scl"))),
      associatorSC_(pset.getParameter<edm::InputTag>("associator_sc")),
      associationMapLCToSCToken_(consumes<ticl::RecoToSimCollectionWithSimClusters>(associatorSC_)),
      associationMapSCToLCToken_(consumes<ticl::SimToRecoCollectionWithSimClusters>(associatorSC_)) {
  produces<ticl::SimTracksterToRecoCollection>();
  produces<ticl::RecoToSimTracksterCollection>();
}

LCToSimTSAssociatorEDProducer::~LCToSimTSAssociatorEDProducer() {}

//
// member functions
//

// ------------ method called to produce the data  ------------
void LCToSimTSAssociatorEDProducer::produce(edm::StreamID, edm::Event &iEvent, const edm::EventSetup &iSetup) const {
  using namespace edm;

  edm::Handle<ticl::LayerClusterToSimTracksterAssociator> theAssociator;
  iEvent.getByToken(associatorToken_, theAssociator);

  Handle<reco::CaloClusterCollection> LCCollection;
  iEvent.getByToken(LCCollectionToken_, LCCollection);

  Handle<ticl::TracksterCollection> SimTSCollection;
  iEvent.getByToken(SimTSCollectionToken_, SimTSCollection);

  Handle<CaloParticleCollection> CPCollection;
  iEvent.getByToken(CPCollectionToken_, CPCollection);
  const auto &LCToCPsColl = iEvent.get(associationMapLCToCPToken_);
  const auto &CPToLCsColl = iEvent.get(associationMapCPToLCToken_);

  Handle<SimClusterCollection> SCCollection;
  iEvent.getByToken(SCCollectionToken_, SCCollection);
  const auto &LCToSCsColl = iEvent.get(associationMapLCToSCToken_);
  const auto &SCToLCsColl = iEvent.get(associationMapSCToLCToken_);

  // associate LC and SimTS
  LogTrace("AssociatorValidator") << "Calling associateRecoToSim method\n";
  ticl::RecoToSimTracksterCollection recSimColl = theAssociator->associateRecoToSim(
      LCCollection, SimTSCollection, CPCollection, LCToCPsColl, SCCollection, LCToSCsColl);

  LogTrace("AssociatorValidator") << "Calling associateSimToReco method\n";
  ticl::SimTracksterToRecoCollection simRecColl = theAssociator->associateSimToReco(
      LCCollection, SimTSCollection, CPCollection, CPToLCsColl, SCCollection, SCToLCsColl);

  auto rts = std::make_unique<ticl::RecoToSimTracksterCollection>(recSimColl);
  auto str = std::make_unique<ticl::SimTracksterToRecoCollection>(simRecColl);

  iEvent.put(std::move(rts));
  iEvent.put(std::move(str));
}

// define this as a plug-in
DEFINE_FWK_MODULE(LCToSimTSAssociatorEDProducer);
