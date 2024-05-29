//
// Original Author:  Leonardo Cristella
//         Created:  Wed Mar  30 10:52:11 CET 2022
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

#include "SimDataFormats/Associations/interface/TracksterToSimTracksterHitLCAssociator.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/HGCalReco/interface/Trackster.h"
#include "SimDataFormats/CaloAnalysis/interface/CaloParticleFwd.h"

#include "FWCore/Utilities/interface/EDGetToken.h"

class TSToSimTSHitLCAssociatorEDProducer : public edm::global::EDProducer<> {
public:
  explicit TSToSimTSHitLCAssociatorEDProducer(const edm::ParameterSet &);
  ~TSToSimTSHitLCAssociatorEDProducer() override;

private:
  void produce(edm::StreamID, edm::Event &, const edm::EventSetup &) const override;

  edm::EDGetTokenT<ticl::TracksterCollection> TSCollectionToken_;
  edm::EDGetTokenT<ticl::TracksterCollection> SimTSCollectionToken_;
  edm::EDGetTokenT<ticl::TracksterCollection> SimTSFromCPCollectionToken_;
  edm::EDGetTokenT<reco::CaloClusterCollection> LCCollectionToken_;
  edm::EDGetTokenT<SimClusterCollection> SCCollectionToken_;
  edm::EDGetTokenT<CaloParticleCollection> CPCollectionToken_;
  edm::EDGetTokenT<std::map<uint, std::vector<uint>>> simTrackstersMap_;
  ticl::validationType valType_;
  edm::EDGetTokenT<ticl::TracksterToSimTracksterHitLCAssociator> associatorToken_;
};

TSToSimTSHitLCAssociatorEDProducer::TSToSimTSHitLCAssociatorEDProducer(const edm::ParameterSet &pset) {
  produces<ticl::SimToRecoCollectionSimTracksters>("simToReco");
  produces<ticl::RecoToSimCollectionSimTracksters>("recoToSim");

  TSCollectionToken_ = consumes<ticl::TracksterCollection>(pset.getParameter<edm::InputTag>("label_tst"));
  SimTSCollectionToken_ = consumes<ticl::TracksterCollection>(pset.getParameter<edm::InputTag>("label_simTst"));
  LCCollectionToken_ = consumes<reco::CaloClusterCollection>(pset.getParameter<edm::InputTag>("label_lcl"));
  SCCollectionToken_ = consumes<SimClusterCollection>(pset.getParameter<edm::InputTag>("label_scl"));
  CPCollectionToken_ = consumes<CaloParticleCollection>(pset.getParameter<edm::InputTag>("label_cp"));
  associatorToken_ =
      consumes<ticl::TracksterToSimTracksterHitLCAssociator>(pset.getParameter<edm::InputTag>("associator"));
}

TSToSimTSHitLCAssociatorEDProducer::~TSToSimTSHitLCAssociatorEDProducer() {}

void TSToSimTSHitLCAssociatorEDProducer::produce(edm::StreamID,
                                                 edm::Event &iEvent,
                                                 const edm::EventSetup &iSetup) const {
  using namespace edm;

  edm::Handle<ticl::TracksterToSimTracksterHitLCAssociator> theAssociator;
  iEvent.getByToken(associatorToken_, theAssociator);

  Handle<ticl::TracksterCollection> TSCollection;
  iEvent.getByToken(TSCollectionToken_, TSCollection);

  Handle<ticl::TracksterCollection> SimTSCollection;
  iEvent.getByToken(SimTSCollectionToken_, SimTSCollection);

  Handle<reco::CaloClusterCollection> LCCollection;
  iEvent.getByToken(LCCollectionToken_, LCCollection);

  Handle<SimClusterCollection> SCCollection;
  iEvent.getByToken(SCCollectionToken_, SCCollection);

  Handle<CaloParticleCollection> CPCollection;
  iEvent.getByToken(CPCollectionToken_, CPCollection);

  // associate TS and SimTS
  LogTrace("AssociatorValidator") << "Calling associateRecoToSim method\n";

  ticl::RecoToSimCollectionSimTracksters recSimColl =
      theAssociator->associateRecoToSim(TSCollection, LCCollection, SCCollection, CPCollection, SimTSCollection);

  LogTrace("AssociatorValidator") << "Calling associateSimToReco method\n";
  ticl::SimToRecoCollectionSimTracksters simRecColl =
      theAssociator->associateSimToReco(TSCollection, LCCollection, SCCollection, CPCollection, SimTSCollection);

  auto rts = std::make_unique<ticl::RecoToSimCollectionSimTracksters>(recSimColl);
  auto str = std::make_unique<ticl::SimToRecoCollectionSimTracksters>(simRecColl);

  iEvent.put(std::move(rts), "recoToSim");
  iEvent.put(std::move(str), "simToReco");
}

DEFINE_FWK_MODULE(TSToSimTSHitLCAssociatorEDProducer);
