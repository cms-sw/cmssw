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

#include "SimDataFormats/Associations/interface/TracksterToSimTracksterAssociator.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/HGCalReco/interface/Trackster.h"

#include "FWCore/Utilities/interface/EDGetToken.h"

//
// class decleration
//

class TSToSimTSAssociatorEDProducer : public edm::global::EDProducer<> {
public:
  explicit TSToSimTSAssociatorEDProducer(const edm::ParameterSet &);
  ~TSToSimTSAssociatorEDProducer() override;

private:
  void produce(edm::StreamID, edm::Event &, const edm::EventSetup &) const override;

  edm::EDGetTokenT<ticl::TracksterCollection> TSCollectionToken_;
  edm::EDGetTokenT<ticl::TracksterCollection> SimTSCollectionToken_;
  edm::EDGetTokenT<reco::CaloClusterCollection> LCCollectionToken_;
  edm::EDGetTokenT<ticl::TracksterToSimTracksterAssociator> associatorToken_;
};

TSToSimTSAssociatorEDProducer::TSToSimTSAssociatorEDProducer(const edm::ParameterSet &pset) {
  produces<ticl::SimToRecoCollectionSimTracksters>("simToReco");
  produces<ticl::RecoToSimCollectionSimTracksters>("recoToSim");

  TSCollectionToken_ = consumes<ticl::TracksterCollection>(pset.getParameter<edm::InputTag>("label_tst"));
  SimTSCollectionToken_ = consumes<ticl::TracksterCollection>(pset.getParameter<edm::InputTag>("label_simTst"));
  LCCollectionToken_ = consumes<reco::CaloClusterCollection>(pset.getParameter<edm::InputTag>("label_lcl"));
  associatorToken_ = consumes<ticl::TracksterToSimTracksterAssociator>(pset.getParameter<edm::InputTag>("associator"));
}

TSToSimTSAssociatorEDProducer::~TSToSimTSAssociatorEDProducer() {}

//
// member functions
//

// ------------ method called to produce the data  ------------
void TSToSimTSAssociatorEDProducer::produce(edm::StreamID, edm::Event &iEvent, const edm::EventSetup &iSetup) const {
  using namespace edm;

  edm::Handle<ticl::TracksterToSimTracksterAssociator> theAssociator;
  iEvent.getByToken(associatorToken_, theAssociator);

  Handle<ticl::TracksterCollection> TSCollection;
  iEvent.getByToken(TSCollectionToken_, TSCollection);

  Handle<ticl::TracksterCollection> SimTSCollection;
  iEvent.getByToken(SimTSCollectionToken_, SimTSCollection);

  Handle<reco::CaloClusterCollection> LCCollection;
  iEvent.getByToken(LCCollectionToken_, LCCollection);

  // associate TS and SimTS
  LogTrace("AssociatorValidator") << "Calling associateRecoToSim method\n";
  ticl::RecoToSimCollectionSimTracksters recSimColl =
      theAssociator->associateRecoToSim(TSCollection, LCCollection, SimTSCollection);

  LogTrace("AssociatorValidator") << "Calling associateSimToReco method\n";
  ticl::SimToRecoCollectionSimTracksters simRecColl =
      theAssociator->associateSimToReco(TSCollection, LCCollection, SimTSCollection);

  auto rts = std::make_unique<ticl::RecoToSimCollectionSimTracksters>(recSimColl);
  auto str = std::make_unique<ticl::SimToRecoCollectionSimTracksters>(simRecColl);

  iEvent.put(std::move(rts), "recoToSim");
  iEvent.put(std::move(str), "simToReco");
}

// define this as a plug-in
DEFINE_FWK_MODULE(TSToSimTSAssociatorEDProducer);
