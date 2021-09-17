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

#include "SimDataFormats/Associations/interface/TracksterToSimClusterAssociator.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/HGCalReco/interface/Trackster.h"

#include "FWCore/Utilities/interface/EDGetToken.h"

//
// class decleration
//

class TSToSCAssociatorEDProducer : public edm::global::EDProducer<> {
public:
  explicit TSToSCAssociatorEDProducer(const edm::ParameterSet &);
  ~TSToSCAssociatorEDProducer() override;

private:
  void produce(edm::StreamID, edm::Event &, const edm::EventSetup &) const override;

  edm::EDGetTokenT<SimClusterCollection> SCCollectionToken_;
  edm::EDGetTokenT<ticl::TracksterCollection> TSCollectionToken_;
  edm::EDGetTokenT<reco::CaloClusterCollection> LCCollectionToken_;
  edm::EDGetTokenT<hgcal::TracksterToSimClusterAssociator> associatorToken_;
};

TSToSCAssociatorEDProducer::TSToSCAssociatorEDProducer(const edm::ParameterSet &pset) {
  produces<hgcal::SimToRecoCollectionTracksters>();
  produces<hgcal::RecoToSimCollectionTracksters>();

  SCCollectionToken_ = consumes<SimClusterCollection>(pset.getParameter<edm::InputTag>("label_scl"));
  TSCollectionToken_ = consumes<ticl::TracksterCollection>(pset.getParameter<edm::InputTag>("label_tst"));
  LCCollectionToken_ = consumes<reco::CaloClusterCollection>(pset.getParameter<edm::InputTag>("label_lcl"));
  associatorToken_ = consumes<hgcal::TracksterToSimClusterAssociator>(pset.getParameter<edm::InputTag>("associator"));
}

TSToSCAssociatorEDProducer::~TSToSCAssociatorEDProducer() {}

//
// member functions
//

// ------------ method called to produce the data  ------------
void TSToSCAssociatorEDProducer::produce(edm::StreamID, edm::Event &iEvent, const edm::EventSetup &iSetup) const {
  using namespace edm;

  edm::Handle<hgcal::TracksterToSimClusterAssociator> theAssociator;
  iEvent.getByToken(associatorToken_, theAssociator);

  Handle<SimClusterCollection> SCCollection;
  iEvent.getByToken(SCCollectionToken_, SCCollection);

  Handle<ticl::TracksterCollection> TSCollection;
  iEvent.getByToken(TSCollectionToken_, TSCollection);

  Handle<reco::CaloClusterCollection> LCCollection;
  iEvent.getByToken(LCCollectionToken_, LCCollection);

  // associate TS and SC
  LogTrace("AssociatorValidator") << "Calling associateRecoToSim method\n";
  hgcal::RecoToSimCollectionTracksters recSimColl =
      theAssociator->associateRecoToSim(TSCollection, LCCollection, SCCollection);

  LogTrace("AssociatorValidator") << "Calling associateSimToReco method\n";
  hgcal::SimToRecoCollectionTracksters simRecColl =
      theAssociator->associateSimToReco(TSCollection, LCCollection, SCCollection);

  auto rts = std::make_unique<hgcal::RecoToSimCollectionTracksters>(recSimColl);
  auto str = std::make_unique<hgcal::SimToRecoCollectionTracksters>(simRecColl);

  iEvent.put(std::move(rts));
  iEvent.put(std::move(str));
}

// define this as a plug-in
DEFINE_FWK_MODULE(TSToSCAssociatorEDProducer);
