// -*- C++ -*-
//
// Package:    VertexConstraintProducer
// Class:      VertexConstraintProducer
//
/**\class VertexConstraintProducer VertexConstraintProducer.cc RecoTracker/ConstraintProducerTest/src/VertexConstraintProducer.cc

Description: <one line class summary>

Implementation:
<Notes on implementation>
*/
//
// Original Author:  Giuseppe Cerati
//         Created:  Tue Jul 10 15:05:02 CEST 2007
//
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "TrackingTools/PatternTools/interface/TrackConstraintAssociation.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

//
// class decleration
//

class VertexConstraintProducer : public edm::global::EDProducer<> {
public:
  explicit VertexConstraintProducer(const edm::ParameterSet&);
  ~VertexConstraintProducer() override = default;

private:
  void produce(edm::StreamID streamid, edm::Event&, const edm::EventSetup&) const override;

  // ----------member data ---------------------------
  const edm::InputTag srcTrkTag_;
  edm::EDGetTokenT<reco::TrackCollection> trkToken_;

  const edm::InputTag srcVtxTag_;
  edm::EDGetTokenT<reco::VertexCollection> vtxToken_;
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
VertexConstraintProducer::VertexConstraintProducer(const edm::ParameterSet& iConfig)
    : srcTrkTag_(iConfig.getParameter<edm::InputTag>("srcTrk")),
      srcVtxTag_(iConfig.getParameter<edm::InputTag>("srcVtx")) {
  //declare the consumes
  trkToken_ = consumes<reco::TrackCollection>(edm::InputTag(srcTrkTag_));
  vtxToken_ = consumes<reco::VertexCollection>(edm::InputTag(srcVtxTag_));

  //register your products
  produces<std::vector<VertexConstraint> >();
  produces<TrackVtxConstraintAssociationCollection>();

  //now do what ever other initialization is needed
}

//
// member functions
//

// ------------ method called to produce the data  ------------
void VertexConstraintProducer::produce(edm::StreamID streamid,
                                       edm::Event& iEvent,
                                       const edm::EventSetup& iSetup) const {
  using namespace edm;

  Handle<reco::TrackCollection> theTCollection;
  iEvent.getByToken(trkToken_, theTCollection);

  Handle<reco::VertexCollection> theVertexHandle;
  iEvent.getByToken(vtxToken_, theVertexHandle);

  edm::RefProd<std::vector<VertexConstraint> > rPairs = iEvent.getRefBeforePut<std::vector<VertexConstraint> >();
  std::unique_ptr<std::vector<VertexConstraint> > pairs(new std::vector<VertexConstraint>);
  std::unique_ptr<TrackVtxConstraintAssociationCollection> output(
      new TrackVtxConstraintAssociationCollection(theTCollection, rPairs));

  int index = 0;

  //primary vertex extraction

  if (!theVertexHandle->empty()) {
    const reco::Vertex& pv = theVertexHandle->front();
    for (reco::TrackCollection::const_iterator i = theTCollection->begin(); i != theTCollection->end(); i++) {
      VertexConstraint tmp(GlobalPoint(pv.x(), pv.y(), pv.z()),
                           GlobalError(pv.covariance(0, 0),
                                       pv.covariance(1, 0),
                                       pv.covariance(1, 1),
                                       pv.covariance(2, 0),
                                       pv.covariance(2, 1),
                                       pv.covariance(2, 2)));
      pairs->push_back(tmp);
      output->insert(reco::TrackRef(theTCollection, index), edm::Ref<std::vector<VertexConstraint> >(rPairs, index));
      index++;
    }
  }

  iEvent.put(std::move(pairs));
  iEvent.put(std::move(output));
}

//define this as a plug-in
DEFINE_FWK_MODULE(VertexConstraintProducer);
