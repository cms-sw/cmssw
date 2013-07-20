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
// $Id: VertexConstraintProducer.cc,v 1.7 2013/02/27 14:58:17 muzaffar Exp $
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

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

class VertexConstraintProducer: public edm::EDProducer {
public:
  explicit VertexConstraintProducer(const edm::ParameterSet&);
  ~VertexConstraintProducer();

private:
  virtual void produce(edm::Event&, const edm::EventSetup&) override;
  virtual void endJob() ;
      
  // ----------member data ---------------------------
  const edm::ParameterSet iConfig_;
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
VertexConstraintProducer::VertexConstraintProducer(const edm::ParameterSet& iConfig) : iConfig_(iConfig)
{
  //register your products
  produces<std::vector<VertexConstraint> >();
  produces<TrackVtxConstraintAssociationCollection>();

  //now do what ever other initialization is needed
}


VertexConstraintProducer::~VertexConstraintProducer()
{
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
}


//
// member functions
//

// ------------ method called to produce the data  ------------
void VertexConstraintProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace edm;
  InputTag srcTag = iConfig_.getParameter<InputTag>("srcTrk");
  Handle<reco::TrackCollection> theTCollection;
  iEvent.getByLabel(srcTag,theTCollection);

  std::auto_ptr<std::vector<VertexConstraint> > pairs(new std::vector<VertexConstraint>);
  std::auto_ptr<TrackVtxConstraintAssociationCollection> output(new TrackVtxConstraintAssociationCollection);
  edm::RefProd<std::vector<VertexConstraint> > rPairs = iEvent.getRefBeforePut<std::vector<VertexConstraint> >();

  int index = 0;
  
  //primary vertex extraction

  InputTag srcTag2 = iConfig_.getParameter<InputTag>("srcVtx");
  edm::Handle<reco::VertexCollection> primaryVertexHandle;
  iEvent.getByLabel(srcTag2,primaryVertexHandle);
  if(primaryVertexHandle->size()>0){
  reco::Vertex pv;
    pv = primaryVertexHandle->front();
    for (reco::TrackCollection::const_iterator i=theTCollection->begin(); i!=theTCollection->end();i++) {
      VertexConstraint tmp(GlobalPoint(pv.x(),pv.y(),pv.z()),GlobalError(pv.xError(),0,pv.yError(),0,0,pv.zError()));  
      pairs->push_back(tmp);
      output->insert(reco::TrackRef(theTCollection,index), edm::Ref<std::vector<VertexConstraint> >(rPairs,index) );
      index++;
    }

  }

  iEvent.put(pairs);
  iEvent.put(output);
}

// ------------ method called once each job just after ending the event loop  ------------
void VertexConstraintProducer::endJob() {
}

//define this as a plug-in
DEFINE_FWK_MODULE(VertexConstraintProducer);
