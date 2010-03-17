// -*- C++ -*-
//
// Package:    L1TrackStubProducer
// Class:      L1TrackStubProducer
// 
/**\class L1TrackStubProducer L1TrackStubProducer.cc L1TriggerOffline/L1TrackStubProducer/src/L1TrackStubProducer.cc

 Description: Produce L1 trigger track stubs

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Jim Brooke
//         Created:  Sun Oct 14 21:13:26 CEST 2007
// $Id: TrackTriggerPrimitiveProducer.cc,v 1.2 2010/02/03 09:46:37 arose Exp $
//
//


// system include files
#include <memory>

// framework include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

// user includes
#include "SimDataFormats/SLHC/interface/TrackTriggerCollections.h"

//
// class decleration
//

class TrackTriggerPrimitiveProducer : public edm::EDProducer {
public:
  explicit TrackTriggerPrimitiveProducer(const edm::ParameterSet&);
  ~TrackTriggerPrimitiveProducer();
  
private:
  virtual void beginJob(const edm::EventSetup&) ;
  virtual void produce(edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;
  
  // ----------member data ---------------------------  
  edm::InputTag inputTag_;

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
TrackTriggerPrimitiveProducer::TrackTriggerPrimitiveProducer(const edm::ParameterSet& iConfig)
{
   //register your products
   produces<TrackTriggerPrimitiveCollection>();

   // input collection
   inputTag_ = iConfig.getParameter<edm::InputTag>("inputTag");

}


TrackTriggerPrimitiveProducer::~TrackTriggerPrimitiveProducer()
{
 
}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
TrackTriggerPrimitiveProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;

   // get hits
   Handle<TrackTriggerHitCollection> hits;
   iEvent.getByLabel(inputTag_, hits);

   // output collection
   std::auto_ptr<TrackTriggerPrimitiveCollection> stubs( new TrackTriggerPrimitiveCollection() );

   /// INSERT CODE HERE TO PRODUCE PRIMITIVES

   // put output into Event
   iEvent.put( stubs );

}

// ------------ method called once each job just before starting event loop  ------------
void 
TrackTriggerPrimitiveProducer::beginJob(const edm::EventSetup&)
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
TrackTriggerPrimitiveProducer::endJob() {
}

//define this as a plug-in
DEFINE_FWK_MODULE(TrackTriggerPrimitiveProducer);

