// -*- C++ -*-
//
// Package:    V0Producer
// Class:      V0Producer
// 
/**\class V0Producer V0Producer.cc MyProducers/V0Producer/src/V0Producer.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Brian Drell
//         Created:  Fri May 18 22:57:40 CEST 2007
// $Id: V0Producer.cc,v 1.11 2009/12/18 20:45:10 wmtan Exp $
//
//


// system include files
#include <memory>

#include "RecoVertex/V0Producer/interface/V0Producer.h"

// Constructor
V0Producer::V0Producer(const edm::ParameterSet& iConfig) :
  theParams(iConfig) {

   // Registering V0 Collections
  //produces<reco::VertexCollection>("Kshort");
  //produces<reco::VertexCollection>("Lambda");
  //produces<reco::VertexCollection>("LambdaBar");

  // Trying this with Candidates instead of the simple reco::Vertex
  produces< reco::VertexCompositeCandidateCollection >("Kshort");
  produces< reco::VertexCompositeCandidateCollection >("Lambda");
  //produces< reco::VertexCompositeCandidateCollection >("LambdaBar");

}

// (Empty) Destructor
V0Producer::~V0Producer() {

}


//
// Methods
//

// Producer Method
void V0Producer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
   using namespace edm;

   // Create V0Fitter object which reconstructs the vertices and creates
   //  (and contains) collections of Kshorts, Lambda0s
   V0Fitter theVees(theParams, iEvent, iSetup);

   // Create auto_ptr for each collection to be stored in the Event
   std::auto_ptr< reco::VertexCompositeCandidateCollection > 
     kShortCandidates( new reco::VertexCompositeCandidateCollection );
   kShortCandidates->reserve( theVees.getKshorts().size() ); 

   std::auto_ptr< reco::VertexCompositeCandidateCollection >
     lambdaCandidates( new reco::VertexCompositeCandidateCollection );
   lambdaCandidates->reserve( theVees.getLambdas().size() );

   std::copy( theVees.getKshorts().begin(),
	      theVees.getKshorts().end(),
	      std::back_inserter(*kShortCandidates) );
   std::copy( theVees.getLambdas().begin(),
	      theVees.getLambdas().end(),
	      std::back_inserter(*lambdaCandidates) );

   // Write the collections to the Event
   iEvent.put( kShortCandidates, std::string("Kshort") );
   iEvent.put( lambdaCandidates, std::string("Lambda") );

}


//void V0Producer::beginJob() {
void V0Producer::beginJob() {
}


void V0Producer::endJob() {
}

//define this as a plug-in
#include "FWCore/PluginManager/interface/ModuleDef.h"

DEFINE_FWK_MODULE(V0Producer);
//DEFINE_FWK_MODULE(V0finder);
