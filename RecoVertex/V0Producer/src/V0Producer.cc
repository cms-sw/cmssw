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
// $Id$
//
//


// system include files
#include <memory>

#include "RecoVertex/V0Producer/interface/V0Producer.h"

// Constructor
V0Producer::V0Producer(const edm::ParameterSet& iConfig) :
  theParams(iConfig),
  trackRecoAlgo(iConfig.getUntrackedParameter("trackRecoAlgorithm", 
			    std::string("ctfWithMaterialTracks"))) {

  // Get parameter to find whether we need to use the refitter
  // Defaults to 1 (true)
  useSmoothedTrax = iConfig.getUntrackedParameter("useSmoothing", 1);
  storeSmoothedTrax = iConfig.getUntrackedParameter(
			       "storeSmoothedTracksInRecoVertex", 1);
  chi2Cut = iConfig.getUntrackedParameter("chi2Cut", 1.);
  rVtxCut = iConfig.getUntrackedParameter("rVtxCut", 0.1);
  vtxSigCut = iConfig.getUntrackedParameter("vtxSignificanceCut", 22.);
  collinCut = iConfig.getUntrackedParameter("collinearityCut", 0.02);
  kShortMassCut = iConfig.getUntrackedParameter("kShortMassCut", 0.25);
  lambdaMassCut = iConfig.getUntrackedParameter("lambdaMassCut", 0.25);

   // Registering V0 Collections
  //produces<reco::VertexCollection>("Kshort");
  //produces<reco::VertexCollection>("Lambda");
  //produces<reco::VertexCollection>("LambdaBar");

  // Trying this with Candidates instead of the simple reco::Vertex
  produces< std::vector<reco::V0Candidate> >("Kshort");
  produces< std::vector<reco::V0Candidate> >("Lambda");
  produces< std::vector<reco::V0Candidate> >("LambdaBar");

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
   //  (and contains) collections of Kshorts, Lambda0s, and Lambda0Bars
   V0Fitter theVees(iEvent, iSetup, trackRecoAlgo, 
		    useSmoothedTrax, storeSmoothedTrax,
		    chi2Cut, rVtxCut, vtxSigCut, collinCut, kShortMassCut,
		    lambdaMassCut);

   // Create auto_ptr for each collection to be stored in the Event
   /*std::auto_ptr<reco::VertexCollection> k0sOut(new
	       reco::VertexCollection( theVees.getKshortCollection() ));
   std::auto_ptr<reco::VertexCollection> L0Out(new
	       reco::VertexCollection( theVees.getLambdaCollection() ));
   std::auto_ptr<reco::VertexCollection> L0BarOut(new
   reco::VertexCollection( theVees.getLambdaBarCollection() ));*/

   std::auto_ptr< std::vector<reco::V0Candidate> > 
     kShortCandidates( new std::vector<reco::V0Candidate>( 
						       theVees.getKshorts()) );
   std::auto_ptr< std::vector<reco::V0Candidate> >
     lambdaCandidates( new std::vector<reco::V0Candidate>(
						       theVees.getLambdas()) );
   std::auto_ptr< std::vector<reco::V0Candidate> >
     lambdaBarCandidates( new std::vector<reco::V0Candidate>(
						    theVees.getLambdaBars()) );

   // Write the collections to the Event
   //iEvent.put( k0sOut, std::string("Kshort") );
   //iEvent.put( L0Out, std::string("Lambda") );
   //iEvent.put( L0BarOut, std::string("LambdaBar") );

   iEvent.put( kShortCandidates, std::string("Kshort") );
   iEvent.put( lambdaCandidates, std::string("Lambda") );
   iEvent.put( lambdaBarCandidates, std::string("LambdaBar") );

}


void V0Producer::beginJob(const edm::EventSetup&) {
}


void V0Producer::endJob() {
}

//define this as a plug-in
#include "FWCore/PluginManager/interface/ModuleDef.h"
DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(V0Producer);
//DEFINE_FWK_MODULE(V0finder);
