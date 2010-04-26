// -*- C++ -*-
//
// Package:    ZmumuPFEmbedder
// Class:      ZmumuPFEmbedder
// 
/**\class ZmumuPFEmbedder ZmumuPFEmbedder.cc MyAna/ZmumuPFEmbedder/src/ZmumuPFEmbedder.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Tomasz Maciej Frueboes
//         Created:  Wed Dec  9 16:14:56 CET 2009
// $Id: ZmumuPFEmbedder.cc,v 1.3 2010/04/22 15:01:19 fruboes Exp $
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
#include <DataFormats/RecoCandidate/interface/RecoCandidate.h>
#include <DataFormats/Candidate/interface/CompositeRefCandidate.h>
#include <DataFormats/MuonReco/interface/Muon.h>

#include <DataFormats/ParticleFlowCandidate/interface/PFCandidate.h>
#include <DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h>

#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/TrackCandidate/interface/TrackCandidateCollection.h"

#include <DataFormats/Math/interface/deltaR.h>
//
// class decleration
//

class ZmumuPFEmbedder : public edm::EDProducer {
   public:
      explicit ZmumuPFEmbedder(const edm::ParameterSet&);
      ~ZmumuPFEmbedder();

   private:
      virtual void beginJob() ;
      virtual void produce(edm::Event&, const edm::EventSetup&);
      void produceTrackColl(edm::Event&, const std::vector< reco::Muon > * toBeAdded );
      virtual void endJob() ;
      
      edm::InputTag _tracks;
      edm::InputTag _selectedMuons;
      
      // ----------member data ---------------------------
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
ZmumuPFEmbedder::ZmumuPFEmbedder(const edm::ParameterSet& iConfig)
  : _tracks(iConfig.getParameter<edm::InputTag>("tracks")),
    _selectedMuons(iConfig.getParameter<edm::InputTag>("selectedMuons"))
{

   //register your products
   // produces< std::vector< reco::Muon >  >("zMusExtracted"); // 
   produces<reco::TrackCollection>();
   produces< std::vector< reco::PFCandidate >  >("forMixing");

   
}

ZmumuPFEmbedder::~ZmumuPFEmbedder()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
ZmumuPFEmbedder::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;
   using namespace reco;

   Handle<PFCandidateCollection> pfIn;
   iEvent.getByLabel("particleFlow",pfIn);

   //std::auto_ptr<std::vector< reco::Muon > > pOut(new std::vector< reco::Muon  > );
   std::auto_ptr<std::vector< reco::PFCandidate > > forMix(new std::vector< reco::PFCandidate  > );
   
   
   //get selected muons
   Handle< std::vector< reco::Muon > > selectedZMuonsHandle;
   iEvent.getByLabel(_selectedMuons, selectedZMuonsHandle);
   const std::vector< reco::Muon > * toBeAdded = selectedZMuonsHandle.product();
   //std::vector< reco::Muon > toBeAdded;
   // TODO - check col size
   reco::Muon l1 = *(toBeAdded->begin());
   reco::Muon l2 = *(toBeAdded->rbegin());
   
   // iterate over pfcandidates, make copy if its not a selected muon
   PFCandidateConstIterator it = pfIn->begin();
   PFCandidateConstIterator itE = pfIn->end();

   for (;it!=itE;++it) {
     int pdg = std::abs( it->pdgId() );
     double dr1 = reco::deltaR( *it, l1); 
     double dr2 = reco::deltaR( *it, l2); 

     if ( pdg == 13 && (dr1 < 0.001 || dr2 < 0.002 ) ) { // it is a selected muon, do not copy
       
       
     } else {
       forMix->push_back(*it);
     }
   }



   produceTrackColl(iEvent, toBeAdded);
   iEvent.put(forMix, "forMixing");


   
}


// produces clean track collection wo muon tracks.
void ZmumuPFEmbedder::produceTrackColl(edm::Event & iEvent, const std::vector< reco::Muon > * toBeAdded )
{
   edm::Handle<reco::TrackCollection> tks;
   iEvent.getByLabel( _tracks, tks);

   std::auto_ptr< reco::TrackCollection  > newCol(new reco::TrackCollection );

   double epsilon = 0.00001;
   int nMatched = 0;
   
   for ( reco::TrackCollection::const_iterator it = tks->begin() ; it != tks->end(); ++it) 
   {
     bool ok = true;
     for ( std::vector< reco::Muon >::const_iterator itTBA = toBeAdded->begin();
                                                            itTBA != toBeAdded->end();
                                                            ++itTBA)
     {
       reco::TrackRef track = itTBA->innerTrack();
       /*
       if (!track.isNonnull()) {
         continue;
       }*/
       double dr = reco::deltaR( *it, *track);
       //std::cout << "TTTT " << dr << std::endl;
       if (dr < epsilon) {
         ++ nMatched;
         ok = false;
       }
     }
     if (ok)  newCol->push_back(*it);  
   }
   if (nMatched!=2) std::cout << "TTT ARGGGHGH " << nMatched << std::endl;

   iEvent.put(newCol);

}
// ------------ method called once each job just before starting event loop  ------------
void 
ZmumuPFEmbedder::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
ZmumuPFEmbedder::endJob() {
}

//define this as a plug-in
DEFINE_FWK_MODULE(ZmumuPFEmbedder);
