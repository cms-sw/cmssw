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
// $Id: ZmumuPFEmbedder.cc,v 1.4 2010/04/26 17:44:33 fruboes Exp $
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
      bool _keepMuonTrack;
      
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
    _selectedMuons(iConfig.getParameter<edm::InputTag>("selectedMuons")),
    _keepMuonTrack(iConfig.getParameter<bool>("keepMuonTrack"))
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
   
   if (selectedZMuonsHandle->size() !=  2) {
     std::cout << "Warning - wrong col size\n";
   }
   
   
   
   reco::Muon l1 = *(toBeAdded->begin());
   reco::Muon l2 = *(toBeAdded->rbegin());
   
   /*
   std::cout << "will transform muons1: "
       << " " <<  l1.charge()
       << " " <<  l1.px() 
       << " " <<  l1.py()
       << " " <<  l1.pz()
       << " " <<  l1.energy()
       << " " << l1.vertex().x() 
       << " " << l1.vertex().y()
       << " " << l1.vertex().z() << std::endl; 
   
   std::cout << "will transform muons2: "
       << " " <<  l2.charge()
       << " " <<  l2.px() 
       << " " <<  l2.py()
       << " " <<  l2.pz()
       << " " <<  l2.energy()
       << " " <<  l2.vertex().x() 
       << " " <<  l2.vertex().y()
       << " " <<  l2.vertex().z() << std::endl; 
   */
   
   // iterate over pfcandidates, make copy if its not a selected muon
   PFCandidateConstIterator it = pfIn->begin();
   PFCandidateConstIterator itE = pfIn->end();

   PFCandidateConstIterator closestPFMuon1 = pfIn->end(); 
   PFCandidateConstIterator closestPFMuon2 = pfIn->end();
    
   PFCandidateConstIterator closestPFNotMuon1 = pfIn->end(); 
   PFCandidateConstIterator closestPFNotMuon2 = pfIn->end(); 

   double drMin1 = -1;
   double drMin2 = -1;
   
   double drMinNotMu1 = -1;
   double drMinNotMu2 = -1;

   
   
   for (;it!=itE;++it) {
     int pdg = std::abs( it->pdgId() );
     double dr1 = reco::deltaR( *it, l1); 
     double dr2 = reco::deltaR( *it, l2);

     
     
     if ( pdg != 13 ) {
       
       
       //forMix->push_back(*it);
       
       if (dr1 < dr2) { // assign to one
         if (drMinNotMu1 < 0 || dr1 < drMinNotMu1) {
           drMinNotMu1 = dr1;
           if (closestPFNotMuon1 != pfIn->end()) forMix->push_back(*closestPFNotMuon1);
           closestPFNotMuon1 = it;
         }
       } else { // closer to "2"
         if (drMinNotMu2 < 0 || dr2 < drMinNotMu2) {
           drMinNotMu2 = dr2;
           if (closestPFNotMuon2 != pfIn->end()) forMix->push_back(*closestPFNotMuon2);
           closestPFNotMuon2 = it;
         }
       } 

       
       
       
     } else {
        
       // assign first muon
       
       if (dr1 < dr2) { // assign to one
         if (drMin1 < 0 || dr1 < drMin1) {
           drMin1 = dr1;
           if (closestPFMuon1 != pfIn->end()) forMix->push_back(*closestPFMuon1);
           closestPFMuon1 = it;
         }
       } else { // closer to "2"
         if (drMin2 < 0 || dr2 < drMin2) {
           drMin2 = dr2;
           if (closestPFMuon2 != pfIn->end()) forMix->push_back(*closestPFMuon2);
           closestPFMuon2 = it;
         }
       } 
       

     
     }
     
     /*
     double dr1 = reco::deltaR( *it, l1); 
     double dr2 = reco::deltaR( *it, l2); 

     if ( pdg == 13 && (dr1 < 0.001 || dr2 < 0.002 ) ) { // it is a selected muon, do not copy
       ++nexcluded;
     } else {
       forMix->push_back(*it);
     }*/
   }


   //std::cout << " DRS: " << drMin1 << " " << drMin2 << std::endl;
   if (closestPFMuon1 == pfIn->end()) {
     std::cout << " Warning: didnt find 1st pfMuon. Will use cand with: "
         << closestPFNotMuon1->eta() << "," << closestPFNotMuon1->pt()
         << "(" << l1.eta() << "," << l1.pt()
         <<  std::endl << std::endl;
   } else {
     forMix->push_back(*closestPFNotMuon1);
   }

   if (closestPFMuon2 == pfIn->end()) {
     std::cout << " Warning: didnt find 2nd pfMuon. Will use cand with: "
         << closestPFNotMuon2->eta() << "," << closestPFNotMuon2->pt()
         << "(" << l2.eta() << "," << l2.pt()
         <<  std::endl << std::endl;
   } else {
     forMix->push_back(*closestPFNotMuon2);
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

       double dr = 100;
       if (!_keepMuonTrack) {
         reco::TrackRef track = itTBA->innerTrack();
         if (track.isNonnull()) {
            
           dr = reco::deltaR( *it, *track);
         } else {
           std::cout << " Track empty\n" ;
         }
       /*
       if (!track.isNonnull()) {
         continue;
       }*/
       }
       //std::cout << "TTTT " << dr << std::endl;
       if (dr < epsilon) {
         ++ nMatched;
         ok = false;
       }
     }
     if (ok)  newCol->push_back(*it);  
   }
   if (nMatched!=2 && !_keepMuonTrack) std::cout << "TTT ARGGGHGH " << nMatched << std::endl;

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
