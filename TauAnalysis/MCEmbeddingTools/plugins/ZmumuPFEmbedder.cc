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
// $Id: ZmumuPFEmbedder.cc,v 1.12 2012/01/27 13:17:12 aburgmei Exp $
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
#include "DataFormats/Common/interface/View.h"
#include <DataFormats/RecoCandidate/interface/RecoCandidate.h>
#include <DataFormats/Candidate/interface/CompositeRefCandidate.h>
#include <DataFormats/MuonReco/interface/Muon.h>

#include <DataFormats/ParticleFlowCandidate/interface/PFCandidate.h>
#include <DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h>

#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/TrackCandidate/interface/TrackCandidateCollection.h"

#include "DataFormats/Candidate/interface/CompositeCandidate.h"
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
      void producePFCandColl(edm::Event&, const std::vector< reco::Particle::LorentzVector > * toBeAdded );
      void produceTrackColl(edm::Event&, const std::vector< reco::Particle::LorentzVector > * toBeAdded );
      virtual void endJob() ;
      
      edm::InputTag _tracks;
      edm::InputTag _selectedMuons;
      bool _useCombinedCandidate;

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
    _useCombinedCandidate(iConfig.getUntrackedParameter<bool>("useCombinedCandidate", false))
{

   //register your products
   // produces< std::vector< reco::Muon >  >("zMusExtracted"); // 
   produces<reco::TrackCollection>("tracks");
   produces< std::vector< reco::PFCandidate >  >("pfCands");

   
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
   std::vector< reco::Particle::LorentzVector > toBeAdded;
   
   if (_useCombinedCandidate)
   {
      edm::Handle< std::vector< reco::CompositeCandidate > > combCandidatesHandle;
      if (iEvent.getByLabel(_selectedMuons, combCandidatesHandle) && combCandidatesHandle->size()>0)
         for (size_t idx = 0; idx < combCandidatesHandle->at(0).numberOfDaughters(); ++idx)			// use only the first combined candidate
            toBeAdded.push_back(combCandidatesHandle->at(0).daughter(idx)->p4());
   }
   else
   {
      edm::Handle< edm::View< reco::Muon > > selectedZMuonsHandle;
      if (iEvent.getByLabel(_selectedMuons, selectedZMuonsHandle))
        for (size_t idx = 0; idx < selectedZMuonsHandle->size(); ++idx)
          toBeAdded.push_back(selectedZMuonsHandle->at(idx).p4());
   }

   if (toBeAdded.size() == 0)
      return;

   producePFCandColl(iEvent, &toBeAdded);
   produceTrackColl(iEvent, &toBeAdded);
}

// produces clean PFCandidate collection w/o muon candidates.
void ZmumuPFEmbedder::producePFCandColl(edm::Event & iEvent, const std::vector< reco::Particle::LorentzVector > * toBeAdded )
{
   edm::Handle<reco::PFCandidateCollection> pfIn;
   iEvent.getByLabel("particleFlow",pfIn);

   //std::vector< reco::Muon > toBeAdded;
   // TODO - check col size
   //reco::Muon l1 = *(toBeAdded->begin());
   //reco::Muon l2 = *(toBeAdded->rbegin());
   
   std::auto_ptr<std::vector< reco::PFCandidate > > newCol(new std::vector< reco::PFCandidate  > );   
   
   //get selected muons
   // iterate over pfcandidates, make copy if its not a selected muon
   reco::PFCandidateConstIterator it = pfIn->begin();
   reco::PFCandidateConstIterator itE = pfIn->end();

   for (;it!=itE;++it) {
     int pdg = std::abs( it->pdgId() );
     double minDR = 10;
     /* 
     double dr1 = reco::deltaR( *it, l1); 
     double dr2 = reco::deltaR( *it, l2); 
     */
     std::vector< reco::Particle::LorentzVector >::const_iterator itSelectedMu = toBeAdded->begin();
     std::vector< reco::Particle::LorentzVector >::const_iterator itSelectedMuE = toBeAdded->end();
     for (; itSelectedMu != itSelectedMuE; ++itSelectedMu ){
       double dr = reco::deltaR( *it, *itSelectedMu);
       if (dr < minDR)  minDR = dr;
     }

     //if ( pdg == 13 && (dr1 < 0.001 || dr2 < 0.002 ) ) { // it is a selected muon, do not copy
     if ( pdg == 13 && (minDR < 0.001 ) ) { // it is a selected muon, do not copy
       
       
     } else {
       newCol->push_back(*it);
     }
   }

   iEvent.put(newCol, "pfCands");
}

// produces clean track collection w/o muon tracks.
void ZmumuPFEmbedder::produceTrackColl(edm::Event & iEvent, const std::vector< reco::Particle::LorentzVector > * toBeAdded )
{
   edm::Handle<reco::TrackCollection> tks;
   iEvent.getByLabel( _tracks, tks);

   std::auto_ptr< reco::TrackCollection  > newCol(new reco::TrackCollection );

   double epsilon = 0.00001;
   unsigned int nMatched = 0;
   
   for ( reco::TrackCollection::const_iterator it = tks->begin() ; it != tks->end(); ++it) 
   {
     bool ok = true;
     for ( std::vector< reco::Particle::LorentzVector >::const_iterator itTBA = toBeAdded->begin();
                                                            itTBA != toBeAdded->end();
                                                            ++itTBA)
     {
       double dr = reco::deltaR( *it, *itTBA);
       //std::cout << "TTTT " << dr << std::endl;
       if (dr < epsilon) {
         ++ nMatched;
         ok = false;
       }
     }
     if (ok)  newCol->push_back(*it);  
   }
   if (nMatched!=toBeAdded->size() ) std::cout << "TTT ARGGGHGH " << nMatched << std::endl;

   iEvent.put(newCol, "tracks");
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
