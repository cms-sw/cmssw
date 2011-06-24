// -*- C++ -*-
//
// Package:    PFCandidateMixer
// Class:      PFCandidateMixer
// 
/**\class PFCandidateMixer PFCandidateMixer.cc MyAna/PFCandidateMixer/src/PFCandidateMixer.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Tomasz Maciej Frueboes
//         Created:  Wed Dec  9 16:14:56 CET 2009
// $Id: PFCandidateMixer.cc,v 1.4 2011/06/24 12:21:57 fruboes Exp $
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
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/Math/interface/deltaR.h"

//
// class decleration
//

class PFCandidateMixer : public edm::EDProducer {
   public:
      explicit PFCandidateMixer(const edm::ParameterSet&);
      ~PFCandidateMixer();

   private:
      virtual void beginJob() ;
      virtual void produce(edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;
      
      edm::InputTag _col1;
      edm::InputTag _col2;
      edm::InputTag _trackCol;
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
PFCandidateMixer::PFCandidateMixer(const edm::ParameterSet& iConfig):
   _col1(iConfig.getUntrackedParameter<edm::InputTag>("col1") ),
   _col2(iConfig.getUntrackedParameter<edm::InputTag>("col2") ),
  _trackCol(iConfig.getUntrackedParameter<edm::InputTag>("trackCol") )
{

   produces< std::vector< reco::PFCandidate >  >(); 

}

PFCandidateMixer::~PFCandidateMixer()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
PFCandidateMixer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;
   using namespace reco;

   Handle< std::vector<reco::Track> > trackCol;
   iEvent.getByLabel( _trackCol, trackCol);

 
   std::vector< Handle<PFCandidateCollection> > colVec; 
   Handle<PFCandidateCollection> pfIn1;
   Handle<PFCandidateCollection> pfIn2;
   iEvent.getByLabel(_col1,pfIn1);
   iEvent.getByLabel(_col2,pfIn2);
   
   colVec.push_back(pfIn1);
   colVec.push_back(pfIn2);

   std::auto_ptr<std::vector< reco::PFCandidate > > pOut(new std::vector< reco::PFCandidate  > );
   
   std::vector< Handle<PFCandidateCollection> >::iterator itCol= colVec.begin();
   std::vector< Handle<PFCandidateCollection> >::iterator itColE= colVec.end();

   int iCol = 0;
   for (;itCol!=itColE; ++itCol){
     if (!itCol->isValid()) {
       std::cout << "Whoops!" << std::endl;
     }
     PFCandidateConstIterator it = (*itCol)->begin();
     PFCandidateConstIterator itE = (*itCol)->end();
     for (;it!=itE;++it) {
       PFCandidate cand(*it);
       size_t i = 0;
       bool found = false;
       double minDR = 9999.;
       int iMinDr = -1;
       if (it->trackRef().isNonnull()) {
         for ( i = 0 ; i < trackCol->size(); ++i){
           if ( reco::deltaR( *(it->trackRef()), trackCol->at(i) )<0.001 ) {
                found = true;
                break; 
           }
           double dr = reco::deltaR( *(it->trackRef()), trackCol->at(i) );
           if ( dr < minDR) {
              iMinDr = i;
              minDR = dr;
           } 
         } 
       } 
       if ( found ){ // ref was found, overwrite in PFCand
         reco::TrackRef trref(trackCol,i);
         cand.setTrackRef(trref);
         //std::cout << " YY track ok"<<std::endl;

       } else { // keep orginall ref
         if (it->trackRef().isNonnull()) {
           std::cout << " XXXXXXXXXXX track not found " 
                 << " col " << iCol
                 << " ch " << it->charge()
                 << " id " << it->pdgId() 
                 << " pt " << it->pt() 
                 << " track: eta " << it->trackRef()->eta()
                 << " pt:  " << it->trackRef()->pt()
                 << " charge:  " << it->trackRef()->charge()
                 <<  std::endl;
           std::cout << " minDR=" << minDR << std::endl; 
           if ( iMinDr >= 0 ) {
                std::cout 
                     << " closest track pt=" << trackCol->at(iMinDr).pt()
                     << " ch=" << trackCol->at(iMinDr).charge()
                     <<  std::endl; 
           } 
           edm::Provenance prov=iEvent.getProvenance(it->trackRef().id());
           edm::InputTag tag(prov.moduleLabel(),  prov.productInstanceName(),   prov.processName());
           std::cout << " trackref in PFCand came from: "   << tag.encode() << std::endl;
 
           
         }
       }
       pOut->push_back(cand);
     }
     ++iCol;
   }
    

   iEvent.put(pOut);

   
}

// ------------ method called once each job just before starting event loop  ------------
void 
PFCandidateMixer::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
PFCandidateMixer::endJob() {
}

//define this as a plug-in
DEFINE_FWK_MODULE(PFCandidateMixer);
