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
// $Id$
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
   _col2(iConfig.getUntrackedParameter<edm::InputTag>("col2") )
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
   for (;itCol!=itColE; ++itCol){
     if (!itCol->isValid()) {
       std::cout << "Whoops!" << std::endl;
     }
     PFCandidateConstIterator it = (*itCol)->begin();
     PFCandidateConstIterator itE = (*itCol)->end();
     for (;it!=itE;++it) {
       pOut->push_back(*it);
     }
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
