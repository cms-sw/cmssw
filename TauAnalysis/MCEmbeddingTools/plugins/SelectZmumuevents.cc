// -*- C++ -*-
//
// Package:    SelectZmumuevents
// Class:      SelectZmumuevents
// 
/**\class SelectZmumuevents SelectZmumuevents.cc MyAna/SelectZmumuevents/src/SelectZmumuevents.cc

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Tomasz Maciej Frueboes
//         Created:  Fri Dec 18 14:29:14 CET 2009
// $Id: SelectZmumuevents.cc,v 1.1 2010/03/17 16:14:10 fruboes Exp $
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDFilter.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <DataFormats/ParticleFlowCandidate/interface/PFCandidate.h>
#include <DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h>
//
// class declaration
//

class SelectZmumuevents : public edm::EDFilter {
   public:
      explicit SelectZmumuevents(const edm::ParameterSet&);
      ~SelectZmumuevents();

   private:
      virtual void beginJob() ;
      virtual bool filter(edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;
      
      // ----------member data ---------------------------
      double _etaMax;
      double _ptMin;
      edm::InputTag _pfColl;
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
SelectZmumuevents::SelectZmumuevents(const edm::ParameterSet& iConfig)
  : _etaMax(iConfig.getUntrackedParameter<double>("etaMax")),
    _ptMin(iConfig.getUntrackedParameter<double>("ptMin")),
    _pfColl(iConfig.getUntrackedParameter<edm::InputTag>("pfCol"))
{

}


SelectZmumuevents::~SelectZmumuevents()
{
 
   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)
}


//
// member functions
//

// ------------ method called on each new Event  ------------
bool
SelectZmumuevents::filter(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  using namespace edm;
  Handle< View<reco::Candidate> > genParts;
  iEvent.getByLabel(_pfColl, genParts);
  
  int cnt = 0;
  for( edm::View<reco::Candidate>::const_iterator it = genParts->begin(); it != genParts->end(); it++ ) 
  {
    if (std::abs(it->pdgId()) == 13 
        && std::abs(it->eta()) < _etaMax
        && it->pt() > _ptMin ) 
    {
      ++cnt; 
    }
  }

   return cnt>1;
}

// ------------ method called once each job just before starting event loop  ------------
void 
SelectZmumuevents::beginJob()
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
SelectZmumuevents::endJob() {
}

//define this as a plug-in
DEFINE_FWK_MODULE(SelectZmumuevents);
