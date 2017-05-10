// -*- C++ -*-
//
// Package:    BoostedJetMerger
// Class:      BoostedJetMerger
// 
// \class BoostedJetMerger BoostedJetMerger.cc TopQuarkAnalysis/TopPairBSM/plugins/BoostedJetMerger.cc
// Description: Class to "deswizzle" information from various pat::Jet collections.
//
// Original Author:  "Salvatore Rappoccio"
//         Created:  Thu May  1 11:37:48 CDT 2008
// $Id: BoostedJetMerger.cc,v 1.1 2013/03/07 20:13:55 srappocc Exp $
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
#include "DataFormats/PatCandidates/interface/Jet.h"

//
// class decleration
//


/// Predicate to use for find_if.
/// This checks whether a given edm::Ptr<reco::Candidate>
/// (as you would get from the reco::BasicJet daughters)
/// to see if it matches the original object ref of
/// another pat::Jet (which is to find the corrected / btagged
/// pat::Jet that corresponds to the subjet in question). 
struct FindCorrectedSubjet {
  // Input the daughter you're interested in checking
  FindCorrectedSubjet( edm::Ptr<reco::Candidate> const & da ) : 
    da_(da) {}

  // Predicate operator to compare an input pat::Jet to. 
  bool operator()( pat::Jet const & subjet ) const {
    edm::Ptr<reco::Candidate> subjetOrigRef = subjet.originalObjectRef();
    if ( da_ == subjetOrigRef ) {
      return true;
    }
    else return false;
  }

  edm::Ptr<reco::Candidate> da_;
};

class BoostedJetMerger : public edm::EDProducer {
   public:
      explicit BoostedJetMerger(const edm::ParameterSet&);
      ~BoostedJetMerger();

   private:
      virtual void beginJob(const edm::EventSetup&) ;
      virtual void produce(edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;
      
      // ----------member data ---------------------------

      // data labels
      edm::InputTag jetLabel_;
      edm::InputTag subjetLabel_;
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
BoostedJetMerger::BoostedJetMerger(const edm::ParameterSet& iConfig) :
  jetLabel_(iConfig.getParameter<edm::InputTag>("jetSrc")),
  subjetLabel_(iConfig.getParameter<edm::InputTag>("subjetSrc"))
{
  //register products
  produces<std::vector<pat::Jet> > ();
}


BoostedJetMerger::~BoostedJetMerger()
{
}


//
// member functions
//

// ------------ method called to produce the data  ------------
void
BoostedJetMerger::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{  

  std::auto_ptr< std::vector<pat::Jet> > outputs( new std::vector<pat::Jet> );
 
  edm::Handle< edm::View<pat::Jet> > jetHandle;
  edm::Handle< edm::View<pat::Jet> > subjetHandle;

  iEvent.getByLabel( jetLabel_, jetHandle );
  iEvent.getByLabel( subjetLabel_, subjetHandle ); 

  for ( edm::View<pat::Jet>::const_iterator ijetBegin = jetHandle->begin(),
	  ijetEnd = jetHandle->end(), ijet = ijetBegin; ijet != ijetEnd; ++ijet ) {
    
    outputs->push_back( *ijet );
    std::vector< edm::Ptr<reco::Candidate> > nextSubjets;

    for ( unsigned int isubjet = 0; isubjet < ijet->numberOfDaughters(); ++isubjet ) {
      edm::Ptr<reco::Candidate> const & subjet = ijet->daughterPtr(isubjet);
      edm::View<pat::Jet>::const_iterator ifound = find_if( subjetHandle->begin(),
							    subjetHandle->end(),
							    FindCorrectedSubjet(subjet) );
      if ( ifound != subjetHandle->end() ) {
	nextSubjets.push_back( subjetHandle->ptrAt( ifound - subjetHandle->begin() ) );

      }
    }
    outputs->back().clearDaughters();
    for ( std::vector< edm::Ptr<reco::Candidate> >::const_iterator nextSubjet = nextSubjets.begin(),
	    nextSubjetEnd = nextSubjets.end(); nextSubjet != nextSubjetEnd; ++nextSubjet ) {
      outputs->back().addDaughter( *nextSubjet );
    }

    
  }

  iEvent.put(outputs);

}

// ------------ method called once each job just before starting event loop  ------------
void 
BoostedJetMerger::beginJob(const edm::EventSetup&)
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
BoostedJetMerger::endJob() {
}

//define this as a plug-in
DEFINE_FWK_MODULE(BoostedJetMerger);
