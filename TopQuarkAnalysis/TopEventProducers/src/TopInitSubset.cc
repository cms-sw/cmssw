#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "AnalysisDataFormats/TopObjects/interface/TtGenEvent.h"
#include "TopQuarkAnalysis/TopEventProducers/interface/TopInitSubset.h"

using namespace std;
using namespace reco;

TopInitSubset::TopInitSubset(const edm::ParameterSet& cfg):
  src_ ( cfg.getParameter<edm::InputTag>( "src" ) )
{
  produces<reco::CandidateCollection>();
}

TopInitSubset::~TopInitSubset()
{
}

void
TopInitSubset::produce(edm::Event& evt, const edm::EventSetup& setup)
{     
  edm::Handle<reco::CandidateCollection> src;
  evt.getByLabel(src_, src);
 
  const reco::CandidateRefProd ref = evt.getRefBeforePut<reco::CandidateCollection>(); 
  std::auto_ptr<reco::CandidateCollection> sel( new reco::CandidateCollection );

  //fill output collection
  fillOutput( *src, *sel );

  evt.put( sel );
}

void TopInitSubset::fillOutput(const reco::CandidateCollection& src, reco::CandidateCollection& sel)
{
  CandidateCollection::const_iterator t=src.begin();
  for( ; t!=src.end(); ++t){
    if( t->status() == TopInitID::status && abs( t->pdgId() )==TopInitID::tID ){ //is top
      for(int idx=0; idx<(int)t->numberOfMothers(); ++idx){      
	GenParticleCandidate* cand = new GenParticleCandidate( t->mother(idx)->threeCharge(), t->mother(idx)->p4(), 
							       t->mother(idx)->vertex(), t->mother(idx)->pdgId(), 
							       t->mother(idx)->status(), false );
	auto_ptr<reco::Candidate> ptr( cand );
	sel.push_back( ptr );
      }
      break;
    }
  }
}
