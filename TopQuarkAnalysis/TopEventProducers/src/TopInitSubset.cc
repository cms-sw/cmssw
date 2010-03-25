#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "AnalysisDataFormats/TopObjects/interface/TtGenEvent.h"
#include "TopQuarkAnalysis/TopEventProducers/interface/TopInitSubset.h"

TopInitSubset::TopInitSubset(const edm::ParameterSet& cfg):
  src_ ( cfg.getParameter<edm::InputTag>( "src" ) )
{
  produces<reco::GenParticleCollection>();
}

TopInitSubset::~TopInitSubset()
{
}

void
TopInitSubset::produce(edm::Event& evt, const edm::EventSetup& setup)
{     
  edm::Handle<reco::GenParticleCollection> src;
  evt.getByLabel(src_, src);
 
  const reco::GenParticleRefProd ref = evt.getRefBeforePut<reco::GenParticleCollection>(); 
  std::auto_ptr<reco::GenParticleCollection> sel( new reco::GenParticleCollection );

  //fill output collection
  fillOutput( *src, *sel );

  evt.put( sel );
}

void TopInitSubset::fillOutput(const reco::GenParticleCollection& src, reco::GenParticleCollection& sel)
{
  reco::GenParticleCollection::const_iterator t=src.begin();
  for( ; t!=src.end(); ++t){
    if( t->status() == TopInitID::status && abs( t->pdgId() )==TopInitID::tID ){ //is top
      for(int idx=0; idx<(int)t->numberOfMothers(); ++idx){      
	reco::GenParticle* cand = new reco::GenParticle( t->mother(idx)->threeCharge(), t->mother(idx)->p4(), 
							       t->mother(idx)->vertex(), t->mother(idx)->pdgId(), 
							       t->mother(idx)->status(), false );
	std::auto_ptr<reco::GenParticle> ptr( cand );
	sel.push_back( *ptr );
      }
      break;
    }
  }
}
