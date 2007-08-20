#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "AnalysisDataFormats/TopObjects/interface/StGenEvent.h"
#include "TopQuarkAnalysis/TopEventProducers/interface/TopDecaySubset.h"
#include "TopQuarkAnalysis/TopEventProducers/interface/StGenEventReco.h"

using namespace std;
using namespace reco;

namespace TopProdID{
  static const int ProtonID = 2212; 
}

StGenEventReco::StGenEventReco(const edm::ParameterSet& cfg):
  src_ ( cfg.getParameter<edm::InputTag>( "src"  ) ),
  init_( cfg.getParameter<edm::InputTag>( "init" ) )
{
  produces<StGenEvent>();
}

StGenEventReco::~StGenEventReco()
{
}

void
StGenEventReco::produce(edm::Event& evt, const edm::EventSetup& setup)
{     
  edm::Handle<reco::CandidateCollection> parts;
  evt.getByLabel(src_,  parts);

  edm::Handle<reco::CandidateCollection> inits;
  evt.getByLabel(init_, inits);

  //find & fill InitialPartons
  std::vector<const reco::Candidate*> initParts;
  CandidateCollection::const_iterator top=inits->begin();
  for( ; top!=inits->end(); ++top){
    if( top->status()==TopDecayID::status && abs( top->pdgId() )==TopDecayID::tID ){
       fillInitialPartons( &(*top), initParts );
      break;
    }
  }

  //add TopDecayTree
  reco::CandidateRefProd cands( parts );

  //add genEvt to the output stream
  StGenEvent* genEvt = new StGenEvent( cands, initParts );
  std::auto_ptr<StGenEvent> gen( genEvt );
  evt.put( gen );
}

void
StGenEventReco::fillInitialPartons(const reco::Candidate* p, std::vector<const reco::Candidate*>& vec)
{
  for(int idx=0; idx<(int)p->numberOfMothers(); ++idx){
    if( p->mother(idx)->pdgId()==TopProdID::ProtonID && p->mother(idx)->numberOfMothers()==0 ){
      if( dynamic_cast<const reco::GenParticleCandidate*>( p ) == 0){
	throw edm::Exception( edm::errors::InvalidReference, "Not a GenParticleCandidate" );
      }
      vec.push_back( p->clone() );
    }
    else{
      fillInitialPartons( p->mother(idx), vec );
    }
  }
  return;
}
