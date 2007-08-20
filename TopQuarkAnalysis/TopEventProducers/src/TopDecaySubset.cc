#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "AnalysisDataFormats/TopObjects/interface/TtGenEvent.h"
#include "TopQuarkAnalysis/TopEventProducers/interface/TopDecaySubset.h"

using namespace std;
using namespace reco;

TopDecaySubset::TopDecaySubset(const edm::ParameterSet& cfg):
  src_ ( cfg.getParameter<edm::InputTag>( "src" ) )
{
  produces<reco::CandidateCollection>();
}

TopDecaySubset::~TopDecaySubset()
{
}

void
TopDecaySubset::produce(edm::Event& evt, const edm::EventSetup& setup)
{     
  edm::Handle<reco::CandidateCollection> src;
  evt.getByLabel(src_, src);
 
  const reco::CandidateRefProd ref = evt.getRefBeforePut<reco::CandidateCollection>(); 
  std::auto_ptr<reco::CandidateCollection> sel( new reco::CandidateCollection );

  //fill output collection
  fillOutput( *src, *sel );
  //fill references
  fillRefs( ref, *sel );

  evt.put( sel );
}

Particle::LorentzVector TopDecaySubset::fourVector(const reco::Candidate& p)
{
  Particle::LorentzVector vec;
  Candidate::const_iterator pd=p.begin();
  for( ; pd!=p.end(); ++pd){
    if( pd->status() == TopDecayID::status ){
      vec+=fourVector( *pd );
    }
    else{
      //skip W with status 2 to
      //prevent double counting
      if( abs(pd->pdgId())!=TopDecayID::WID )
	vec+=pd->p4();
    }
  }
  return vec;
}

void TopDecaySubset::fillOutput(const reco::CandidateCollection& src, reco::CandidateCollection& sel)
{
  CandidateCollection::const_iterator t=src.begin();
  for(int idx=-1; t!=src.end(); ++t){
    if( t->status() == TopDecayID::status && abs( t->pdgId() )==TopDecayID::tID ){ //is top      
      GenParticleCandidate* cand = new GenParticleCandidate( t->threeCharge(), fourVector( *t ), 
							     t->vertex(), t->pdgId(), t->status(), false );
      auto_ptr<reco::Candidate> ptr( cand );
      sel.push_back( ptr );
      ++idx;

      //keep top index for the map for 
      //management of the daughter refs 
      int iTop=idx, iW=0;
      vector<int> topDaughs, wDaughs;

      //iterate over top daughters
      Candidate::const_iterator td=t->begin();
      for( ; td!=t->end(); ++td){
	if( td->status()==TopDecayID::status && abs( td->pdgId() )==TopDecayID::bID ){ //is beauty	  
	  GenParticleCandidate* cand = new GenParticleCandidate( td->threeCharge(), fourVector( *td ), 
								 td->vertex(), td->pdgId(), td->status(), false );
	  auto_ptr<Candidate> ptr( cand );
	  sel.push_back( ptr );	  
	  topDaughs.push_back( ++idx ); //push index of top daughter
	}
	if( td->status()==TopDecayID::status && abs( td->pdgId() )==TopDecayID::WID ){ //is W boson
	  GenParticleCandidate* cand = new GenParticleCandidate( td->threeCharge(), fourVector( *td ), 
								 td->vertex(), td->pdgId(), td->status(), true );
	  auto_ptr<Candidate> ptr( cand );
	  sel.push_back( ptr );
	  topDaughs.push_back( ++idx ); //push index of top daughter

	  //keep W idx 
	  //for the map
	  iW=idx;

	  //iterate over W daughters
	  Candidate::const_iterator wd=td->begin();
	  for( ; wd!=td->end(); ++wd){
	    if (wd->pdgId() != td->pdgId()) {

	      GenParticleCandidate* cand = new GenParticleCandidate( wd->threeCharge(), fourVector( *wd ), 
								     wd->vertex(), wd->pdgId(), wd->status(), false);
	      auto_ptr<Candidate> ptr( cand );
	      sel.push_back( ptr );
	      wDaughs.push_back( ++idx ); //push index of wBoson daughter
	    }
	  }
	}
      }
      refs_[ iTop ]=topDaughs;
      refs_[ iW ]=wDaughs;
    }
  }
}

void TopDecaySubset::fillRefs(const reco::CandidateRefProd& ref, reco::CandidateCollection& sel)
{ 
  CandidateCollection::iterator p=sel.begin();
  for(int idx=0; p!=sel.end(); ++p, ++idx){
    //find daughter reference vectors in refs_ and add daughters
    map<int, vector<int> >::const_iterator daughters=refs_.find( idx );
    if( daughters!=refs_.end() ){
      vector<int>::const_iterator daughter = daughters->second.begin();
      for( ; daughter!=daughters->second.end(); ++daughter){
	GenParticleCandidate* part = dynamic_cast<GenParticleCandidate* > (&(*p));
	if(part == 0){
	  throw edm::Exception( edm::errors::InvalidReference, "Not a GenParticleCandidate" );
	}
	part->addDaughter( CandidateRef(ref, *daughter) );
      }
    }
  }
}
