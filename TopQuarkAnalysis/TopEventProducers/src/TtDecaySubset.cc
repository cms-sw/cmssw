#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "AnalysisDataFormats/TopObjects/interface/TtGenEvent.h"
#include "TopQuarkAnalysis/TopEventProducers/interface/TtDecaySubset.h"

using namespace std;
using namespace reco;

namespace TtDecayID{
  static const int status = 3;
  static const int tID = 6;
  static const int bID = 5;
  static const int WID =24;
}

TtDecaySubset::TtDecaySubset(const edm::ParameterSet& cfg):
  src_ ( cfg.getParameter<edm::InputTag>( "src" ) )
{
  produces<reco::CandidateCollection>();
}

TtDecaySubset::~TtDecaySubset()
{
}

void
TtDecaySubset::produce(edm::Event& evt, const edm::EventSetup& setup)
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

Particle::LorentzVector TtDecaySubset::fourVector(const reco::Candidate& p)
{
  Particle::LorentzVector vec;
  Candidate::const_iterator pd=p.begin();
  for( ; pd!=p.end(); ++pd){
    if( status( *pd )==TtDecayID::status ){
      vec+=fourVector( *pd );
    }
    else{
      if( abs(pd->pdgId())!=24 )
	vec+=pd->p4();
    }
  }
  return vec;
}

void TtDecaySubset::fillOutput(const reco::CandidateCollection& src, reco::CandidateCollection& sel)
{
  CandidateCollection::const_iterator t=src.begin();
  for(int idx=-1; t!=src.end(); ++t){
    if( status( *t )==TtDecayID::status && abs( t->pdgId() )==TtDecayID::tID ){ //is top      
      GenParticleCandidate* cand = new GenParticleCandidate( t->charge(), fourVector( *t ), 
							     t->vertex(), t->pdgId(), status( *t ) );
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
	if( status( *td )==TtDecayID::status && abs( td->pdgId() )==TtDecayID::bID ){ //is beauty	  
	  GenParticleCandidate* cand = new GenParticleCandidate( td->charge(), fourVector( *td ), 
								 td->vertex(), td->pdgId(), status( *td ) );
	  auto_ptr<Candidate> ptr( cand );
	  sel.push_back( ptr );	  
	  topDaughs.push_back( ++idx ); //push index of top daughter
	}
	if( status( *td )==TtDecayID::status && abs( td->pdgId() )==TtDecayID::WID ){ //is W boson
	  GenParticleCandidate* cand = new GenParticleCandidate( td->charge(), fourVector( *td ), 
								 td->vertex(), td->pdgId(), status( *td ) );
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
	      GenParticleCandidate* cand = new GenParticleCandidate( wd->charge(), fourVector( *wd ), 
								     wd->vertex(), wd->pdgId(), status( *wd ) );	      auto_ptr<Candidate> ptr( cand );
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

void TtDecaySubset::fillRefs(const reco::CandidateRefProd& ref, reco::CandidateCollection& sel)
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
