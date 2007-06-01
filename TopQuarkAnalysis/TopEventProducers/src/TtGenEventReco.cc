#include "TopQuarkAnalysis/TopEventProducers/interface/TtGenEventReco.h"

using namespace std;
using namespace reco;

static const int stat = 3;
static const int tId  = 6;
static const int bId  = 5;
static const int WId  =24;

TtGenEventReco::TtGenEventReco(const edm::ParameterSet& cfg):
  src_ ( cfg.template getParameter<edm::InputTag>( "src" ) )
{
  produces<reco::CandidateCollection>();
}

TtGenEventReco::~TtGenEventReco()
{
}

void
TtGenEventReco::produce(edm::Event& evt, const edm::EventSetup& setup)
{     
  edm::Handle<reco::CandidateCollection> src;
  evt.getByLabel(src_, src);
 
  const reco::CandidateRefProd ref = evt.getRefBeforePut<reco::CandidateCollection>(); 
  std::auto_ptr<reco::CandidateCollection> sel(new reco::CandidateCollection);

  //fill output collection
  fillOutput( *src, *sel );
  //fill references
  fillRefs( ref, *sel );

  evt.put( sel );
}

void TtGenEventReco::fillOutput(const reco::CandidateCollection& src, reco::CandidateCollection& sel)
{
  CandidateCollection::const_iterator t=src.begin();
  for(int idx=-1; t!=src.end(); ++t){
    if( status( *t )==stat && abs( t->pdgId() )==tId ){
      GenParticleCandidate* cand = new GenParticleCandidate( t->charge(), t->p4(), 
							     t->vertex(), t->pdgId(), status( *t ) );
      auto_ptr<reco::Candidate> ptr( cand );
      sel.push_back( ptr );
      ++idx;
      int inode=idx, ileaf=0;
      vector<int> node, leaf;
      Candidate::const_iterator td=t->begin();
      for( ; td!=t->end(); ++td){
	if( status( *td )==stat && abs( td->pdgId() )==bId ){
	  GenParticleCandidate* cand = new GenParticleCandidate( td->charge(), td->p4(), 
								 td->vertex(), td->pdgId(), status( *td ) );
	  auto_ptr<Candidate> ptr( cand );
	  sel.push_back( ptr );
	  node.push_back( ++idx ); 
	}
	if( status( *td )==stat && abs( td->pdgId() )==WId ){
	  GenParticleCandidate* cand = new GenParticleCandidate( td->charge(), td->p4(), 
								 td->vertex(), td->pdgId(), status( *td ) );
	  auto_ptr<Candidate> ptr( cand );
	  sel.push_back( ptr );
	  node.push_back( ++idx ); 
	  ileaf=idx;
	  Candidate::const_iterator wd=td->begin();
	  for( ; wd!=td->end(); ++wd){
	    GenParticleCandidate* cand = new GenParticleCandidate( wd->charge(), wd->p4(), 
								   wd->vertex(), wd->pdgId(), status( *wd ) );
	    auto_ptr<Candidate> ptr( cand );
	    sel.push_back( ptr );
	    leaf.push_back( ++idx );
	  }
	}
      }
      refs_[ inode ]=node;
      refs_[ ileaf ]=leaf;
    }
  }
}

void TtGenEventReco::fillRefs(const reco::CandidateRefProd& ref, reco::CandidateCollection& sel)
{ 
  CandidateCollection::iterator p=sel.begin();
  for(int idx=0; p!=sel.end(); ++p, ++idx){
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
