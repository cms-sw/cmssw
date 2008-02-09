#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "AnalysisDataFormats/TopObjects/interface/TtGenEvent.h"
#include "TopQuarkAnalysis/TopEventProducers/interface/TopDecaySubset.h"

using namespace std;
using namespace reco;

TopDecaySubset::TopDecaySubset(const edm::ParameterSet& cfg):
  src_ ( cfg.getParameter<edm::InputTag>( "src" ) )
{
  produces<reco::GenParticleCollection>();
}

TopDecaySubset::~TopDecaySubset()
{
}

void
TopDecaySubset::produce(edm::Event& evt, const edm::EventSetup& setup)
{     
  edm::Handle<reco::GenParticleCollection> src;
  evt.getByLabel(src_, src);
 
  const reco::GenParticleRefProd ref = evt.getRefBeforePut<reco::GenParticleCollection>(); 
  std::auto_ptr<reco::GenParticleCollection> sel( new reco::GenParticleCollection );

  //clear existing refs
  refs_.clear();  
  //fill output collection
  fillOutput( *src, *sel );
  //fill references
  fillRefs( ref, *sel );

  evt.put( sel );
}

Particle::LorentzVector TopDecaySubset::fourVector(const reco::GenParticle::const_iterator first,
						   const reco::GenParticle::const_iterator last)
{
  Particle::LorentzVector vec;
  reco::GenParticle::const_iterator p=first;
  for( ; p!=last; ++p){
    if( p->status() == TopDecayID::status ){
      vec+=fourVector( p->begin(), p->end() );
    }
    else{
      //skip W with status 2 to
      //prevent double counting
      if( abs(p->pdgId())!=TopDecayID::WID ){
	vec+=p->p4();
      }
    }
  }
  return vec;
}

void TopDecaySubset::fillOutput(const reco::GenParticleCollection& src, reco::GenParticleCollection& sel)
{
  GenParticleCollection::const_iterator t=src.begin();
  for(int idx=-1; t!=src.end(); ++t){
    if( t->status() == TopDecayID::status && abs( t->pdgId() )==TopDecayID::tID ){ //is top      
      GenParticle* cand = new GenParticle( t->threeCharge(), fourVector( t->begin(), t->end() ), 
					   t->vertex(), t->pdgId(), t->status(), false );
      auto_ptr<reco::GenParticle> ptr( cand );
      sel.push_back( *ptr );
      ++idx;
      //keep top index for the map for 
      //management of the daughter refs 
      int iTop=idx, iW=0;
      vector<int> topDaughs, wDaughs;
      //iterate over top daughters
      GenParticle::const_iterator td=t->begin();
      for( ; td!=t->end(); ++td){
	if( td->status()==TopDecayID::status && abs( td->pdgId() )==TopDecayID::bID ){ //is beauty	  
	  GenParticle* cand = new GenParticle( td->threeCharge(), fourVector( td->begin(), td->end() ), 
					       td->vertex(), td->pdgId(), td->status(), false );
	  auto_ptr<GenParticle> ptr( cand );
	  sel.push_back( *ptr );	  
	  topDaughs.push_back( ++idx ); //push index of top daughter
	}
	if( td->status()==TopDecayID::status && abs( td->pdgId() )==TopDecayID::WID ){ //is W boson
	  GenParticle* cand = new GenParticle( td->threeCharge(), fourVector( td->begin(), td->end() ), 
					       td->vertex(), td->pdgId(), td->status(), true );
	  auto_ptr<GenParticle> ptr( cand );
	  sel.push_back( *ptr );
	  topDaughs.push_back( ++idx ); //push index of top daughter
	  //keep W idx 
	  //for the map
	  iW=idx;
	  //iterate over W daughters
	  GenParticle::const_iterator wd=td->begin();
	  for( ; wd!=td->end(); ++wd){
	    if (wd->pdgId() != td->pdgId()) {
	      GenParticle* cand = new GenParticle( wd->threeCharge(), fourVector( wd->begin(), wd->end() ), 
						   wd->vertex(), wd->pdgId(), wd->status(), false);
	      auto_ptr<GenParticle> ptr( cand );
	      sel.push_back( *ptr );
	      wDaughs.push_back( ++idx ); //push index of wBoson daughter
              if( wd->status()==TopDecayID::status && abs( wd->pdgId() )==TopDecayID::tauID ){ //is tau
	        fillTree(idx,*wd,sel);
	      }
	    }
	  }
	}
      }
      refs_[ iTop ]=topDaughs;
      refs_[ iW ]=wDaughs;
    }
  }
}

void TopDecaySubset::fillRefs(const reco::GenParticleRefProd& ref, reco::GenParticleCollection& sel)
{ 
  GenParticleCollection::iterator p=sel.begin();
  for(int idx=0; p!=sel.end(); ++p, ++idx){
    //find daughter reference vectors in refs_ and add daughters
    map<int, vector<int> >::const_iterator daughters=refs_.find( idx );
    if( daughters!=refs_.end() ){
      vector<int>::const_iterator daughter = daughters->second.begin();
      for( ; daughter!=daughters->second.end(); ++daughter){
	GenParticle* part = dynamic_cast<GenParticle* > (&(*p));
	if(part == 0){
	  throw edm::Exception( edm::errors::InvalidReference, "Not a GenParticle" );
	}
	part->addDaughter( GenParticleRef(ref, *daughter) );
        sel[*daughter].addMother( GenParticleRef(ref, idx) );
      }
    }
  }
}

void TopDecaySubset::fillTree(int& idx, const reco::GenParticle& particle, reco::GenParticleCollection& sel)
{
  vector<int> daughters;
  int idx0 = idx;
  GenParticle::const_iterator daughter=particle.begin();
  for( ; daughter!=particle.end(); ++daughter){
    GenParticle* cand = new GenParticle( daughter->threeCharge(), fourVector( daughter->begin(), daughter->end() ),
                                                           daughter->vertex(), daughter->pdgId(), daughter->status(), false);
    auto_ptr<GenParticle> ptr( cand );
    sel.push_back( *ptr );
    daughters.push_back( ++idx ); //push index of daughter
    fillTree(idx,*daughter,sel);  //continue recursively
  }  
  if(daughters.size()) {
     refs_[ idx0 ] = daughters;
  }
}

