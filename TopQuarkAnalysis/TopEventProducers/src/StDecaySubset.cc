#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "AnalysisDataFormats/TopObjects/interface/StGenEvent.h"
#include "TopQuarkAnalysis/TopEventProducers/interface/StDecaySubset.h"

using namespace std;
using namespace reco;

namespace StDecayID{
  static const int status = 3;
  static const int tID = 6;
  static const int bID = 5;
  static const int WID =24;
}

StDecaySubset::StDecaySubset(const edm::ParameterSet& cfg):
  src_ ( cfg.getParameter<edm::InputTag>( "src" ) )
{
  switchOption    = cfg.getParameter<int>("SwitchChainType");
  produces<reco::GenParticleCollection>();
}

StDecaySubset::~StDecaySubset()
{
}

void
StDecaySubset::produce(edm::Event& evt, const edm::EventSetup& setup)
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

Particle::LorentzVector StDecaySubset::fourVector(const reco::GenParticle& p)
{
  Particle::LorentzVector vec;
  GenParticle::const_iterator pd=p.begin();
  for( ; pd!=p.end(); ++pd){
    if( pd->status()==StDecayID::status ){
      vec+=fourVector( *pd );
    }
    else{
      //skip W with status 2 to
      //prevent double counting
      if( abs(pd->pdgId())!=StDecayID::WID )
	vec+=pd->p4();
    }
  }
  return vec;
}

void StDecaySubset::fillOutput(const reco::GenParticleCollection& src, reco::GenParticleCollection& sel)
{
  if (switchOption==1) { // standard option: look for top and W, and navigate through the decay chains
    GenParticleCollection::const_iterator t=src.begin();
    for(int idx=-1; t!=src.end(); ++t){
      if( t->status()==StDecayID::status && abs( t->pdgId() )==StDecayID::tID ){ //is top      
	GenParticle* cand = new GenParticle( t->charge(), fourVector( *t ), 
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
	  if( td->status()==StDecayID::status && abs( td->pdgId() )==StDecayID::bID ){ //is beauty	  
	    GenParticle* cand = new GenParticle( td->charge(), fourVector( *td ), 
								   td->vertex(), td->pdgId(), td->status(), false );
	    auto_ptr<GenParticle> ptr( cand );
	    sel.push_back( *ptr );	  
	    topDaughs.push_back( ++idx ); //push index of top daughter
	  }
	  if( td->status()==StDecayID::status && abs( td->pdgId() )==StDecayID::WID ){ //is W boson
	    GenParticle* cand = new GenParticle( td->charge(), fourVector( *td ), 
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
		GenParticle* cand = new GenParticle( wd->charge(), fourVector( *wd ), 
								       wd->vertex(), wd->pdgId(), wd->status(), false );	      auto_ptr<GenParticle> ptr( cand );
		sel.push_back( *ptr );
		wDaughs.push_back( ++idx ); //push index of wBoson daughter
	      }
	    }
	  }
	}
	refs_[ iTop ]=topDaughs;
	refs_[ iW ]=wDaughs;
      }
    }
  } else if (switchOption==2) { // this is needed, for example, for the SingleTop generator, since it doesn't save the intermediate particles (lepton, neutrino and b are directly daughters of the incoming partons)

    int iP;
    vector<int> ipDaughs;

    GenParticleCollection::const_iterator ip1=src.begin();
    GenParticleCollection::const_iterator ip2=src.begin();
    for(int idx=-1; ip1!=src.end(); ++ip1){
      for(; ip2!=src.end(); ++ip2){

	//iterate over the daughters of both
	GenParticle::const_iterator td1=ip1->begin();
	GenParticle::const_iterator td2=ip2->begin();
	for( ; td1!=ip1->end(); ++td1){
	  for( ; td2!=ip2->end(); ++td2){
	    if (td1 == td2) { // daughter of both initial state partons

	      //	      ++idx;
	      //	      iP=idx;

	      GenParticle* cand = new GenParticle( td2->charge(), fourVector( *td2 ), 
								     td2->vertex(), td2->pdgId(), td2->status(), false );
	      auto_ptr<GenParticle> ptr( cand );
	      sel.push_back( *ptr );	  
	      ipDaughs.push_back( ++idx ); //push index of daughter
	      iP=idx;
	    }
	    refs_[ iP ]=ipDaughs;
	  }
	}// end of double loop on daughters

      }
    }
  }

}

void StDecaySubset::fillRefs(const reco::GenParticleRefProd& ref, reco::GenParticleCollection& sel)
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
