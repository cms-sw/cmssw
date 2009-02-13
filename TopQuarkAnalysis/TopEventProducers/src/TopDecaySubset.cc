#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "AnalysisDataFormats/TopObjects/interface/TtGenEvent.h"
#include "TopQuarkAnalysis/TopEventProducers/interface/TopDecaySubset.h"

using namespace std;
using namespace reco;

static const unsigned int kMAXEVT=5; //maximal number of events to print for debugging

TopDecaySubset::TopDecaySubset(const edm::ParameterSet& cfg):
  pdg_( cfg.getParameter<unsigned int >( "pdgId" ) ),
  src_( cfg.getParameter<edm::InputTag>( "src" ) )
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

  // clear existing refs
  clearReferences();
  
  // fill output collection depending on whether the W
  // boson is contained in the generator listing or not

  // for  top branch
  wInDecayChain(*src, TopDecayID::tID) ? fillFromFullListing(*src, *sel, TopDecayID::tID) : fillFromTruncatedListing(*src, *sel, TopDecayID::tID);
  // for ~top branch
  wInDecayChain(*src,-TopDecayID::tID) ? fillFromFullListing(*src, *sel,-TopDecayID::tID) : fillFromTruncatedListing(*src, *sel,-TopDecayID::tID);

  // fill references
  fillReferences( ref, *sel );

  // print decay chain for debugging
  printSource( *src, pdg_);
  printTarget( *sel, pdg_);

  // fan out to event
  evt.put( sel );
}

Particle::LorentzVector 
TopDecaySubset::getP4(const reco::GenParticle::const_iterator first,
		      const reco::GenParticle::const_iterator last, int pdgId, double mass)
{
  Particle::LorentzVector vec;
  reco::GenParticle::const_iterator p=first;
  for( ; p!=last; ++p){
    if( p->status() == TopDecayID::unfrag ){
      vec+=getP4( p->begin(), p->end(), p->pdgId(), p->p4().mass() );
      if( abs(pdgId)==TopDecayID::tID && fabs(vec.mass()-mass)/mass<0.01){
	//break if top mass is in accordance with status 3 particle. 
	//Then the real top is reconstructed and adding more gluons 
	//and qqbar pairs would end up in virtualities. 
	break;
      }
    }
    else{ 
      if( abs(pdgId)==TopDecayID::WID ){//in case of W
	if( abs(p->pdgId())!=TopDecayID::WID ){
	  //skip W with status 2 to 
	  //prevent double counting
	  vec+=p->p4();
	}
      }
      else{
	vec+=p->p4();
      }
    }
  }
  return vec;
}

Particle::LorentzVector 
TopDecaySubset::getP4(const reco::GenParticle::const_iterator first,
		      const reco::GenParticle::const_iterator last, int pdgId)
{
  Particle::LorentzVector vec;
  reco::GenParticle::const_iterator p=first;
  for( ; p!=last; ++p){
    if( p->status()<=TopDecayID::stable && p->pdgId() == pdgId){
      vec=p->p4();
      break;
    }
    else if( p->status()==TopDecayID::unfrag){
      vec+=getP4(p->begin(), p->end(), pdgId);   
    }
  }
  return vec;
}

void
TopDecaySubset::clearReferences()
{
  // clear vector of references 
  refs_.clear();  
  // set idx for mother particles to start value of -1
  // (first entry will set it raise it to 0)
  motherPartIdx_=-1;
}

bool 
TopDecaySubset::wInDecayChain(const reco::GenParticleCollection& src, const int& partId)
{
  bool isContained=false;
  for(GenParticleCollection::const_iterator t=src.begin(); t!=src.end(); ++t){
    if( t->status() == TopDecayID::unfrag && t->pdgId()==partId ){ 
      for(GenParticle::const_iterator td=t->begin(); td!=t->end(); ++td){
	if( td->status()==TopDecayID::unfrag && abs( td->pdgId() )==TopDecayID::WID ){ 
	  isContained=true;
	  break;
	}
      }
    }
  }
  if( !isContained ){
    edm::LogWarning( "decayChain" )
      << " W boson is not contained in decay chain in the original gen particle listing.   \n"
      << " A status 2 equivalent W candidate will be re-reconstructed from the W daughters \n"
      << " but the hadronization of the W might be screwed up. Contact an expert for the   \n"
      << " generator in use to assure that what you are doing is ok.";     
  }
  return isContained;
}

void 
TopDecaySubset::fillFromFullListing(const reco::GenParticleCollection& src, reco::GenParticleCollection& sel, const int& partId)
{
  for(GenParticleCollection::const_iterator t=src.begin(); t!=src.end(); ++t){
    if( t->status() == TopDecayID::unfrag && t->pdgId()==partId ){ 
      //if source particle is top or topBar     
      GenParticle* cand = new GenParticle( t->threeCharge(), getP4( t->begin(), t->end(), t->pdgId(), t->p4().mass() ),
					   t->vertex(), t->pdgId(), t->status(), false );
      auto_ptr<reco::GenParticle> ptr( cand );
      sel.push_back( *ptr );
      ++motherPartIdx_;
      
      //keep top index for the map for 
      //management of the daughter refs
      int iTop=motherPartIdx_, iW=0;
      vector<int> topDaughs, wDaughs;
      //iterate over top daughters
      GenParticle::const_iterator td=t->begin();
      for( ; td!=t->end(); ++td){
	if( td->status()==TopDecayID::unfrag && abs( td->pdgId() )==TopDecayID::bID ){ 
	  //is beauty
	  GenParticle* cand = new GenParticle( td->threeCharge(), getP4( td->begin(), td->end(), td->pdgId() ), //take stable particle p4
					       td->vertex(), td->pdgId(), td->status(), false );
	  auto_ptr<GenParticle> ptr( cand );
	  sel.push_back( *ptr );	  
	  topDaughs.push_back( ++motherPartIdx_ ); //push index of top daughter
	}
	if( td->status()==TopDecayID::unfrag && abs( td->pdgId() )==TopDecayID::WID ){ 
	  //is W boson
	  GenParticle* cand = new GenParticle( td->threeCharge(), getP4( td->begin(), td->end(), td->pdgId()), //take stable particle p4 
					       td->vertex(), td->pdgId(), td->status(), true );
	  auto_ptr<GenParticle> ptr( cand );
	  sel.push_back( *ptr );
	  topDaughs.push_back( ++motherPartIdx_ ); //push index of top daughter	
	  iW=motherPartIdx_; //keep W idx for the map

	  //iterate over W daughters
	  GenParticle::const_iterator wd=td->begin();
	  for( ; wd!=td->end(); ++wd){
	    //make sure the W daughter is stable and not the W itself
	    if( wd->status()==TopDecayID::unfrag && !(abs(wd->pdgId())==TopDecayID::WID) ) {
	      GenParticle* cand = new GenParticle( wd->threeCharge(), getP4( wd->begin(), wd->end(), wd->pdgId() ), //take stable particle p4
						   wd->vertex(), wd->pdgId(), wd->status(), false);
	      auto_ptr<GenParticle> ptr( cand );
	      sel.push_back( *ptr );
	      wDaughs.push_back( ++motherPartIdx_ ); //push index of wBoson daughter
	      
              if( wd->status()==TopDecayID::unfrag && abs( wd->pdgId() )==TopDecayID::tauID ){ 
		//is tau
	        fillTree(motherPartIdx_,wd->begin(),sel); //pass daughter of tau which is of status
		                               //2 and by this skip status 3 particle
	      }
	    } 
	  }
	}
      }
      refs_[ iTop ]=topDaughs;
      refs_[ iW   ]=wDaughs; 
    }
  }
}

void 
TopDecaySubset::fillFromTruncatedListing(const reco::GenParticleCollection& src, reco::GenParticleCollection& sel, const int& partId)
{
  //needed for W reconstruction from daughters
  reco::Particle::Point wVtx;
  reco::Particle::Charge wQ=0;
  reco::Particle::LorentzVector wP4;
  for(GenParticleCollection::const_iterator t=src.begin(); t!=src.end(); ++t){
    if( t->status() == TopDecayID::unfrag && t->pdgId()==partId ){ 
      //if source particle is top or topBar     
      GenParticle* cand = new GenParticle( t->threeCharge(), getP4( t->begin(), t->end(), t->pdgId(), t->p4().mass() ),
					   t->vertex(), t->pdgId(), t->status(), false );
      auto_ptr<reco::GenParticle> ptr( cand );
      sel.push_back( *ptr );
      ++motherPartIdx_;
      
      //keep top index for the map for 
      //management of the daughter refs
      int iTop=motherPartIdx_, iW=0;
      vector<int> topDaughs, wDaughs;
      //iterate over top daughters
      for(GenParticle::const_iterator td=t->begin(); td!=t->end(); ++td){
	if( td->status()==TopDecayID::unfrag && abs( td->pdgId() )==TopDecayID::bID ){ 
	  //is beauty
	  GenParticle* cand = new GenParticle( td->threeCharge(), getP4( td->begin(), td->end(), td->pdgId() ), //take stable particle p4
					       td->vertex(), td->pdgId(), td->status(), false );
	  auto_ptr<GenParticle> ptr( cand );
	  sel.push_back( *ptr );	  
	  topDaughs.push_back( ++motherPartIdx_ ); //push index of top daughter
	}
	//count all 4-vectors but the b
	if( td->status()==TopDecayID::unfrag && abs( td->pdgId() )!=TopDecayID::bID ){
	  //is non-beauty
	  GenParticle* cand = new GenParticle( td->threeCharge(), getP4( td->begin(), td->end(), td->pdgId() ), //take stable particle p4
					       td->vertex(), td->pdgId(), td->status(), false);
	  auto_ptr<GenParticle> ptr( cand );
	  sel.push_back( *ptr );
	  //get w quantities from its daughters; take care of the non-trivial 
	  //charge definition of quarks/leptons and of the non integer charge
	  //of the quarks
	  if( fabs(td->pdgId() )<TopDecayID::tID ) //quark
	    wQ += ((td->pdgId()>0)-(td->pdgId()<0))*abs(cand->threeCharge());
	  else //lepton
	    wQ +=-((td->pdgId()>0)-(td->pdgId()<0))*abs(cand->charge());
	  wVtx=cand->vertex();	      
	  wP4+=cand->p4();
	  wDaughs.push_back( ++motherPartIdx_ ); //push index of wBoson daughter
	  
	  if( td->status()==TopDecayID::unfrag && abs( td->pdgId() )==TopDecayID::tauID ){ 
	    //is tau
	    fillTree(motherPartIdx_,td->begin(),sel); //pass daughter of tau which is of status
	    //2 and by this skip status 3 particle
	  }
	}
	if( (td+1)==t->end() ){
	  //reco of W boson is completed
	  GenParticle* cand = new GenParticle( wQ, wP4, wVtx, wQ*TopDecayID::WID, TopDecayID::unfrag, false);
	  auto_ptr<GenParticle> ptr( cand );
	  sel.push_back( *ptr );
	  //push index of top daughter	
	  topDaughs.push_back( ++motherPartIdx_ ); 
	  //keep W idx for the map
	  iW=motherPartIdx_; 
	  //reset W quantities for next W boson
	  wQ  = 0;
	  wVtx= reco::Particle::Point(0, 0, 0);
	  wP4 = reco::Particle::LorentzVector(0, 0, 0, 0);
	}
      }
      refs_[ iTop ]=topDaughs;
      refs_[ iW   ]=wDaughs;
    }
  }
}

void 
TopDecaySubset::fillReferences(const reco::GenParticleRefProd& ref, reco::GenParticleCollection& sel)
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

void 
TopDecaySubset::fillTree(int& idx, const GenParticle::const_iterator part, reco::GenParticleCollection& sel)
{
  vector<int> daughters;
  int idx0 = idx;
  GenParticle::const_iterator daughter=part->begin();
  for( ; daughter!=part->end(); ++daughter){
    GenParticle* cand = new GenParticle( daughter->threeCharge(), getP4( daughter->begin(), daughter->end(), daughter->pdgId() ),
					 daughter->vertex(), daughter->pdgId(), daughter->status(), false);
    auto_ptr<GenParticle> ptr( cand );
    sel.push_back( *ptr );
    daughters.push_back( ++idx ); //push index of daughter
    fillTree(idx,daughter,sel);   //continue recursively
  }  
  if(daughters.size()) {
     refs_[ idx0 ] = daughters;
  }
}

void 
TopDecaySubset::printSource(const reco::GenParticleCollection& src, const int& pdgId=0)
{
  unsigned int idx=0;
  std::string linestr;
  for(GenParticleCollection::const_iterator q=src.begin(); q!=src.end(); ++q, ++idx){
    unsigned int jdx=0;
    std::string daugstr; // keeps pdgIds of daughters
    for(GenParticle::const_iterator qd=q->begin(); qd!=q->end(); ++qd, ++jdx){
      if(jdx<kMAXEVT){
	char buffer[5];
	sprintf(buffer, "%i", qd->pdgId());
	daugstr += buffer;
	if(qd+1 != q->end()){
	  daugstr += ",";
	}
      }
      else if(jdx==kMAXEVT){
	char buffer[5];
	sprintf(buffer, "%i", q->numberOfDaughters());
	daugstr += "...(";
	daugstr += buffer;
	daugstr += ")";
      }
    }
    daugstr += "\n";
    char buffer[100];
    sprintf(buffer, "%8i%15i%10i%25s", idx, q->pdgId(), q->status(), daugstr.c_str());
    linestr += buffer; 
  }
  edm::LogVerbatim( "inputChain" ) 
    << "\nParticle-idx      pdgId      status        pdgId of Daughters"
    << "\n============================================================="
    << "\n" << linestr
    << "------------------------\n"
    << "'...' means more than" << kMAXEVT;
}


void 
TopDecaySubset::printTarget(reco::GenParticleCollection& sel, const int& pdgId=0)
{
  GenParticleCollection::iterator q=sel.begin();
  for(int idx=0; q!=sel.end(); ++q, ++idx){
    if( (pdgId==0 && sel[idx].pdgId()==6) || abs(sel[idx].pdgId())==pdgId){
      std::string linestr;
      GenParticleCollection::iterator p=sel.begin();
      for(int idx=0; p!=sel.end(); ++p, ++idx){
	map<int, vector<int> >::const_iterator daughters=refs_.find( idx );
	std::string daugstr; // keeps pdgIds of daughters
	if( daughters!=refs_.end() ){	  
	  for(vector<int>::const_iterator daughter = daughters->second.begin(); 
	      daughter!=daughters->second.end(); ++daughter){
	    //convert pdgId into c string w/o too much trouble
	    char buffer[5];
	    sprintf( buffer, "%i", sel[*daughter].pdgId() ); 
	    daugstr += buffer;
	    if(daughter+1 != daughters->second.end()){
	      daugstr += ",";
	    }
	  }
	  daugstr += "\n"; 
	}
	else{
	  daugstr += ("-\n");
	}
	char buffer[100];
	sprintf(buffer, "%8i%15i%10i%25s", idx, sel[idx].pdgId(), sel[idx].status(), daugstr.c_str());
	linestr += buffer;
      }
      edm::LogVerbatim( "decayChain" ) 
	<< "\nParticle-idx      pdgId      status        pdgId of Daughters"
	<< "\n============================================================="
	<< "\n" << linestr;
    }
  }
}
