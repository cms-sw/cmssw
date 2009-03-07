#include "TString.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "AnalysisDataFormats/TopObjects/interface/TtGenEvent.h"
#include "TopQuarkAnalysis/TopEventProducers/interface/TopDecaySubset.h"


// maximal number of daughters 
// to be printed for debugging
static const unsigned int kMAX=5; 


TopDecaySubset::TopDecaySubset(const edm::ParameterSet& cfg):
  src_( cfg.getParameter<edm::InputTag>( "src" ) )
{
  // produces a set of gep particle collections following
  // the decay branch of top quarks to the first level of 
  // stable decay products
  produces<reco::GenParticleCollection>();
  produces<reco::GenParticleCollection>("beforePS");
  produces<reco::GenParticleCollection>("afterPS" ); 
  produces<reco::GenParticleCollection>("ME"      );
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
  // create target vectors
  std::auto_ptr<reco::GenParticleCollection> selME      ( new reco::GenParticleCollection );
  std::auto_ptr<reco::GenParticleCollection> selStable  ( new reco::GenParticleCollection );
  std::auto_ptr<reco::GenParticleCollection> selBeforePS( new reco::GenParticleCollection );
  std::auto_ptr<reco::GenParticleCollection> selAfterPS ( new reco::GenParticleCollection );

  // fill output vectors with references
  printSource(*src);
  fillOutput (*src, *selME,       ref, kME      );
  fillOutput (*src, *selStable,   ref, kStable  );
  fillOutput (*src, *selBeforePS, ref, kBeforePS);
  fillOutput (*src, *selAfterPS,  ref, kAfterPS );

  // write vectors to event
  evt.put( selME, "ME" );
  evt.put( selStable   );
  evt.put( selBeforePS, "beforePS" );
  evt.put( selAfterPS,  "afterPS"  );
}


void
TopDecaySubset::fillOutput(const reco::GenParticleCollection& src, reco::GenParticleCollection& target, const reco::GenParticleRefProd& ref, FillMode mode)
{
  // clear existing refs
  clearReferences();
  
  // fill output for top branch
  wInDecayChain(src, TopDecayID::tID) ? 
    fromFullListing (src, target, TopDecayID::tID, mode): 
    fromTruncListing(src, target, TopDecayID::tID, mode);
  // fill output for anti-top branch
  wInDecayChain(src,-TopDecayID::tID) ? 
    fromFullListing (src, target,-TopDecayID::tID, mode): 
    fromTruncListing(src, target,-TopDecayID::tID, mode);

  // fill references
  fillReferences(ref, target);

  // print decay chain for debugging
  printTarget(target);
}

void
TopDecaySubset::clearReferences()
{
  // clear vector of references 
  refs_.clear();  
  // set idx for mother particles to a start value
  // of -1 (the first entry will raise it to 0)
  motherPartIdx_=-1;
}

bool 
TopDecaySubset::wInDecayChain(const reco::GenParticleCollection& src, const int& partId)
{
  bool isContained=false;
  for(reco::GenParticleCollection::const_iterator t=src.begin(); t!=src.end(); ++t){
    if( t->status() == TopDecayID::unfrag && t->pdgId()==partId ){ 
      for(reco::GenParticle::const_iterator td=t->begin(); td!=t->end(); ++td){
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
TopDecaySubset::fromFullListing(const reco::GenParticleCollection& src, reco::GenParticleCollection& target, const int& partId, FillMode mode)
{
  int statusFlag;
  // determine status flag of the new 
  // particle depending on the FillMode
  mode == kME ? statusFlag=3 : statusFlag=2;

  for(reco::GenParticleCollection::const_iterator t=src.begin(); t!=src.end(); ++t){
    if( t->status() == TopDecayID::unfrag && t->pdgId()==partId ){ 
      // if particle is top or anti-top depending 
      // on the function's argument partId 
      std::auto_ptr<reco::GenParticle> topPtr( new reco::GenParticle( t->threeCharge(), p4( t, statusFlag), t->vertex(), t->pdgId(), statusFlag, false ) );
      target.push_back( *topPtr );
      ++motherPartIdx_;
      // keep the top index for the map to manage the daughter refs
      int iTop=motherPartIdx_; 
      std::vector<int> topDaughters;
      // define the W boson index (to be set later) for the map to 
      // manage the daughter refs
      int iW = 0;
      std::vector<int> wDaughters;

      //iterate over top daughters
      for(reco::GenParticle::const_iterator td=t->begin(); td!=t->end(); ++td){
	if( td->status()==TopDecayID::unfrag && abs( td->pdgId() )<=TopDecayID::bID ){ 
	  // if particle is beauty or other quark 
	  if(mode == kAfterPS){
	    addRadiation(motherPartIdx_,td,target); 
	  } 
	  std::auto_ptr<reco::GenParticle> bPtr( new reco::GenParticle( td->threeCharge(), p4( td, statusFlag ), td->vertex(), td->pdgId(), statusFlag, false ) );
	  target.push_back( *bPtr );	  
	  // increment & push index of the top daughter
	  topDaughters.push_back( ++motherPartIdx_ ); 
	  if(mode == kBeforePS){
	    addRadiation(motherPartIdx_,td,target); 
	  }
	}
	if( td->status()==TopDecayID::unfrag && abs( td->pdgId() )==TopDecayID::WID ){ 
	  // if particle is is W boson
	  if(mode == kAfterPS){
	    addRadiation(motherPartIdx_,td,target); 
	  }
	  std::auto_ptr<reco::GenParticle> wPtr(  new reco::GenParticle( td->threeCharge(), p4( td, statusFlag), td->vertex(), td->pdgId(), statusFlag, true ) );
	  target.push_back( *wPtr );
	  // increment & push index of the top daughter
	  topDaughters.push_back( ++motherPartIdx_ );
	  // keep the W idx for the map
	  iW=motherPartIdx_; 
	  if(mode == kBeforePS){
	    addRadiation(motherPartIdx_,td,target); 
	  }

	  // iterate over W daughters
	  for(reco::GenParticle::const_iterator wd=td->begin(); wd!=td->end(); ++wd){
	    // make sure the W daughter is of status unfrag and not the W itself
	    if( wd->status()==TopDecayID::unfrag && !(abs(wd->pdgId())==TopDecayID::WID) ) {
	      if(mode == kAfterPS){
		addRadiation(motherPartIdx_,wd,target); 
	      } 
	      std::auto_ptr<reco::GenParticle> qPtr( new reco::GenParticle( wd->threeCharge(), p4( wd, statusFlag ), wd->vertex(), wd->pdgId(), statusFlag, false) );
	      target.push_back( *qPtr );
	      // increment & push index of the top daughter
	      wDaughters.push_back( ++motherPartIdx_ );
	      if(mode == kBeforePS){
		if(abs(wd->pdgId())<TopDecayID::tID)
		  // restrict to radiation from quarks
		  addRadiation(motherPartIdx_,wd,target); 
              }
	      if( wd->status()==TopDecayID::unfrag && abs( wd->pdgId() )==TopDecayID::tauID ){ 
		// add tau daughters if the particle is a tau pass
		// the daughter of the tau which is of status 2
		addTauDaughters(motherPartIdx_,wd->begin(),target); 
	      }
	    } 
	  }
	}
	if(td->status()==TopDecayID::stable && ( td->pdgId()==TopDecayID::glueID || abs(td->pdgId())<TopDecayID::bID)){
	  // collect additional radiation from the top 
	  std::auto_ptr<reco::GenParticle> radPtr( new reco::GenParticle( td->threeCharge(), td->p4(), td->vertex(), td->pdgId(), statusFlag, false ) );
	  target.push_back( *radPtr );	
	  // increment & push index of the top daughter  
	  topDaughters.push_back( ++motherPartIdx_ );
	}
      }
      // add potential sisters of the top quark; this is only
      // done for top not for anti-top to prevent double counting
      if(t->numberOfMothers()>0 && t->pdgId()==TopDecayID::tID){
        for(reco::GenParticle::const_iterator ts = t->mother()->begin(); ts!=t->mother()->end(); ++ts){
	  // loop over all daughters of the top mother i.e.
	  // the two top quarks and their potential sisters
	  if(abs(ts->pdgId())!=TopDecayID::tID){
	    // add all further particles but the two top
	    // quarks 
	    reco::GenParticle* cand = new reco::GenParticle( ts->threeCharge(), ts->p4(), ts->vertex(), ts->pdgId(), ts->status(), false );
	    std::auto_ptr<reco::GenParticle> ptr( cand );
	    target.push_back( *ptr );
	    ++motherPartIdx_;
	  }
	}
      }
      // fill the map for the administration 
      // of daughter indices
      refs_[ iTop ] = topDaughters;
      refs_[ iW   ] = wDaughters; 
    }
  }
}

void 
TopDecaySubset::fromTruncListing(const reco::GenParticleCollection& src, reco::GenParticleCollection& target, const int& partId, FillMode mode)
{
  int statusFlag;
  // determine status flag of the new 
  // particle depending on the FillMode
  mode == kME ? statusFlag=3 : statusFlag=2;

  // needed for W reconstruction from 
  // daughters
  reco::Particle::Point wVtx;
  reco::Particle::Charge wQ=0;
  reco::Particle::LorentzVector wP4;
  for(reco::GenParticleCollection::const_iterator t=src.begin(); t!=src.end(); ++t){
    if( t->status() == TopDecayID::unfrag && t->pdgId()==partId ){ 
      // if particle is top or anti-top 
      std::auto_ptr<reco::GenParticle> topPtr( new reco::GenParticle( t->threeCharge(), p4( t, statusFlag), t->vertex(), t->pdgId(), statusFlag, false ) );
      target.push_back( *topPtr );
      ++motherPartIdx_;
      // keep the top index for the map to manage the daughter refs
      int iTop=motherPartIdx_; 
      std::vector<int> topDaughters;
      // define the W boson index (to be set later) for the map to 
      // manage the daughter refs
      int iW = 0;
      std::vector<int> wDaughters;
      
      //iterate over top daughters
      for(reco::GenParticle::const_iterator td=t->begin(); td!=t->end(); ++td){
	if( td->status()==TopDecayID::unfrag && abs( td->pdgId() )==TopDecayID::bID ){ 
	  // if particle is beauty or other quark 
	  if(mode == kAfterPS){
	    addRadiation(motherPartIdx_,td,target); 
	  } 
	  std::auto_ptr<reco::GenParticle> bPtr( new reco::GenParticle( td->threeCharge(), p4( td, statusFlag ), td->vertex(), td->pdgId(), statusFlag, false ) );
	  target.push_back( *bPtr );	  
	  // increment & push index of the top daughter
	  topDaughters.push_back( ++motherPartIdx_ ); 
	  if(mode == kBeforePS){
	    addRadiation(motherPartIdx_,td,target); 
	  }
	}

	// count all 4-vectors but the b quark 
	// as daughters of the W boson
	if( td->status()==TopDecayID::unfrag && abs( td->pdgId() )!=TopDecayID::bID ){
	  if(mode == kAfterPS){
	    addRadiation(motherPartIdx_,td,target); 
	  } 
	  reco::GenParticle* cand = new reco::GenParticle( td->threeCharge(), p4( td, statusFlag ), td->vertex(), td->pdgId(), statusFlag, false);
	  std::auto_ptr<reco::GenParticle> qPtr( cand );
	  target.push_back( *qPtr );
	  // increment & push index of the top daughter
	  wDaughters.push_back( ++motherPartIdx_ );
	  if(mode == kBeforePS){
	    if(abs(td->pdgId())<TopDecayID::tID)
	      // restrict to radiation from quarks
	      addRadiation(motherPartIdx_,td,target); 
	  }

	  // reconstruct the quantities of the W boson from its daughters; take 
	  // care of the non-trivial charge definition of quarks/leptons and of 
	  // the non-integer charge of the quarks
	  if( fabs(td->pdgId() )<TopDecayID::tID ) //quark
	    wQ += ((td->pdgId()>0)-(td->pdgId()<0))*abs(cand->threeCharge());
	  else //lepton
	    wQ +=-((td->pdgId()>0)-(td->pdgId()<0))*abs(cand->charge());
	  wVtx=cand->vertex();	      
	  wP4+=cand->p4();

	  if( td->status()==TopDecayID::unfrag && abs( td->pdgId() )==TopDecayID::tauID ){ 
	    // add tau daughters if the particle is a tau pass
	    // the daughter of the tau which is of status 2
	    addTauDaughters(motherPartIdx_,td->begin(),target); 
	  }
	}
	if( (td+1)==t->end() ){
	  // the reconstruction of the W boson 
	  // candidate; now add it to the list
	  if(mode == kAfterPS){
	    addRadiation(motherPartIdx_,td,target); 
	  }
	  std::auto_ptr<reco::GenParticle> wPtr(  new reco::GenParticle( td->threeCharge(), p4( td, statusFlag), td->vertex(), td->pdgId(), statusFlag, true ) );
	  target.push_back( *wPtr );
	  // increment & push index of the top daughter
	  topDaughters.push_back( ++motherPartIdx_ );
	  // keep the W idx for the map
	  iW=motherPartIdx_; 
	  if(mode == kBeforePS){
	    addRadiation(motherPartIdx_,td,target); 
	  }
	  
	  // reset the quantities of the W boson 
	  // for the next reconstruction cycle
	  wQ  = 0;
	  wVtx= reco::Particle::Point(0, 0, 0);
	  wP4 = reco::Particle::LorentzVector(0, 0, 0, 0);
	}
	if(td->status()==TopDecayID::stable && ( td->pdgId()==TopDecayID::glueID || abs(td->pdgId())<TopDecayID::bID)){
	  // collect additional radiation from the top 
	  std::auto_ptr<reco::GenParticle> radPtr( new reco::GenParticle( td->threeCharge(), td->p4(), td->vertex(), td->pdgId(), statusFlag, false ) );
	  target.push_back( *radPtr );	
	  // increment & push index of the top daughter  
	  topDaughters.push_back( ++motherPartIdx_ );
	}
      }
      // add potential sisters of the top quark; this is only
      // done for top not for anti-top to prevent double counting
      if(t->numberOfMothers()>0 && t->pdgId()==TopDecayID::tID){
        for(reco::GenParticle::const_iterator ts = t->mother()->begin(); ts!=t->mother()->end(); ++ts){
	  // loop over all daughters of the top mother i.e.
	  // the two top quarks and their potential sisters
	  if(abs(ts->pdgId())!=TopDecayID::tID){
	    // add all further particles but the two top
	    // quarks 
	    reco::GenParticle* cand = new reco::GenParticle( ts->threeCharge(), ts->p4(), ts->vertex(), ts->pdgId(), ts->status(), false );
	    std::auto_ptr<reco::GenParticle> ptr( cand );
	    target.push_back( *ptr );
	    ++motherPartIdx_;
	  }
	}
      }
      // fill the map for the administration 
      // of daughter indices
      refs_[ iTop ] = topDaughters;
      refs_[ iW   ] = wDaughters; 
    }
  }
}

reco::Particle::LorentzVector 
TopDecaySubset::p4(const std::vector<reco::GenParticle>::const_iterator top, int statusFlag)
{
  // calculate the four vector for top/anti-top quarks from 
  // the W boson and the b quark plain or including all 
  // additional radiation depending on switch 'plain'
  if(statusFlag==TopDecayID::unfrag){
    // return 4 momentum as it is
    return top->p4();
  }
  reco::Particle::LorentzVector vec;
  for(reco::GenParticle::const_iterator p=top->begin(); p!=top->end(); ++p){
    if( p->status() == TopDecayID::unfrag ){
      // decend by one level for each
      // status 3 particle on the way
      vec+=p4( p, statusFlag );
      if(abs(p->pdgId())==TopDecayID::tID && fabs(vec.mass()-(p->p4().mass()))/p->p4().mass()<0.1 ){
	// break if top mass is in accordance with status 3 particle. 
	// then the real top is reconstructed and adding more gluons 
	// and qqbar pairs would end up in virtualities. 
	break;
      }
    }
    else{ 
      if( abs(top->pdgId())==TopDecayID::WID ){
	// in case of a W boson skip the status 
	// 2 particle to prevent double counting
	if( abs(p->pdgId())!=TopDecayID::WID ) 
	  vec+=p->p4();
      } 
      else{
	// add all four vectors for each stable
	// particle (status 1 or 2) on the way 
	// else
	vec+=p->p4();
      }
    }
  }
  return vec;
}

reco::Particle::LorentzVector 
TopDecaySubset::p4(const reco::GenParticle::const_iterator part, int statusFlag)
{
  // calculate the four vector for all top daughters from 
  // their daughters including additional radiation 
  if(statusFlag==TopDecayID::unfrag){
    // return 4 momentum as it is
    return part->p4();
  }
  reco::Particle::LorentzVector vec;
  for(reco::GenParticle::const_iterator p=part->begin(); p!=part->end(); ++p){
    if( p->status()<=TopDecayID::stable && 
	p->pdgId ()==TopDecayID::WID){
      vec=p->p4();
    }
    else{
      if(p->status()<=TopDecayID::stable){
	// sum up the p4 of all stable particles 
	// (of status 1 or 2)
	vec+=p->p4();
      }
      else 
	if( p->status()==TopDecayID::unfrag)
	  // if the particle is unfragmented (i.e.
	  // status 3) decend by one level
	  vec+=p4(p, statusFlag);   
    }
  }
  return vec;
}

void 
TopDecaySubset::addRadiation(int& idx, const reco::GenParticle::const_iterator part, reco::GenParticleCollection& target)
{
  std::vector<int> daughters;
  int idxBuffer = idx;
  for(reco::GenParticle::const_iterator daughter=part->begin(); daughter!=part->end(); ++daughter){
    if(daughter->status()<=TopDecayID::stable && daughter->pdgId()!=part->pdgId()){
      // skip comment lines and make sure that
      // the daughter is not double counted
      std::auto_ptr<reco::GenParticle> ptr(  new reco::GenParticle( daughter->threeCharge(), daughter->p4(), daughter->vertex(), daughter->pdgId(), daughter->status(), false) );
      target.push_back( *ptr );
      daughters.push_back( ++idx ); //push index of daughter
    }
  }  
  if(daughters.size()) {
     refs_[ idxBuffer ] = daughters;
  }
}

void 
TopDecaySubset::addTauDaughters(int& idx, const reco::GenParticle::const_iterator part, reco::GenParticleCollection& target)
{
  std::vector<int> daughters;
  int idxBuffer = idx;
  for(reco::GenParticle::const_iterator daughter=part->begin(); daughter!=part->end(); ++daughter){
    std::auto_ptr<reco::GenParticle> ptr( new reco::GenParticle( daughter->threeCharge(), daughter->p4(), daughter->vertex(), daughter->pdgId(), daughter->status(), false) );
    target.push_back( *ptr );
    // increment & push index of daughter
    daughters.push_back( ++idx );
    // continue recursively
    addTauDaughters(idx,daughter,target);  
  }  
  if(daughters.size()) {
     refs_[ idxBuffer ] = daughters;
  }
}

void 
TopDecaySubset::fillReferences(const reco::GenParticleRefProd& ref, reco::GenParticleCollection& sel)
{ 
  reco::GenParticleCollection::iterator p=sel.begin();
 for(int idx=0; p!=sel.end(); ++p, ++idx){
 //find daughter reference vectors in refs_ and add daughters
   std::map<int, std::vector<int> >::const_iterator daughters=refs_.find( idx );
   if( daughters!=refs_.end() ){
     for(std::vector<int>::const_iterator daughter = daughters->second.begin(); 
	 daughter!=daughters->second.end(); ++daughter){
       reco::GenParticle* part = dynamic_cast<reco::GenParticle* > (&(*p));
       if(part == 0){
	 throw edm::Exception( edm::errors::InvalidReference, "Not a GenParticle" );
       }
       part->addDaughter( reco::GenParticleRef(ref, *daughter) );
       sel[*daughter].addMother( reco::GenParticleRef(ref, idx) );
     }
   }
 }
}

void 
TopDecaySubset::printSource(const reco::GenParticleCollection& src)
{
  edm::LogVerbatim log("TopDecaySubset::printSource");
  log << "\n   idx   pdg   stat      px          py         pz             mass          daughter pdg's  "
      << "\n===========================================================================================\n";

  for(unsigned int t=0; t<src.size(); ++t){
    if( src[t].pdgId()==TopDecayID::tID ){
      // restrict to the top in order 
      // to have it shown only once 
      int idx=0;
      for(reco::GenParticleCollection::const_iterator p=src.begin(); p!=src.end(); ++p, ++idx){
	// loop the top daughters
	log << std::right << std::setw( 5) << idx
	    << std::right << std::setw( 7) << src[idx].pdgId()
	    << std::right << std::setw( 5) << src[idx].status() << "  "
	    << std::right << std::setw(10) << std::setprecision( 6 ) << src[idx].p4().x() << "  "	
	    << std::right << std::setw(10) << std::setprecision( 6 ) << src[idx].p4().y() << "  "	
	    << std::right << std::setw(10) << std::setprecision( 6 ) << src[idx].p4().z() << "  "	
	    << std::right << std::setw(15) << std::setprecision( 6 ) << src[idx].p4().mass() 
	    << "   ";
	// search for potential daughters; if they exits 
	// print the daughter to the screen in the last 
	// column of the table separated by ','
	TString pdgIds;
	unsigned int jdx=0;
	for(reco::GenParticle::const_iterator d=p->begin(); d!=p->end(); ++d, ++jdx){
	  if(jdx<kMAX){
	    pdgIds+=d->pdgId();
	    if(d+1 != p->end()){
	      pdgIds+= ",";
	    }
	  }
	  else{
	    pdgIds+="...(";
	    pdgIds+= p->numberOfDaughters();
	    pdgIds+=")";
	    break;
	  }
 	}
	if(idx>0){
	  log << std::setfill( ' ' ) << std::right << std::setw(15) << pdgIds; 
	  log << "\n";
	}
	else{
	  log << std::setfill( ' ' ) << std::right << std::setw(15) << "-\n";
 	}
      }
    }
  }
}

void 
TopDecaySubset::printTarget(reco::GenParticleCollection& sel)
{
  edm::LogVerbatim log("TopDecaySubset::printTarget");
  log << "\n   idx   pdg   stat      px          py         pz             mass          daughter pdg's  "
      << "\n===========================================================================================\n";

  for(unsigned int t=0; t<sel.size(); ++t){
    if( sel[t].pdgId()==TopDecayID::tID ){
      // restrict to the top in order 
      // to have it shown only once      
      int idx=0;
      for(reco::GenParticleCollection::iterator p=sel.begin(); p!=sel.end(); ++p, ++idx){
	// loop the top daughters
	log << std::right << std::setw( 5) << idx
	    << std::right << std::setw( 7) << sel[idx].pdgId()
	    << std::right << std::setw( 5) << sel[idx].status() << "  "
	    << std::right << std::setw(10) << std::setprecision( 6 ) << sel[idx].p4().x() << "  "	
	    << std::right << std::setw(10) << std::setprecision( 6 ) << sel[idx].p4().y() << "  "	
	    << std::right << std::setw(10) << std::setprecision( 6 ) << sel[idx].p4().z() << "  "	
	    << std::right << std::setw(15) << std::setprecision( 6 ) << sel[idx].p4().mass() 
	    << "   ";
	// search for potential daughters; if they exits 
	// print the daughter to the screen in the last 
	// column of the table separated by ','
	TString pdgIds;
	std::map<int, std::vector<int> >::const_iterator daughters=refs_.find( idx );
	if( daughters!=refs_.end() ){
	  unsigned int jdx=0;
	  for(std::vector<int>::const_iterator d = daughters->second.begin(); d!=daughters->second.end(); ++d, ++jdx){
	    if(jdx<kMAX){
	      pdgIds+=sel[*d].pdgId();
	      if(d+1 != daughters->second.end()){
		pdgIds+= ",";
	      }
	    }
	    else{
	      pdgIds+="...(";
	      pdgIds+= daughters->second.size();
	      pdgIds+=")";
	      break;
	    }
	  }
	  log << std::setfill( ' ' ) << std::right << std::setw(15) << pdgIds; 
	  log << "\n";
	}
	else{
	  log << std::setfill( ' ' ) << std::right << std::setw(15) << "-\n";
	}
      }
    }
  }
}
