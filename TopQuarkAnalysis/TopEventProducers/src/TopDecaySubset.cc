#include "TString.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "AnalysisDataFormats/TopObjects/interface/TopGenEvent.h"
#include "TopQuarkAnalysis/TopEventProducers/interface/TopDecaySubset.h"


// maximal number of daughters 
// to be printed for debugging
static const unsigned int kMAX=5; 

/// default constructor
TopDecaySubset::TopDecaySubset(const edm::ParameterSet& cfg):
  addRadiatedGluons_( cfg.getParameter<bool>( "addRadiatedGluons" ) ),
  src_( cfg.getParameter<edm::InputTag>( "src" ) )
{
  // mapping of the corresponding fillMode; see FillMode 
  // enumerator of TopDecaySubset for available modes
  if( cfg.getParameter<std::string>( "fillMode" ) == "kME"     ){ fillMode_=kME;     }
  if( cfg.getParameter<std::string>( "fillMode" ) == "kStable" ){ fillMode_=kStable; }
  // produces a set of gen particle collections following
  // the decay branch of top quarks to the first level of 
  // stable decay products
  produces<reco::GenParticleCollection>(); 
}

/// default destructor
TopDecaySubset::~TopDecaySubset()
{
}

/// write output into the event
void
TopDecaySubset::produce(edm::Event& event, const edm::EventSetup& setup)
{     
  // get source collection
  edm::Handle<reco::GenParticleCollection> src;
  event.getByLabel(src_, src);

  // print full listing of input collection for 
  // debuging with 'TopDecaySubset_printSource'
  printSource(*src);
  // determine shower model
  showerModel_=checkShowerModel(*src);

  // create target vector
  std::auto_ptr<reco::GenParticleCollection> target( new reco::GenParticleCollection );
  // get ref product from the event
  const reco::GenParticleRefProd ref = event.getRefBeforePut<reco::GenParticleCollection>(); 
  // clear existing refs
  clearReferences();  
  // check sanity
  checkSanity(*src);
  // fill output for top branch
  fillListing(*src, *target, TopDecayID::tID);
  // fill output for anti-top branch
  fillListing(*src, *target,-TopDecayID::tID);
  // fill references
  fillReferences(ref, *target);

  // print full listing of input collection for 
  // debuging with 'TopDecaySubset_printTarget'
  printTarget(*target);

  // write vectors to the event
  event.put(target);
}

/// check the sanity of the input particle listing
void
TopDecaySubset::checkSanity(const reco::GenParticleCollection& parts) const
{
  if( !checkWBoson(parts, TopDecayID::tID) ){
    throw edm::Exception( edm::errors::LogicError, "W boson is not contained in the original particle listing \n");
  }
  if( showerModel_==kNone ){
    throw edm::Exception( edm::errors::LogicError, "Particle listing does not correspond to any of the supported listings \n");
  }
}

/// check whether the W boson is contained in the original gen particle listing
bool 
TopDecaySubset::checkWBoson(const reco::GenParticleCollection& parts, const int& partId) const
{
  bool isContained=false;
  for(reco::GenParticleCollection::const_iterator t=parts.begin(); t!=parts.end(); ++t){
    if( t->status() == TopDecayID::unfrag && t->pdgId()==partId ){ 
      for(reco::GenParticle::const_iterator td=((showerModel_==kPythia)?t->begin():t->begin()->begin()); td!=((showerModel_==kPythia)?t->end():t->begin()->end()); ++td){
        if( (showerModel_==kPythia)?td->status()==TopDecayID::unfrag:td->status()==TopDecayID::stable && abs( td->pdgId() )==TopDecayID::WID ){ 
	  isContained=true;
	  break;
        }
      }
    }
  }
  if( !isContained ){
    edm::LogWarning( "decayChain" )
      << " W boson is not contained in decay chain in the gen particle listing \n"
      << " of the input collection. This means that the hadronization of the W \n"
      << " boson might be screwed up. Contact an expert for the generator in   \n"
      << " use to assure that what you are doing makes sense.";     
  }
  return isContained;
}

/// check the decay chain for the exploited shower model
TopDecaySubset::ShowerModel
TopDecaySubset::checkShowerModel(const reco::GenParticleCollection& parts) const
{
  for(reco::GenParticleCollection::const_iterator top=parts.begin(); top!=parts.end(); ++top){
    if(top->pdgId()==TopDecayID::tID && top->status()==TopDecayID::unfrag){
      // check for kHerwig type showers: here the status 3 top quark will 
      // have a single status 2 top quark as daughter, which has again 3 
      // or more status 2 daughters
      if( top->numberOfDaughters()==1){
	if( top->begin()->pdgId ()==TopDecayID::tID && top->begin()->status()==TopDecayID::stable && top->begin()->numberOfDaughters()>1)
	  return kHerwig;
      }
      // check for kPythia type showers: here the status 3 top quark will 
      // have all decay products and a status 2 top quark as daughters
      // the status 2 top quark will be w/o further daughters
      if( top->numberOfDaughters() >1){
	bool containsWBoson=false, containsQuarkDaughter=false;
	for(reco::GenParticle::const_iterator td=top->begin(); td!=top->end(); ++td){
	  if( td->pdgId () <TopDecayID::tID ) containsQuarkDaughter=true;
	  if( td->pdgId ()==TopDecayID::WID ) containsWBoson=true;
	}
	if(containsQuarkDaughter && containsWBoson)
	  return kPythia;
      }
    }
  }
  edm::LogWarning( "decayChain" )
    << " Can not find back any of the supported hadronization models. Models \n"
    << " which are supported are:                                            \n"
    << " Pythia  LO(+PS): Top(status 3) --> WBoson(status 3), Quark(status 3)\n"
    << " Herwig NLO(+PS): Top(status 2) --> Top(status 3) --> Top(status 2)    ";
  return kNone;
}

/// fill output vector for full decay chain 
void 
TopDecaySubset::fillListing(const reco::GenParticleCollection& src, reco::GenParticleCollection& target, const int& partId)
{
  unsigned int statusFlag;
  // determine status flag of the new 
  // particle depending on the FillMode
  fillMode_ == kME ? statusFlag=3 : statusFlag=2;

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
      for(reco::GenParticle::const_iterator td=((showerModel_==kPythia)?t->begin():t->begin()->begin()); td!=((showerModel_==kPythia)?t->end():t->begin()->end()); ++td){
	if( td->status()==TopDecayID::unfrag && abs( td->pdgId() )<=TopDecayID::bID ){ 
	  // if particle is beauty or other quark 
	  std::auto_ptr<reco::GenParticle> bPtr( new reco::GenParticle( td->threeCharge(), p4( td, statusFlag ), td->vertex(), td->pdgId(), statusFlag, false ) );
	  target.push_back( *bPtr );	  
	  // increment & push index of the top daughter
	  topDaughters.push_back( ++motherPartIdx_ ); 
	  if(addRadiatedGluons_){
	    addRadiation(motherPartIdx_,td,target); 
	  }
	}
	reco::GenParticle::const_iterator buffer = (showerModel_==kPythia)?td:td->begin();
	if( buffer->status()==TopDecayID::unfrag && abs( buffer->pdgId() )==TopDecayID::WID ){ 
	  // if particle is is W boson
	  std::auto_ptr<reco::GenParticle> wPtr(  new reco::GenParticle( buffer->threeCharge(), p4( buffer, statusFlag), buffer->vertex(), buffer->pdgId(), statusFlag, true ) );
	  target.push_back( *wPtr );
	  // increment & push index of the top daughter
	  topDaughters.push_back( ++motherPartIdx_ );
	  // keep the W idx for the map
	  iW=motherPartIdx_; 
	  if(addRadiatedGluons_){
	    addRadiation(motherPartIdx_,buffer,target); 
	  }
	  // iterate over W daughters
	  for(reco::GenParticle::const_iterator wd=((showerModel_==kPythia)?buffer->begin():buffer->begin()->begin()); wd!=((showerModel_==kPythia)?buffer->end():buffer->begin()->end()); ++wd){
	    // make sure the W daughter is of status unfrag and not the W itself
	    if( wd->status()==TopDecayID::unfrag && !(abs(wd->pdgId())==TopDecayID::WID) ) {
	      std::auto_ptr<reco::GenParticle> qPtr( new reco::GenParticle( wd->threeCharge(), p4( wd, statusFlag ), wd->vertex(), wd->pdgId(), statusFlag, false) );
	      target.push_back( *qPtr );
	      // increment & push index of the top daughter
	      wDaughters.push_back( ++motherPartIdx_ );
	      if(addRadiatedGluons_){
		addRadiation(motherPartIdx_,wd,target); 
              }
	      if( wd->status()==TopDecayID::unfrag && abs( wd->pdgId() )==TopDecayID::tauID ){ 
		// add tau daughters if the particle is a tau pass
		// the daughter of the tau which is of status 2
		addDaughters(motherPartIdx_,wd->begin(),target); 
	      }
	    } 
	  }
	}
	if(addRadiatedGluons_ && buffer->status()==TopDecayID::stable && ( buffer->pdgId()==TopDecayID::glueID || abs(buffer->pdgId())<TopDecayID::bID)){
	  // collect additional radiation from the top 
	  std::auto_ptr<reco::GenParticle> radPtr( new reco::GenParticle( buffer->threeCharge(), buffer->p4(), buffer->vertex(), buffer->pdgId(), statusFlag, false ) );
	  target.push_back( *radPtr );	
	  // increment & push index of the top daughter  
	  topDaughters.push_back( ++motherPartIdx_ );
	}
      }
      // add potential sisters of the top quark;
      // only for top to prevent double counting
      if(t->numberOfMothers()>0 && t->pdgId()==TopDecayID::tID){
        for(reco::GenParticle::const_iterator ts = t->mother()->begin(); ts!=t->mother()->end(); ++ts){
	  // loop over all daughters of the top mother i.e.
	  // the two top quarks and their potential sisters
	  if(abs(ts->pdgId())!=t->pdgId() && ts->pdgId()!=t->mother()->pdgId()){
	    // add all further particles but the two top quarks and potential 
	    // cases where the mother of the top has itself as daughter
	    reco::GenParticle* cand = new reco::GenParticle( ts->threeCharge(), ts->p4(), ts->vertex(), ts->pdgId(), ts->status(), false );
	    std::auto_ptr<reco::GenParticle> sPtr( cand );
	    target.push_back( *sPtr );
	    if(ts->begin()!=ts->end()){ 
	      // in case the sister has daughters increment
	      // and add the first generation of daughters
	      addDaughters(++motherPartIdx_,ts->begin(),target,false);
	    }
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

/// calculate lorentz vector from input 
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
      // descend by one level for each
      // status 3 particle on the way
      vec+=p4( p, statusFlag );
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
	vec+=p->p4();
	if( vec.mass()-top->mass()>0 ){
 	  // continue adding up gluons and qqbar pairs on the top 
 	  // line untill the nominal top mass is reached; then 
 	  // break in order to prevent picking up virtualities
	  break;
	}
      }
    }
  }
  return vec;
}

/// calculate lorentz vector from input (dedicated to top reconstruction)
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
    if( p->status()<=TopDecayID::stable && abs(p->pdgId ())==TopDecayID::WID){
      vec=p->p4();
    }
    else{
      if(p->status()<=TopDecayID::stable){
	// sum up the p4 of all stable particles 
	// (of status 1 or 2)
	vec+=p->p4();
      }
      else{
	if( p->status()==TopDecayID::unfrag){
	  // if the particle is unfragmented (i.e.
	  // status 3) descend by one level
	  vec+=p4(p, statusFlag);   
	}
      }
    }
  }
  return vec;
}

/// fill vector including all radiations from quarks originating from W/top
void 
TopDecaySubset::addRadiation(int& idx, const reco::GenParticle::const_iterator part, reco::GenParticleCollection& target)
{
  std::vector<int> daughters;
  int idxBuffer = idx;
  for(reco::GenParticle::const_iterator daughter=part->begin(); daughter!=part->end(); ++daughter){
    if(daughter->status()<=TopDecayID::stable && daughter->pdgId()!=part->pdgId()){
      // skip comment lines and make sure that
      // the particle is not double counted as 
      // dasughter of itself
      std::auto_ptr<reco::GenParticle> ptr(  new reco::GenParticle( daughter->threeCharge(), daughter->p4(), daughter->vertex(), daughter->pdgId(), daughter->status(), false) );
      target.push_back( *ptr );
      daughters.push_back( ++idx ); //push index of daughter
    }
  }  
  if(daughters.size()) {
    refs_[ idxBuffer ] = daughters;
  }
}

/// recursively fill vector for all further decay particles of a given particle
void 
TopDecaySubset::addDaughters(int& idx, const reco::GenParticle::const_iterator part, reco::GenParticleCollection& target, bool recursive)
{
  std::vector<int> daughters;
  int idxBuffer = idx;
  for(reco::GenParticle::const_iterator daughter=part->begin(); daughter!=part->end(); ++daughter){
    std::auto_ptr<reco::GenParticle> ptr( new reco::GenParticle( daughter->threeCharge(), daughter->p4(), daughter->vertex(), daughter->pdgId(), daughter->status(), false) );
    target.push_back( *ptr );
    // increment & push index of daughter
    daughters.push_back( ++idx );
    // continue recursively if desired
    if(recursive){
      addDaughters(idx,daughter,target);  
    }
  }  
  if(daughters.size()) {
    refs_[ idxBuffer ] = daughters;
  }
}

/// clear references
void
TopDecaySubset::clearReferences()
{
  // clear vector of references 
  refs_.clear();  
  // set idx for mother particles to a start value
  // of -1 (the first entry will raise it to 0)
  motherPartIdx_=-1;
}

/// fill references for output vector
void 
TopDecaySubset::fillReferences(const reco::GenParticleRefProd& ref, reco::GenParticleCollection& sel)
{ 
  int idx=0;
  for(reco::GenParticleCollection::iterator p=sel.begin(); p!=sel.end(); ++p, ++idx){
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

/// print the whole listing if particle with pdgId is contained in the top decay chain
void 
TopDecaySubset::printSource(const reco::GenParticleCollection& src)
{
  edm::LogVerbatim log("TopDecaySubset_printSource");
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

/// print the whole decay chain if particle with pdgId is contained in the top decay chain
void 
TopDecaySubset::printTarget(reco::GenParticleCollection& sel)
{
  edm::LogVerbatim log("TopDecaySubset_printTarget");
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
