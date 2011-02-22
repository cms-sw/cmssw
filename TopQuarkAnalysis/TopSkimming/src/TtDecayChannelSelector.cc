#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "AnalysisDataFormats/TopObjects/interface/TopGenEvent.h"
#include "TopQuarkAnalysis/TopSkimming/interface/TtDecayChannelSelector.h"

// static const string for status check in  
// TtDecayChannelSelector::search functions
static const std::string kGenParticles = "genParticles";

// number of top branches for decay selection
static const unsigned int kTopBranches   = 2;

// maximal number of possible leptonic decay 
// channels
static const unsigned int kDecayChannels = 3;


TtDecayChannelSelector::TtDecayChannelSelector(const edm::ParameterSet& cfg):
  invert_ ( cfg.getParameter<bool>("invert" ) ),
  allowLepton_(false), allow1Prong_(false), 
  allow3Prong_(false)
{
  // tau decays are not restricted if this PSet does not exist at all
  restrictTauDecays_=cfg.existsAs<edm::ParameterSet>("restrictTauDecays");
  // determine allowed tau decays
  if(restrictTauDecays_){
    edm::ParameterSet allowedTauDecays = cfg.getParameter<edm::ParameterSet>("restrictTauDecays");
    // tau decays are not restricted if none of the following parameters exists
    restrictTauDecays_=(allowedTauDecays.existsAs<bool>("leptonic"  )|| 
			allowedTauDecays.existsAs<bool>("oneProng"  )|| 
			allowedTauDecays.existsAs<bool>("threeProng") );
    // specify the different possible restrictions of the tau decay channels
    allowLepton_ = (allowedTauDecays.existsAs<bool>("leptonic"  ) ? allowedTauDecays.getParameter<bool>("leptonic"  ) : false); 
    allow1Prong_ = (allowedTauDecays.existsAs<bool>("oneProng"  ) ? allowedTauDecays.getParameter<bool>("oneProng"  ) : false); 
    allow3Prong_ = (allowedTauDecays.existsAs<bool>("threeProng") ? allowedTauDecays.getParameter<bool>("threeProng") : false);
  }
  // allowed top decays PSet
  edm::ParameterSet allowedTopDecays = cfg.getParameter<edm::ParameterSet>("allowedTopDecays");

  // fill decayBranchA_
  edm::ParameterSet decayBranchA = allowedTopDecays.getParameter<edm::ParameterSet>("decayBranchA");
  decayBranchA_.push_back(decayBranchA.getParameter<bool>("electron"));
  decayBranchA_.push_back(decayBranchA.getParameter<bool>("muon"    ));
  decayBranchA_.push_back(decayBranchA.getParameter<bool>("tau"     ));

  // fill decay branchB_
  edm::ParameterSet decayBranchB = allowedTopDecays.getParameter<edm::ParameterSet>("decayBranchB");
  decayBranchB_.push_back(decayBranchB.getParameter<bool>("electron"));
  decayBranchB_.push_back(decayBranchB.getParameter<bool>("muon"    ));
  decayBranchB_.push_back(decayBranchB.getParameter<bool>("tau"     ));

  // fill allowedDecays_
  for(unsigned int d=0; d<kDecayChannels; ++d){
    allowedDecays_.push_back(decayBranchA_[d]+decayBranchB_[d]);
  }
}

TtDecayChannelSelector::~TtDecayChannelSelector()
{ 
} 

bool
TtDecayChannelSelector::operator()(const reco::GenParticleCollection& parts, std::string inputType) const
{
  bool verbose=false; // set this to true for debugging and add TtDecayChannelSelector category to the MessageLogger in your cfg file
  unsigned int iLep=0;
  unsigned int iTop=0,iBeauty=0,iElec=0,iMuon=0,iTau=0;
  for(reco::GenParticleCollection::const_iterator top=parts.begin(); top!=parts.end(); ++top){
    if( search(top, TopDecayID::tID, inputType) ){
      ++iTop;
      for(reco::GenParticle::const_iterator td=top->begin(); td!=top->end(); ++td){
	if( search(td, TopDecayID::bID, inputType) ){
	  ++iBeauty;
	}
	if( search(td, TopDecayID::WID, inputType) ){
	  for(reco::GenParticle::const_iterator wd=td->begin(); wd!=td->end(); ++wd){
	    if( std::abs(wd->pdgId())==TopDecayID::elecID ){
	      ++iElec;
	    }
	    if( std::abs(wd->pdgId())==TopDecayID::muonID ){
	      ++iMuon;
	    }
	    if( std::abs(wd->pdgId())==TopDecayID::tauID  ){ 
	      if(restrictTauDecays_){
		// count as iTau if it is leptonic, one-prong
		// or three-prong and ignore increasing iLep
		// though else
		if(tauDecay(*wd)){
		  ++iTau; 
		} else{
		  ++iLep; 
		}
	      }
	      else{
		++iTau;
	      }
	    }
	  }
	}
      }
    }
  }
  if(verbose) {
    edm::LogVerbatim log("TtDecayChannelSelector");
    log << "----------------------"   << "\n"
	<< " iTop    : " << iTop      << "\n"
	<< " iBeauty : " << iBeauty   << "\n"
	<< " iElec   : " << iElec     << "\n"
	<< " iMuon   : " << iMuon     << "\n"
	<< " iTau    : " << iTau+iLep;
    if(restrictTauDecays_ && (iTau+iLep)>0){
      log << " (" << iTau << ")\n";
    }
    else{
      log << "\n";
    }
    log << "- - - - - - - - - - - "   << "\n";
  }
  iLep+=iElec+iMuon+iTau;

  bool accept=false;
  unsigned int channel = decayChannel();
  if( (iTop==2) && (iBeauty==2) ){
    if( channel==iLep ){
      if( channel==0 ){
        // no lepton: accept without restriction we already 
	// know that the number of leptons is correct
	accept=true;
      }
      if( channel==1 ){
        // one lepton: check that this one is allowed
        accept=(iElec&&allowedDecays_[Elec]) || (iMuon&&allowedDecays_[Muon]) || (iTau&&allowedDecays_[Tau]);
      }
      if( channel==2 ){
	if( checkSum(allowedDecays_)==channel ){
	  // no redundancy
	  accept = (allowedDecays_[Elec]==(int)iElec) && (allowedDecays_[Muon]==(int)iMuon) && (allowedDecays_[Tau]==(int)iTau);
	}
	else{
	  // reject events with wrong tau decays
	  if(iElec+iMuon+iTau!=channel){
	    accept = false;
	  }
	  else {
	    if((iElec==2)||(iMuon==2)||(iTau==2)) {
	      // same lepton twice: check that this is allowed.
	      accept = (allowedDecays_[Elec]==(int)iElec)||(allowedDecays_[Muon]==(int)iMuon)||(allowedDecays_[Tau]==(int)iTau);
	    } 
	    else {
	      // two different leptons: look if there is a possible combination
	      accept = ( ((iElec&&decayBranchA_[Elec])&&((iMuon&&decayBranchB_[Muon])||(iTau &&decayBranchB_[Tau ]))) ||
	                 ((iMuon&&decayBranchA_[Muon])&&((iElec&&decayBranchB_[Elec])||(iTau &&decayBranchB_[Tau ]))) ||
			 ((iTau &&decayBranchA_[Tau ])&&((iElec&&decayBranchB_[Elec])||(iMuon&&decayBranchB_[Muon])))   );
	    }
	  }
	}
      }
    }
    accept=( (!invert_&& accept) || (!(!invert_)&& !accept) );
  }
  else{
    edm::LogWarning ( "NoVtbDecay" ) << "Decay is not via Vtb";
  }
  if(verbose)
    edm::LogVerbatim("TtDecayChannelSelector") << " accept  : " << accept;
  return accept;
}

bool
TtDecayChannelSelector::search(reco::GenParticleCollection::const_iterator& part, int pdgId, std::string& inputType) const
{
  if(inputType==kGenParticles){
    return (std::abs(part->pdgId())==pdgId && part->status()==TopDecayID::unfrag) ? true : false;
  }
  else{
    return (std::abs(part->pdgId())==pdgId) ? true : false;
  }
}

bool
TtDecayChannelSelector::search(reco::GenParticle::const_iterator& part, int pdgId, std::string& inputType) const
{
  if(inputType==kGenParticles){
    return (std::abs(part->pdgId())==pdgId && part->status()==TopDecayID::unfrag) ? true : false;
  }
  else{
    return (std::abs(part->pdgId())==pdgId) ? true : false;
  }
}

unsigned int 
TtDecayChannelSelector::countProngs(const reco::Candidate& part) const
{
  // if stable, return 1 or 0
  if(part.status()==1){
    return (part.charge()!=0);
  }
  // if unstable, call recursively on daughters
  int prong =0;
  for(reco::Candidate::const_iterator daughter=part.begin();daughter!=part.end(); ++daughter){
    prong += countProngs(*daughter);
  }
  return prong;
}

bool
TtDecayChannelSelector::tauDecay(const reco::Candidate& tau) const
{
  bool leptonic = false;
  unsigned int nch = 0;
  // loop on tau decays, check for an elec
  // or muon and count charged particles
  for(reco::Candidate::const_iterator daughter=tau.begin();daughter!=tau.end(); ++daughter){
    // if the tau daughter is again a tau, this means that the particle has 
    // still to be propagated; in that case, return the result of the same 
    // method applied on the daughter of the current particle
    if(daughter->pdgId()==tau.pdgId()){
      return tauDecay(*daughter);
    }
    // check for leptons
    leptonic |= (std::abs(daughter->pdgId())==TopDecayID::elecID || std::abs(daughter->pdgId())==TopDecayID::muonID);
    // count charged particles
    nch += countProngs(*daughter);
  }
  return ((allowLepton_ &&  leptonic)          ||
	  (allow1Prong_ && !leptonic && nch==1)||
	  (allow3Prong_ && !leptonic && nch==3));
}
