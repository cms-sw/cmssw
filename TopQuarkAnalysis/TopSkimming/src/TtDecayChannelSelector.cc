#include <iostream>

#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "TopQuarkAnalysis/TopSkimming/interface/TtDecayChannelSelector.h"

TtDecayChannelSelector::TtDecayChannelSelector(const edm::ParameterSet& cfg):
  invert_ ( cfg.getParameter<bool>("invert" ) )
{
  chn1_ = cfg.getParameter<Decay>("channel_1"); 
  chn2_ = cfg.getParameter<Decay>("channel_2"); 
  tauDecay_  = cfg.getParameter<Decay>("tauDecays");

  channel_=0;
  //determine decay channel
  if( count(chn1_.begin(), chn1_.end(), 1) > 0 ){ ++channel_; }
  if( count(chn2_.begin(), chn2_.end(), 1) > 0 ){ ++channel_; }

  summed_=0;
  //fill vector of allowed leptons
  Decay::const_iterator idx1=chn1_.begin(),idx2=chn2_.begin(); 
  for( ; idx1!=chn1_.end(), idx2!=chn2_.end(); ++idx1, ++idx2){
    summed_+=(*idx1)+(*idx2);
    decay_.push_back( (*idx1)+(*idx2) );
  }

  parseDecayInput(chn1_, chn2_);
  parseTauDecayInput(tauDecay_);

}

TtDecayChannelSelector::~TtDecayChannelSelector()
{ } 

bool
TtDecayChannelSelector::operator()(const reco::CandidateCollection& parts) const
{
  int iTop=0,iBeauty=0,iElec=0,iMuon=0,iTau=0;
  reco::CandidateCollection::const_iterator top=parts.begin();
  int iLep=0;
  for(; top!=parts.end(); ++top){
    if( top->status()==3 && abs((*top).pdgId())==6 ){
      ++iTop;
      reco::Candidate::const_iterator td=(*top).begin();
      for(; td!=(*top).end(); ++td){
	if( td->status()==3 && abs((*td).pdgId())==5 )
	  {++iBeauty;}
	if( td->status()==3 && abs((*td).pdgId())==24 ){
	  reco::Candidate::const_iterator wd=(*td).begin();
	  for(; wd!=(*td).end(); ++wd){
	    if( abs((*wd).pdgId())==11 ){++iElec;}
	    if( abs((*wd).pdgId())==13 ){++iMuon;}
	    if( abs((*wd).pdgId())==15 ){ if(checkTauDecay(*wd)) ++iTau; else ++iLep; }
	  }
	}
      }
    }
  }
  iLep+=iElec+iMuon+iTau;
  bool accept=false;
  if( (iTop==2) && (iBeauty==2) ){
    if( channel_==iLep ){
      if( channel_==0 ){
        // no lepton: accept without restriction 
	// (we already know the number of leptons is right)
	accept=true;
      }
      if( channel_==1 ){
        // one lepton: check that that one is allowed
        accept = (iElec&&decay_[Elec]) || (iMuon&&decay_[Muon]) || (iTau&&decay_[Tau]);
      }
      if( channel_==2 ){
	if( summed_==channel_ ){
	  // no redundancy
	  accept = (decay_[Elec]==iElec) && (decay_[Muon]==iMuon) && (decay_[Tau]==iTau);
	}
	else{
	  // that first test is to quickly reject event with wrong tau decay
	  if(iElec+iMuon+iTau!=2) 
	    accept = false;
	  else {
	    if((iElec==2)||(iMuon==2)||(iTau==2)) {
	      // same lepton twice: check that this is allowed.
	      accept = (decay_[Elec]==iElec)||(decay_[Muon]==iMuon)||(decay_[Tau]==iTau);
	    } else {
	      // two different leptons: look if there is a possible combination
	      accept = ( ((iElec&&chn1_[Elec])&&((iMuon&&chn2_[Muon])||(iTau &&chn2_[Tau ]))) ||
	                 ((iMuon&&chn1_[Muon])&&((iElec&&chn2_[Elec])||(iTau &&chn2_[Tau ]))) ||
			 ((iTau &&chn1_[Tau ])&&((iElec&&chn2_[Elec])||(iMuon&&chn2_[Muon])))   );
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
  return accept;
}

unsigned int 
TtDecayChannelSelector::countChargedParticles(const reco::Candidate& part) const
{
  // if stable, return 1 or 0
  if(part.status()==1) return (part.charge()!=0);
  // if unstable, call recursively on daughters
  int nch =0;
  for(reco::Candidate::const_iterator daughter=part.begin();daughter!=part.end(); ++daughter){
    nch += countChargedParticles(*daughter);
  }
  return nch;
}


bool
TtDecayChannelSelector::checkTauDecay(const reco::Candidate& tau) const
{
  bool leptonic = false;
  unsigned int nch = 0;
  // loop on tau decays, check for an electron or muon and count charged particles
  for(reco::Candidate::const_iterator daughter=tau.begin();daughter!=tau.end(); ++daughter){
    // if the tau daughter is a tau, it means the particle has still to be propagated.
    // In that case, return the result of the same method on that daughter.
    if(daughter->pdgId()==tau.pdgId()) return checkTauDecay(*daughter);
    // check for leptons
    leptonic |= (abs(daughter->pdgId())==11 || abs(daughter->pdgId())==13);
    // count charged particles
    nch += countChargedParticles(*daughter);
  }
  
  return ( (tauDecay_[Leptonic] && leptonic)            ||
           (tauDecay_[OneProng] && !leptonic && nch==1) ||
           (tauDecay_[ThreeProng] && !leptonic && nch>1)    );
}

void
TtDecayChannelSelector::parseDecayInput(Decay& chn1, Decay& chn2) const
{
  //---------------------------------------------
  //check for correct size of the input vectors
  //---------------------------------------------
  if( chn1.size()!=3 ){
    throw edm::Exception( edm::errors::Configuration, 
			  "'channel_1' must contain 3 values" );
  }
  if( chn2.size()!=3 ){
    throw edm::Exception( edm::errors::Configuration, 
			  "'channel_2' must contain 3 values" );
  }

  //---------------------------------------------
  //check for correct entries in input vectors
  //---------------------------------------------
  Decay::const_iterator idx1=chn1.begin(),idx2=chn2.begin(); 
  for( ; idx1!=chn1.end(), idx2!=chn2.end(); ++idx1, ++idx2){
    if( !(0<=(*idx1) && (*idx1)<=1) ){
      throw edm::Exception( edm::errors::Configuration, 
			    "'channel_1' may only contain values 0 or 1" );
    }
    if( !(0<=(*idx2) && (*idx2)<=1) ){
      throw edm::Exception( edm::errors::Configuration, 
			    "'channel_2' may only contain values 0 or 1" );
    }
  }

  return;
}

void
TtDecayChannelSelector::parseTauDecayInput(Decay& chn) const
{
  //---------------------------------------------
  //check for correct size of the input vector
  //---------------------------------------------
  if( chn.size()!=3 ){
    throw edm::Exception( edm::errors::Configuration, 
			  "'tauDecays' must contain 3 values" );
  }

  //---------------------------------------------
  //check for correct entries in input vectors
  //---------------------------------------------
  for(Decay::const_iterator idx=chn.begin() ; idx!=chn.end(); ++idx){
    if( !(0<=(*idx) && (*idx)<=1) ){
      throw edm::Exception( edm::errors::Configuration, 
			    "'tauDecays' may only contain values 0 or 1" );
    }
  }

  //---------------------------------------------
  //check for unambigous decay channel selection
  //---------------------------------------------
  if((count(chn.begin(), chn.end(), 1) == 0) && decay_[2]) {
    throw edm::Exception( edm::errors::Configuration, 
			  "No tau decay allowed while tau channels are allowed." );
  }
  return;
}

