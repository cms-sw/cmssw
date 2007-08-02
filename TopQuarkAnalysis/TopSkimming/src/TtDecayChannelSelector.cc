#include <iostream>

#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "TopQuarkAnalysis/TopSkimming/interface/TtDecayChannelSelector.h"

TtDecayChannelSelector::TtDecayChannelSelector(const edm::ParameterSet& cfg):
  invert_ ( cfg.getParameter<bool>("invert" ) )
{
  Decay chn1 = cfg.getParameter<Decay>("channel_1");
  Decay chn2 = cfg.getParameter<Decay>("channel_2");

  parseDecayInput(chn1, chn2);

  channel_=0;
  //determine decay channel
  if( count(chn1.begin(), chn1.end(), 1) > 0 ){ ++channel_; }
  if( count(chn2.begin(), chn2.end(), 1) > 0 ){ ++channel_; }

  summed_=0;
  //fill vector of allowed leptons
  Decay::const_iterator idx1=chn1.begin(),idx2=chn2.begin();
  for( ; idx1!=chn1.end(), idx2!=chn2.end(); ++idx1, ++idx2){
    summed_+=(*idx1)+(*idx2);
    decay_.push_back( (*idx1)+(*idx2) );
  }
}

TtDecayChannelSelector::~TtDecayChannelSelector()
{ } 

bool
TtDecayChannelSelector::operator()(const reco::CandidateCollection& parts) const
{
  int iTop=0,iBeauty=0,iElec=0,iMuon=0,iTau=0;
  reco::CandidateCollection::const_iterator top=parts.begin();
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
	    if( abs((*wd).pdgId())==15 ){++iTau; }
	  }
	}
      }
    }
  }
  int iLep=iElec+iMuon+iTau;

  bool accept=false;
  if( (iTop==2) && (iBeauty==2) ){
    if( channel_==iLep ){
      if( channel_==0 ){
	accept=true;
      }
      if( channel_==1 ){
	if( summed_==channel_ ){
	  // exactly one lepton channel is selected
	  if( (iElec==decay_[Elec] && 
	       iMuon==decay_[Muon] && 
	       iTau ==decay_[Tau ]) ){
	    accept=true;
	  }
	}
	else{
	  // more than one lepton channel is selected
	  // split in two cases: two & three channels
	  if( ((decay_[Elec]>0 && decay_[Muon]>0) && ((iElec+iMuon)==channel_)) ||
	      ((decay_[Muon]>0 && decay_[Tau ]>0) && ((iMuon+iTau )==channel_)) ||
	      ((decay_[Tau ]>0 && decay_[Elec]>0) && ((iTau +iElec)==channel_)) ){
	    accept=true;
	  }
	  if( ((decay_[Elec]>0 && 
		decay_[Muon]>0 && 
		decay_[Tau ]>0)&& ((iElec+iMuon+iTau)==channel_)) ){
	    accept=true;
	  }
	}
      }
      if( channel_==2 ){
	if( summed_==channel_ ){
	  // one or two separate lepton channels are 
	  // selected; covers two cases: same flavor 
	  // (e.g.ee) and separate flavor (e.g.emu)
	  if( (decay_[Elec]>1 && iElec==channel_) || 
	      (decay_[Muon]>1 && iMuon==channel_) || 
	      (decay_[Tau ]>1 && iTau ==channel_) ){
	    accept=true;
	  }
	  if( ((decay_[Elec]>0 && decay_[Muon]>0) && (iElec==1 && iMuon==1)) ||
	      ((decay_[Muon]>0 && decay_[Tau ]>0) && (iMuon==1 && iTau ==1)) ||
	      ((decay_[Tau ]>0 && decay_[Elec]>0) && (iTau ==1 && iElec==1)) ){
	    accept=true;
	  }
	}
	else{
	  // more than two lepton channel or two lepton 
	  // channel is selected where >1 lepton is re-
	  // quired definitely to be present
	  if( ((decay_[Elec]>0 && decay_[Muon]>0) && ((iElec+iMuon)==channel_) ) ||
	      ((decay_[Muon]>0 && decay_[Tau ]>0) && ((iMuon+iTau )==channel_) ) ||
	      ((decay_[Tau ]>0 && decay_[Elec]>0) && ((iTau +iElec)==channel_) ) ){
	    accept=true;
	  }
	  if( ((decay_[Elec]>0 && 
		decay_[Muon]>0 && 
		decay_[Tau ]>0)&& ((iElec+iMuon+iTau)==channel_)) ){
	    accept=true;
	  }
	  if( (decay_[Elec]==1 && iElec==channel_) ||
	      (decay_[Muon]==1 && iMuon==channel_) ||
	      (decay_[Tau ]==1 && iTau ==channel_) ){
	    accept=false;
	  }
	}
      }
    }
    accept=( (!invert_&& accept) || (!(!invert_)&& !accept) );
  }
  else{
    edm::LogWarning ( "NoVtbDecay" ) << "Decay is not via Vtb";
  }
//   if(accept){
//     std::cout << "iElec  : " << iElec   << "\n"
// 	      << "iMuon  : " << iMuon   << "\n"
// 	      << "iTau   : " << iTau    << "\n"
// 	      << std::endl;
//   }  
  return accept;
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

  //---------------------------------------------
  //check for unambigous decay channel selection
  //---------------------------------------------
  if( (count(chn1.begin(), chn1.end(), 1) == 0) &&
      (count(chn2.begin(), chn2.end(), 1)  > 0) ){
    throw edm::Exception( edm::errors::Configuration, 
			  "found dilepton channel being selected w/o first lepton" );
  }
  return;
}

