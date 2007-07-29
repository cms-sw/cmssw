#include <iostream>

#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "TopQuarkAnalysis/TopSkimming/interface/TtDecayChannelSelector.h"

TtDecayChannelSelector::TtDecayChannelSelector(const edm::ParameterSet& cfg):
  invert_( cfg.getParameter<bool>("invert") ),
  chn1_( cfg.getParameter<std::vector<int> >("channel_1")),
  chn2_( cfg.getParameter<std::vector<int> >("channel_2"))
{
  //check for correct size of the input vectors
  if( chn1_.size()!=3 )
    throw edm::Exception( edm::errors::Configuration, "'channel_1' must contain 3 values" );
  if( chn2_.size()!=3 )
    throw edm::Exception( edm::errors::Configuration, "'channel_2' must contain 3 values" );

  //check for correct entries in the input vectors
  Decay::const_iterator leaf1=chn1_.begin(),leaf2=chn2_.begin(); 
  for( ; leaf1!=chn1_.end(), leaf2!=chn2_.end(); ++leaf1, ++leaf2){
    if( !(0<=(*leaf1) && (*leaf1)<=1) )
      throw edm::Exception( edm::errors::Configuration, "'channel_1' may only contain values 0 or 1" );
    if( !(0<=(*leaf2) && (*leaf2)<=1) )
      throw edm::Exception( edm::errors::Configuration, "'channel_2' may only contain values 0 or 1" );
    decay_.push_back( (*leaf1)+(*leaf2) );
  }

  //check for unambigous decay channel selection
  if( (count(chn1_.begin(), chn1_.end(), 1) == 0) &&
      (count(chn2_.begin(), chn2_.end(), 1)  > 0) ){
    throw edm::Exception( edm::errors::Configuration, "found dilepton channel being selected w/o first lepton" );
  }

  channel_=0;
  //determine decay channel
  if( count(chn1_.begin(), chn1_.end(), 1) > 0 ){ ++channel_; }
  if( count(chn2_.begin(), chn2_.end(), 1) > 0 ){ ++channel_; }
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
    if( channel_==iLep )
      if( (channel_==0)
	  ||
	  (channel_==1) &&
	  ( ((chn1_[Elec]==1 || chn2_[Elec]==1) && (iElec==iLep))||
	    ((chn1_[Muon]==1 || chn2_[Muon]==1) && (iMuon==iLep))||
	    ((chn1_[Tau ]==1 || chn2_[Tau ]==1) && (iTau ==iLep)))
	  ||
	  (channel_==2) &&
	  ( ((chn1_[Elec]==1 && chn2_[Elec]==1) && (iElec==iLep))||
	    ((chn1_[Muon]==1 && chn2_[Muon]==1) && (iMuon==iLep))||
	    ((chn1_[Tau ]==1 && chn2_[Tau ]==1) && (iTau ==iLep))||
	    (( (decay_[Elec]<2 || decay_[Muon]<2) &&
	       ( (chn1_[Elec]==1 && chn2_[Muon]==1) || (chn1_[Muon]==1 && chn2_[Elec]==1) ) && 
	       ( iElec==1 && iMuon==1 )
	       ) ||
	     ( (decay_[Elec]<2 || decay_[Tau ]<2) &&
	       ( (chn1_[Elec]==1 && chn2_[Tau ]==1) || (chn1_[Tau ]==1 && chn2_[Elec]==1) ) && 
	       ( iElec==1 && iTau ==1 )
	       ) ||	    
	     ( (decay_[Muon]<2 || decay_[Tau ]<2) &&
	       ( (chn1_[Muon]==1 && chn2_[Tau ]==1) || (chn1_[Tau ]==1 && chn2_[Muon]==1) ) && 
	       ( iMuon==1 && iTau ==1 )
	       )
	     )
	    )
	  )
	accept=true;
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
