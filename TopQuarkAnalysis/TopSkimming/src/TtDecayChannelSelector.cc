#include <vector>
#include <iostream>

#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "TopQuarkAnalysis/TopSkimming/interface/TtDecayChannelSelector.h"

TtDecayChannelSelector::TtDecayChannelSelector(const edm::ParameterSet& cfg):
  invert_( cfg.getParameter<bool>("invert") ),
  oneLep_( cfg.getParameter<std::vector<int> >("one_lepton_channels")),
  twoLep_( cfg.getParameter<std::vector<int> >("two_lepton_channels"))
{
  if( oneLep_.size()!=3 )
    throw edm::Exception( edm::errors::Configuration, "'one_lepton_channel' must contain 3 values" );
  if( twoLep_.size()!=3 )
    throw edm::Exception( edm::errors::Configuration, "'two_lepton_channel' must contain 3 values" );

  Decay::const_iterator leaf1=oneLep_.begin(),leaf2=twoLep_.begin(); 
  for( ; leaf1!=oneLep_.end(), leaf2!=twoLep_.end(); ++leaf1, ++leaf2){
    if( !(0<=(*leaf1) && (*leaf1)<=1) )
      throw edm::Exception( edm::errors::Configuration, "values of 'one_lepton_channel' may only be 0 or 1" );
    if( !(0<=(*leaf2) && (*leaf2)<=1) )
      throw edm::Exception( edm::errors::Configuration, "values of 'two_lepton_channel' may only be 0 or 1" );
  }

  channel_=0;
  if( count(oneLep_.begin(), oneLep_.end(), 1)>0 ){ channel_+=1; }
  if( count(twoLep_.begin(), twoLep_.end(), 1)>0 ){ channel_+=2; }

  if( channel_>2 )
    throw edm::Exception( edm::errors::Configuration, "switch one of the 'lepton_channel`s' to (0,0,0), please" );
}

TtDecayChannelSelector::~TtDecayChannelSelector()
{ } 

bool
TtDecayChannelSelector::operator()(const reco::CandidateCollection& parts) const
{
  int iTop=0,iBeauty=0,iElec=0,iMuon=0,iTau=0;
  reco::CandidateCollection::const_iterator top=parts.begin();
  for(; top!=parts.end(); ++top){ 
    if( reco::status(*top)==3 && abs((*top).pdgId())==6 ){
      ++iTop;
      reco::Candidate::const_iterator td=(*top).begin();
      for(; td!=(*top).end(); ++td){
	if( reco::status(*td)==3 && abs((*td).pdgId())==5 )
	  {++iBeauty;}
	if( reco::status(*td)==3 && abs((*td).pdgId())==24 ){
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
	  ( ((oneLep_[Elec]==1) && (iElec==iLep))||
	    ((oneLep_[Muon]==1) && (iMuon==iLep))||
	    ((oneLep_[Tau ]==1) && (iTau ==iLep)))
	  ||
	  (channel_==2) &&
	  ( ((twoLep_[Elec]==1) && (iElec==iLep))||
	    ((twoLep_[Muon]==1) && (iMuon==iLep))||
	    ((twoLep_[Tau ]==1) && (iTau ==iLep))))
	accept=true;
    accept=( (!invert_&& accept) || (!(!invert_)&& !accept) );
  }
  else{
    edm::LogWarning ( "NoVtbDecay" ) << "Decay is not via Vtb";
  }
 if(accept){
   std::cout << "iElec  : " << iElec   << "\n"
	     << "iMuon  : " << iMuon   << "\n"
	     << "iTau   : " << iTau    << "\n"
	     << std::endl;
 }  
 return accept;
}
