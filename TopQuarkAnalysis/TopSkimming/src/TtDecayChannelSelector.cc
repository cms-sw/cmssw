#include <vector>
#include <iostream>

#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "TopQuarkAnalysis/TopSkimming/interface/TtDecayChannelSelector.h"

TtDecayChannelSelector::TtDecayChannelSelector(const edm::ParameterSet& cfg):
  topChn_( cfg.getParameter<int >("chn") ),
  invChn_( cfg.getParameter<bool>("invert_chn") ),
  subChn_( cfg.getParameter<bool>("enable_subchn")),
  invSub_( cfg.getParameter<bool>("invert_subchn")),
  lepVec_( cfg.getParameter<std::vector<int> >("subchn"))
{
  if( !(0<=topChn_ && topChn_<=2) ){
    throw edm::Exception( edm::errors::Configuration,
			  "value of 'selectChn' may only be between 0 and 2" );
  }
  if( lepVec_.size()!=3 ){
    throw edm::Exception( edm::errors::Configuration,
			  "vector<int> 'leptonCannel' must contain 3 values" );
  }
  std::vector<int>::const_iterator v;
  for(v=lepVec_.begin(); v<lepVec_.end(); ++v)
    if( !(0<=(*v) && (*v)<=1) )
      throw edm::Exception( edm::errors::Configuration,
			    "values of vector<int> 'leptonChannel' may only be 0 or 1" );
  
  //invChn_ switches subChn_ automatically off
  if(topChn_==0) subChn_= false;
  if(invChn_   ) subChn_= false;
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
    if( iLep==topChn_){
      accept=true;
      if( subChn_&& topChn_>0 ){
	accept=((iElec==iLep && lepVec_[0]) || 
		(iMuon==iLep && lepVec_[1]) ||
		(iTau ==iLep && lepVec_[2]));
	accept=((  !invSub_  && accept)|| 
		(!(!invSub_) &&!accept));
      }
    }
    accept=( (!invChn_&& accept) || (!(!invChn_)&& !accept) );
  }
  else{
    edm::LogWarning ( "NoVtbDecay" ) << "Decay is not via Vtb";
  }
//  if(accept){
//    std::cout << "iElec: " << iElec << "\n"
//	      << "iMuon: " << iMuon << "\n"
//	      << "iTau : " << iTau  << "\n"
//	      << "accept " << accept<< "\n"
//	      << std::endl;
//  }  
  return accept;
}
