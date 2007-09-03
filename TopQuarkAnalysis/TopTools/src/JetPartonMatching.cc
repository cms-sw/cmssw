//
// Author:  Jan Heyninck
// Created: Tue Apr  3 17:33:23 PDT 2007
//
// $Id: JetPartonMatching.cc,v 1.1 2007/07/04 16:51:38 heyninck Exp $
//
#include "TopQuarkAnalysis/TopTools/interface/JetPartonMatching.h"


// constructors
JetPartonMatching::JetPartonMatching() {}

JetPartonMatching::JetPartonMatching(vector<const reco::Candidate*> &q, vector<reco::GenJet> &j, int spAorDR){
  vector<const reco::Candidate*> js;
  for(size_t i=0; i<j.size(); i++) js.push_back(&(j[i]));
  partons = q;
  jets = js; 
  spaceAngleOrDeltaR = spAorDR;
  this->calculate();
}

JetPartonMatching::JetPartonMatching(vector<const reco::Candidate*> &q, vector<reco::CaloJet> &j, int spAorDR){
  vector<const reco::Candidate*> js;
  for(size_t i=0; i<j.size(); i++) js.push_back(&(j[i]));
  partons = q;
  jets = js;
  spaceAngleOrDeltaR = spAorDR;
  this->calculate();
}

JetPartonMatching::JetPartonMatching(vector<const reco::Candidate*> &q, vector<const reco::Candidate*> &j, int spAorDR){
  partons = q;
  jets = j;
  spaceAngleOrDeltaR = spAorDR;
  this->calculate();
}


// destructor
JetPartonMatching::~JetPartonMatching() {}


// calculate the matching
void 	JetPartonMatching::calculate(){

  //cout<<"@@@@====> parton size = "<< partons.size()<< "  jets size =  "<<jets.size() <<  endl;
  if(partons.size()>jets.size() || partons.size()==0) return;
  
  // Make a DeltaAngle map of size (nrjets x nrpartons)
  vector< pair<unsigned int,double> > qjAngle;
  for (unsigned int iq=0; iq<partons.size(); iq++){ 
    for(unsigned int ij=0; ij<jets.size(); ij++){
      double qjang = 0;
      if(spaceAngleOrDeltaR == 1){
        qjang = ROOT::Math::VectorUtil::Angle( jets[ij]->p4(), partons[iq]->p4() );
      }
      else if(spaceAngleOrDeltaR == 2){
        qjang = ROOT::Math::VectorUtil::DeltaR( jets[ij]->p4(), partons[iq]->p4() );
      }
      else {
        cout<<"JETPARTONMATCHING: incorrect choice of metric, select SpaceAngle (1) or DeltaR (2) in constructor !!!"<<endl;
      }		    
      qjAngle.push_back(pair<unsigned int,double>(iq*jets.size()+ij,qjang) );
      //cout<<"parton: "<<iq<<" jet:"<<ij<<" Angle: "<<qjang<<endl;
    }
  }
  
  // Iterative method that starts from the the smallest qjAngle value in the vector
  while(matching.size()<partons.size()){
    
    // find smallest angle value in vector
    unsigned int minIndex = 1;
    double minValue = 1000.;
    for(unsigned int a=0; a<qjAngle.size(); a++){
      if(qjAngle[a].second<minValue){
        minIndex = a;
	minValue = qjAngle[a].second;
      }
    }
    
    // find corresponding parton and jet index in original partons and jets vector
    unsigned int partonIndex = qjAngle[minIndex].first/jets.size();
    unsigned int jetIndex   = qjAngle[minIndex].first - jets.size()*partonIndex;
    //cout<<"  found min qjAngle between parton "<<partonIndex<<" and jet "<<jetIndex<<" ("<<qjAngle[minIndex].second<<")"<<endl;
    
    // add match to vector
    pair<unsigned int,unsigned int> match(partonIndex,jetIndex);
    matching.push_back(match);
    
    // remove all qjAngle values for the already matched parton 
    //cout<<"    erased values ";
    for(unsigned int a=0; a<qjAngle.size(); a++){
      unsigned int pIndex = qjAngle[a].first/jets.size();
      unsigned int jIndex = qjAngle[a].first-jets.size()*pIndex;
      if( (pIndex == partonIndex) || (jIndex == jetIndex) ) {
        //cout<<qjAngle[a].second<<" ";
        qjAngle.erase(qjAngle.begin()+a,qjAngle.begin()+a+1); 
	--a;
      }
    }
    //cout<<endl;
  }
  //cout<<endl<<endl<<endl;  
}

    
    
    
// member to get the matching jetIndex for a certain partonIndex  
int   	JetPartonMatching::getMatchForParton(unsigned int partonIndex){
  int jetIndex = -999;
  for(size_t i=0; i<matching.size(); i++){
    if(matching[i].first == partonIndex) jetIndex = matching[i].second;
  }
  return jetIndex;
}
    
 
 
    
// member to get the angle between a certain parton and its best matched jet    
double 	JetPartonMatching::getAngleForParton(unsigned int partonIndex){
  unsigned int iq = partonIndex;
  int ij = this->getMatchForParton(partonIndex);
  double qjAngle = -999.;
  if(ij > -1){
    if(spaceAngleOrDeltaR == 1){
      qjAngle = ROOT::Math::VectorUtil::Angle( jets[ij]->p4(), partons[iq]->p4() );
    }
    else if(spaceAngleOrDeltaR == 2){
      qjAngle = ROOT::Math::VectorUtil::DeltaR( jets[ij]->p4(), partons[iq]->p4() );
    }
    else {
      cout<<"JETPARTONMATCHING: incorrect choice of metric, select SpaceAngle (1) or DeltaR (2) in constructor !!!"<<endl;
    }
  }
  return qjAngle;
}



// member to get the sum of the angles between parton and matched jet
double 	JetPartonMatching::getSumAngles(){
  double sumAngles = 0;
  for(size_t i=0; i<matching.size(); i++){
    sumAngles += this->getAngleForParton(i);
  }
  return sumAngles;
}
