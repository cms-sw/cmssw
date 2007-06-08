//
// Author:  Jan Heyninck
// Created: Tue Apr  3 17:33:23 PDT 2007
//
// $Id: TopObjectResolutionCalc.cc,v 1.1 2007/05/08 14:03:05 heyninck Exp $
//
#include "TopQuarkAnalysis/TopObjectResolutions/interface/TopObjectResolutionCalc.h"

// constructor with path; default should not be used
TopObjectResolutionCalc::TopObjectResolutionCalc(TString resopath) {
  cout << "=== Constructing a TopObjectResolutionCalc... " << endl; 
  resoFile = new TFile(resopath);
  if (!resoFile) cout<<"### No resolutions fits for this object available... ###"<<endl;
 TString objectType = resopath;
  while(objectType.Index("/")>0) objectType.Remove(0,objectType.Index("/")+1);
  objectType.Remove(0,objectType.Index("_")+1);
  if(objectType.Index("_")>0){
    objectType.Remove(objectType.Index("_"),objectType.Length());
  }
  else{
    objectType.Remove(objectType.Index("."),objectType.Length());
  }
  TString  resObsName[6] = {"pres","eres","thres","phres","etres","etares"};
  TString def[1] = {"_abs"};
  for(Int_t ro=0; ro<6; ro++) {
    if(objectType == "muon" ||objectType == "electron" ||objectType == "lJets" ||objectType == "bJets"){
      for(Int_t i=0; i<10; i++) { 
        TString obsName = objectType; obsName += resObsName[ro]; obsName += "_etabin"; obsName += i; obsName += def[0]; 
        TH1F *tmp = (TH1F*) (resoFile->GetKey(obsName)->ReadObj());
        fResVsET[ro][i] = (TF1)(*(tmp -> GetFunction("F_"+obsName)));
      } 
    } else{
      TString obsName = objectType; obsName += resObsName[ro]; obsName += "_etabin0"; obsName += def[0]; 
      TH1F *tmp = (TH1F*) (resoFile->GetKey(obsName)->ReadObj());
      fResVsET[ro][0] = (TF1)(*(tmp -> GetFunction("F_"+obsName)));
    }
  }
  cout << "=== done." << endl;

}


// destructor
TopObjectResolutionCalc::~TopObjectResolutionCalc() {
  delete resoFile;
}

double TopObjectResolutionCalc::getObsRes(int obs, int eta, double eT){
  double res = fResVsET[obs][eta].Eval(eT);
  return res;
}

void  TopObjectResolutionCalc::operator()(TopElectron& obj){
  double etabin[11] = {0,0.1625,0.325,0.5,0.675,0.8625,1.0625,1.275,1.55,1.85,2.5}; 
  for(Int_t i=0; i<10; i++){
    if(fabs(obj.eta()) > etabin[i] && fabs(obj.eta()) < etabin[++i]){ 
      obj.setResPinv(  this->getObsRes(0,i,obj.et()) );	
      obj.setResD(     this->getObsRes(1,i,obj.et()) );	
      obj.setResTheta( this->getObsRes(2,i,obj.et()) );	 
      obj.setResPhi(   this->getObsRes(3,i,obj.et()) );	
      obj.setResET(    this->getObsRes(4,i,obj.et()) );	
      obj.setResEta(   this->getObsRes(5,i,obj.et()) );	
    } 
  }
}
void  TopObjectResolutionCalc::operator()(TopMuon& obj){
  double etabin[11] = {0,0.175,0.35,0.5125,0.6875,0.875,1.075,1.3,1.575,1.9,2.5};
  for(Int_t i=0; i<10; i++){
    if(fabs(obj.eta()) > etabin[i] && fabs(obj.eta()) < etabin[++i]){ 
      obj.setResPinv(  this->getObsRes(0,i,obj.et()) );	 
      obj.setResD(     this->getObsRes(1,i,obj.et()) );	 
      obj.setResTheta( this->getObsRes(2,i,obj.et()) );	 
      obj.setResPhi(   this->getObsRes(3,i,obj.et()) );	 
      obj.setResET(    this->getObsRes(4,i,obj.et()) );	 
      obj.setResEta(   this->getObsRes(5,i,obj.et()) );
      }	 
  }
}
void  TopObjectResolutionCalc::operator()(TopJet& obj){
  double etabin[11] = {0,0.175,0.3625,0.5375,0.725,0.925,1.15,1.4,1.7,2.075,2.5};
  for(Int_t i=0; i<10; i++){
    if(fabs(obj.eta()) > etabin[i] && fabs(obj.eta()) < etabin[++i]){ 
      obj.setResPinv(  this->getObsRes(0,i,obj.et()) );	
      obj.setResD(     this->getObsRes(1,i,obj.et()) );	
      obj.setResTheta( this->getObsRes(2,i,obj.et()) );	   
      obj.setResPhi(   this->getObsRes(3,i,obj.et()) );	
      obj.setResET(    this->getObsRes(4,i,obj.et()) );	
      obj.setResEta(   this->getObsRes(5,i,obj.et()) );	
    } 
  }
}
void  TopObjectResolutionCalc::operator()(TopMET& obj){
  obj.setResPinv(  this->getObsRes(0,0,obj.et())  );
  obj.setResD(     this->getObsRes(1,0,obj.et())  );
  obj.setResTheta( 1000000.  );   			// Total freedom
  obj.setResPhi(   this->getObsRes(3,0,obj.et())  );
  obj.setResET(    this->getObsRes(4,0,obj.et())  );
  obj.setResEta(   1000000.  );    			// Total freedom
}
