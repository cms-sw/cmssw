//
// Author:  Jan Heyninck
// Created: Tue Apr  3 17:33:23 PDT 2007
//
// $Id: TopObjectResolutionCalc.cc,v 1.6 2007/08/06 12:44:29 tsirig Exp $
//
#include "TopQuarkAnalysis/TopObjectResolutions/interface/TopObjectResolutionCalc.h"

// constructor with path; default should not be used
TopObjectResolutionCalc::TopObjectResolutionCalc(TString resopath,bool useNNet = false):useNN(useNNet) {
  std::cout << "=== Constructing a TopObjectResolutionCalc... " << std::endl; 
  resoFile = new TFile(resopath);
  if (!resoFile) std::cout<<"### No resolutions fits for this object available... ###"<<std::endl;
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
  if(useNN) {
    for(Int_t ro=0; ro<6; ro++) {
      TString obsName = objectType; obsName += resObsName[ro]; obsName += "_NN"; obsName += def[0];
      network[ro] = (TMultiLayerPerceptron*) resoFile->GetKey(obsName)->ReadObj();
    }
  } else {
   for(Int_t ro=0; ro<6; ro++) {
    if(objectType == "muon" ||objectType == "electron" ||objectType == "lJets" ||objectType == "bJets" || objectType == "tau"){
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
  }
  std::cout << "=== done." << std::endl;
}


// destructor
TopObjectResolutionCalc::~TopObjectResolutionCalc() {
  delete resoFile;
}

double TopObjectResolutionCalc::getObsRes(int obs, int eta, double eT){
  if(useNN) throw edm::Exception( edm::errors::LogicError, 
                                  "TopObjectResolutionCalc::getObsRes should never be called when using a NN for resolutions." );
  double res = fResVsET[obs][eta].Eval(eT);
  return res;
}

void  TopObjectResolutionCalc::operator()(TopElectron& obj){
  if(useNN) {
    double v[2];
    v[0]=obj.et();
    v[1]=obj.eta();
    obj.setResPinv(  network[0]->Evaluate(0,v ));
    obj.setResD(     network[1]->Evaluate(0,v ));	
    obj.setResTheta( network[2]->Evaluate(0,v ));	 
    obj.setResPhi(   network[3]->Evaluate(0,v ));	
    obj.setResET(    network[4]->Evaluate(0,v ));	
    obj.setResEta(   network[5]->Evaluate(0,v ));
  } else {
    double etabin[11] = {0,0.1625,0.325,0.5,0.675,0.8625,1.0625,1.275,1.55,1.85,2.5}; 
    int bin = 9;
    for(Int_t i=0; i<10; i++) {
      if(fabs(obj.eta()) > etabin[i] && fabs(obj.eta()) < etabin[i+1]) bin = i;
    }
    obj.setResPinv(  this->getObsRes(0,bin,obj.et()) );	
    obj.setResD(     this->getObsRes(1,bin,obj.et()) );	
    obj.setResTheta( this->getObsRes(2,bin,obj.et()) );	 
    obj.setResPhi(   this->getObsRes(3,bin,obj.et()) );	
    obj.setResET(    this->getObsRes(4,bin,obj.et()) );	
    obj.setResEta(   this->getObsRes(5,bin,obj.et()) );
  }
}

void  TopObjectResolutionCalc::operator()(TopMuon& obj){
  if(useNN) {
    double v[2];
    v[0]=obj.et();
    v[1]=obj.eta();
    obj.setResPinv(  network[0]->Evaluate(0,v ));
    obj.setResD(     network[1]->Evaluate(0,v ));	
    obj.setResTheta( network[2]->Evaluate(0,v ));	 
    obj.setResPhi(   network[3]->Evaluate(0,v ));	
    obj.setResET(    network[4]->Evaluate(0,v ));	
    obj.setResEta(   network[5]->Evaluate(0,v ));
  } else {
    double etabin[11] = {0,0.175,0.35,0.5125,0.6875,0.875,1.075,1.3,1.575,1.9,2.5};
    int bin = 9;
    for(Int_t i=0; i<10; i++) {
      if(fabs(obj.eta()) > etabin[i] && fabs(obj.eta()) < etabin[i+1]) bin = i;
    }
    obj.setResPinv(  this->getObsRes(0,bin,obj.et()) );	
    obj.setResD(     this->getObsRes(1,bin,obj.et()) );	
    obj.setResTheta( this->getObsRes(2,bin,obj.et()) );	 
    obj.setResPhi(   this->getObsRes(3,bin,obj.et()) );	
    obj.setResET(    this->getObsRes(4,bin,obj.et()) );	
    obj.setResEta(   this->getObsRes(5,bin,obj.et()) );
  }
}

void  TopObjectResolutionCalc::operator()(TopJet& obj){
  if(useNN) {
    double v[2];
    v[0]=obj.et();
    v[1]=obj.eta();
    obj.setResPinv(  network[0]->Evaluate(0,v ));
    obj.setResD(     network[1]->Evaluate(0,v ));	
    obj.setResTheta( network[2]->Evaluate(0,v ));	 
    obj.setResPhi(   network[3]->Evaluate(0,v ));	
    obj.setResET(    network[4]->Evaluate(0,v ));	
    obj.setResEta(   network[5]->Evaluate(0,v ));
  } else {
    double etabin[11] = {0,0.175,0.3625,0.5375,0.725,0.925,1.15,1.4,1.7,2.075,2.5};
    int bin = 9;
    for(Int_t i=0; i<10; i++) {
      if(fabs(obj.eta()) > etabin[i] && fabs(obj.eta()) < etabin[i+1]) bin = i;
    }
    obj.setResPinv(  this->getObsRes(0,bin,obj.et()) );	
    obj.setResD(     this->getObsRes(1,bin,obj.et()) );	
    obj.setResTheta( this->getObsRes(2,bin,obj.et()) );	 
    obj.setResPhi(   this->getObsRes(3,bin,obj.et()) );	
    obj.setResET(    this->getObsRes(4,bin,obj.et()) );	
    obj.setResEta(   this->getObsRes(5,bin,obj.et()) );
  }
}

void  TopObjectResolutionCalc::operator()(TopMET& obj){
  if(useNN) {
    double v[2];
    v[0]=obj.et();
    v[1]=obj.eta();
    obj.setResPinv(  network[0]->Evaluate(0,v ));
    obj.setResD(     network[1]->Evaluate(0,v ));	
    obj.setResPhi(   network[3]->Evaluate(0,v ));	
    obj.setResET(    network[4]->Evaluate(0,v ));	
    obj.setResTheta( 1000000.  );   			// Total freedom
    obj.setResEta(   1000000.  );    			// Total freedom
  } else {
    obj.setResPinv(  this->getObsRes(0,0,obj.et())  );
    obj.setResD(     this->getObsRes(1,0,obj.et())  );
    obj.setResTheta( 1000000.  );   			// Total freedom
    obj.setResPhi(   this->getObsRes(3,0,obj.et())  );
    obj.setResET(    this->getObsRes(4,0,obj.et())  );
    obj.setResEta(   1000000.  );    			// Total freedom
  }
}

void  TopObjectResolutionCalc::operator()(TopTau& obj){
  if(useNN) {
    double v[2];
    v[0]=obj.et();
    v[1]=obj.eta();
    obj.setResPinv(  network[0]->Evaluate(0,v ));
    obj.setResD(     network[1]->Evaluate(0,v ));	
    obj.setResTheta( network[2]->Evaluate(0,v ));	 
    obj.setResPhi(   network[3]->Evaluate(0,v ));	
    obj.setResET(    network[4]->Evaluate(0,v ));	
    obj.setResEta(   network[5]->Evaluate(0,v ));
  } else {
    double etabin[11] = {0,0.175,0.3625,0.5375,0.725,0.925,1.15,1.4,1.7,2.075,2.5};
    int bin = 9;
    for(Int_t i=0; i<10; i++) {
      if(fabs(obj.eta()) > etabin[i] && fabs(obj.eta()) < etabin[i+1]) bin = i;
    }
    obj.setResPinv(  this->getObsRes(0,bin,obj.et()) );	
    obj.setResD(     this->getObsRes(1,bin,obj.et()) );	
    obj.setResTheta( this->getObsRes(2,bin,obj.et()) );	 
    obj.setResPhi(   this->getObsRes(3,bin,obj.et()) );	
    obj.setResET(    this->getObsRes(4,bin,obj.et()) );	
    obj.setResEta(   this->getObsRes(5,bin,obj.et()) );
  }
}

