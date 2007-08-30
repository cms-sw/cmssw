//
// Author:  Jan Heyninck
// Created: Tue Apr  3 17:33:23 PDT 2007
//
// $Id: TopObjectResolutionCalc.cc,v 1.5.2.2 2007/08/30 17:08:41 heyninck Exp $
//
#include "TopQuarkAnalysis/TopObjectResolutions/interface/TopObjectResolutionCalc.h"

// constructor with path; default should not be used
TopObjectResolutionCalc::TopObjectResolutionCalc(TString resopath,bool useNNet = false):useNN(useNNet) {
  std::cout << "=== Constructing a TopObjectResolutionCalc... " << std::endl; 
  resoFile = new TFile(resopath);
  if (!resoFile) std::cout<<"### No resolutions fits for this file available: "<<resopath<<"... ###"<<std::endl;
  TString  resObsName[8] = {"_ares","_bres","_cres","_dres","_thres","_phres","_etres","_etares"};
  
  TList* keys = resoFile->GetListOfKeys();
  TIter nextitem(keys);
  TKey* key = NULL;
  while((key = (TKey*)nextitem())) {
    TString name = key->GetName();
    if(useNN) {
      for(Int_t ro=0; ro<8; ro++) {
        TString obsName = obsName += resObsName[ro]; obsName += "_NN";
        if(name.Contains(obsName)){
	  network[ro] = (TMultiLayerPerceptron*) resoFile->GetKey(name)->ReadObj();
	}
      }
    }
    else 
    { 
      if(name.Contains("etabin") && (!name.Contains("etbin"))) {
        for(int p=0; p<8; p++){
          if(name.Contains(resObsName[p])){
            TString etabin = name; etabin.Remove(0,etabin.Index("_")+1); etabin.Remove(0,etabin.Index("_")+7);
            int etaBin = etabin.Atoi();
            TH1F *tmp = (TH1F*) (resoFile->GetKey(name)->ReadObj());
            fResVsET[p][etaBin] = (TF1)(*(tmp -> GetFunction("F_"+name)));
	  }
        }
      }
    }
  }
  // find etabin values
  TH1F *tmpEta = (TH1F*) (resoFile->GetKey("hEtaBins")->ReadObj());
  for(int b=1; b<=tmpEta->GetNbinsX(); b++) etabinVals.push_back(tmpEta->GetXaxis()->GetBinLowEdge(b));
  etabinVals.push_back(tmpEta->GetXaxis()->GetBinUpEdge(tmpEta->GetNbinsX()));
  cout<<"Found "<<etabinVals.size()-1<< " eta-bins with edges: ( ";
  for(size_t u=0; u<etabinVals.size(); u++) cout<<etabinVals[u]<<", ";
  cout<<"\b\b )"<<endl;
  
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

int TopObjectResolutionCalc::getEtaBin(double eta){
  int nrEtaBins = etabinVals.size()-1;
  int bin = nrEtaBins-1;
  for(int i=0; i<nrEtaBins; i++) {
    if(fabs(eta) > etabinVals[i] && fabs(eta) < etabinVals[i+1]) bin = i;
  }
  return bin;
}

void  TopObjectResolutionCalc::operator()(TopElectron& obj){
  if(useNN) {
    double v[2];
    v[0]=obj.et();
    v[1]=obj.eta();
    obj.setResA(     network[0]->Evaluate(0,v ));
    obj.setResB(     network[1]->Evaluate(0,v ));
    obj.setResC(     network[2]->Evaluate(0,v ));
    obj.setResD(     network[3]->Evaluate(0,v ));	
    obj.setResTheta( network[4]->Evaluate(0,v ));	 
    obj.setResPhi(   network[5]->Evaluate(0,v ));	
    obj.setResET(    network[6]->Evaluate(0,v ));	
    obj.setResEta(   network[7]->Evaluate(0,v ));
  } else {
    int bin = this->getEtaBin(obj.eta());
    obj.setResA(     this->getObsRes(0,bin,obj.et()) );
    obj.setResB(     this->getObsRes(1,bin,obj.et()) );
    obj.setResC(     this->getObsRes(2,bin,obj.et()) );	
    obj.setResD(     this->getObsRes(3,bin,obj.et()) );	
    obj.setResTheta( this->getObsRes(4,bin,obj.et()) );	 
    obj.setResPhi(   this->getObsRes(5,bin,obj.et()) );	
    obj.setResET(    this->getObsRes(6,bin,obj.et()) );	
    obj.setResEta(   this->getObsRes(7,bin,obj.et()) );
  }
}

void  TopObjectResolutionCalc::operator()(TopMuon& obj){
  if(useNN) {
    double v[2];
    v[0]=obj.et();
    v[1]=obj.eta();
    obj.setResA(     network[0]->Evaluate(0,v ));
    obj.setResB(     network[1]->Evaluate(0,v ));
    obj.setResC(     network[2]->Evaluate(0,v ));
    obj.setResD(     network[3]->Evaluate(0,v ));	
    obj.setResTheta( network[4]->Evaluate(0,v ));	 
    obj.setResPhi(   network[5]->Evaluate(0,v ));	
    obj.setResET(    network[6]->Evaluate(0,v ));	
    obj.setResEta(   network[7]->Evaluate(0,v ));
  } else {
    int bin = this->getEtaBin(obj.eta());
    obj.setResA(     this->getObsRes(0,bin,obj.et()) );
    obj.setResB(     this->getObsRes(1,bin,obj.et()) );
    obj.setResC(     this->getObsRes(2,bin,obj.et()) );	
    obj.setResD(     this->getObsRes(3,bin,obj.et()) );	
    obj.setResTheta( this->getObsRes(4,bin,obj.et()) );	 
    obj.setResPhi(   this->getObsRes(5,bin,obj.et()) );	
    obj.setResET(    this->getObsRes(6,bin,obj.et()) );	
    obj.setResEta(   this->getObsRes(7,bin,obj.et()) );
  }
}

void  TopObjectResolutionCalc::operator()(TopJet& obj){
  if(useNN) {
    double v[2];
    v[0]=obj.et();
    v[1]=obj.eta();
    obj.setResA(     network[0]->Evaluate(0,v ));
    obj.setResB(     network[1]->Evaluate(0,v ));
    obj.setResC(     network[2]->Evaluate(0,v ));
    obj.setResD(     network[3]->Evaluate(0,v ));	
    obj.setResTheta( network[4]->Evaluate(0,v ));	 
    obj.setResPhi(   network[5]->Evaluate(0,v ));	
    obj.setResET(    network[6]->Evaluate(0,v ));	
    obj.setResEta(   network[7]->Evaluate(0,v ));
  } else {
    int bin = this->getEtaBin(obj.eta());
    obj.setResA(     this->getObsRes(0,bin,obj.et()) );
    obj.setResB(     this->getObsRes(1,bin,obj.et()) );
    obj.setResC(     this->getObsRes(2,bin,obj.et()) );	
    obj.setResD(     this->getObsRes(3,bin,obj.et()) );	
    obj.setResTheta( this->getObsRes(4,bin,obj.et()) );	 
    obj.setResPhi(   this->getObsRes(5,bin,obj.et()) );	
    obj.setResET(    this->getObsRes(6,bin,obj.et()) );	
    obj.setResEta(   this->getObsRes(7,bin,obj.et()) );
  }
}

void  TopObjectResolutionCalc::operator()(TopMET& obj){
  if(useNN) {
    double v[2];
    v[0]=obj.et();
    v[1]=obj.eta();
    obj.setResA(     network[0]->Evaluate(0,v ));
    obj.setResB(     network[1]->Evaluate(0,v ));
    obj.setResC(     network[2]->Evaluate(0,v ));
    obj.setResD(     network[3]->Evaluate(0,v ));
    obj.setResTheta( 1000000.  );   			// Total freedom	
    obj.setResPhi(   network[5]->Evaluate(0,v ));	
    obj.setResET(    network[6]->Evaluate(0,v ));	
    obj.setResEta(   1000000.  );    			// Total freedom
  } else {
    obj.setResA(     this->getObsRes(0,0,obj.et())  );
    obj.setResC(     this->getObsRes(1,0,obj.et())  );
    obj.setResB(     this->getObsRes(2,0,obj.et())  );
    obj.setResD(     this->getObsRes(3,0,obj.et())  );
    obj.setResTheta( 1000000.  );   			// Total freedom
    obj.setResPhi(   this->getObsRes(5,0,obj.et())  );
    obj.setResET(    this->getObsRes(6,0,obj.et())  );
    obj.setResEta(   1000000.  );    			// Total freedom
  }
}

void  TopObjectResolutionCalc::operator()(TopTau& obj){
  if(useNN) {
    double v[2];
    v[0]=obj.et();
    v[1]=obj.eta();
    obj.setResA(     network[0]->Evaluate(0,v ));
    obj.setResB(     network[1]->Evaluate(0,v ));
    obj.setResC(     network[2]->Evaluate(0,v ));
    obj.setResD(     network[3]->Evaluate(0,v ));	
    obj.setResTheta( network[4]->Evaluate(0,v ));	 
    obj.setResPhi(   network[5]->Evaluate(0,v ));	
    obj.setResET(    network[6]->Evaluate(0,v ));	
    obj.setResEta(   network[7]->Evaluate(0,v ));
  } else {
    int bin = this->getEtaBin(obj.eta());
    obj.setResA(     this->getObsRes(0,bin,obj.et()) );
    obj.setResB(     this->getObsRes(1,bin,obj.et()) );
    obj.setResC(     this->getObsRes(2,bin,obj.et()) );	
    obj.setResD(     this->getObsRes(3,bin,obj.et()) );	
    obj.setResTheta( this->getObsRes(4,bin,obj.et()) );	 
    obj.setResPhi(   this->getObsRes(5,bin,obj.et()) );	
    obj.setResET(    this->getObsRes(6,bin,obj.et()) );	
    obj.setResEta(   this->getObsRes(7,bin,obj.et()) );
  }
}

