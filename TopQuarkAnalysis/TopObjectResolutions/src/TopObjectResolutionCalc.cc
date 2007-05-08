//
// Author:  Jan Heyninck
// Created: Tue Apr  3 17:33:23 PDT 2007
//
// $Id: TopObjectResolutionCalc.cc,v 1.2 2007/05/04 00:58:04 heyninck Exp $
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
  TString def[2] = {"_abs","_rel"};
  for(Int_t ro=0; ro<6; ro++) {
    for(Int_t aor=0; aor<2; aor++) {
      for(Int_t par=0; par<3; par++) {
        TString obsName3 = objectType; obsName3 += resObsName[ro]; obsName3 += "_par"; obsName3 += par; obsName3 += def[aor];
        TH1F *tmp = (TH1F*) (resoFile->GetKey(obsName3)->ReadObj());
	if(tmp->GetEntries()>1){
	  fResPar[ro][aor][par] = (TF1)(*(tmp -> GetFunction("F_"+obsName3)));
        }
      }
      TString obsName = objectType; obsName += resObsName[ro]; obsName += "_etabin0"; obsName += def[aor]; 
      TH1F *tmp = (TH1F*) (resoFile->GetKey(obsName)->ReadObj());
      fResVsET[ro][aor] = (TF1)(*(tmp -> GetFunction("F_"+obsName)));
    }
  }
  cout << "=== done." << endl;

}


// destructor
TopObjectResolutionCalc::~TopObjectResolutionCalc() {
  delete resoFile;
}

double TopObjectResolutionCalc::getObsRes(int obs, double eT,double eta){
  double par1 = fResPar[obs][0][0].Eval(eta);
  double par2 = fResPar[obs][0][1].Eval(eta);
  double par3 = fResPar[obs][0][2].Eval(eta);
  fResVsET[obs][0].SetParameters(par1,par2,par3);
  double res = fResVsET[obs][0].Eval(eT);
  return res;
}

double TopObjectResolutionCalc::getObsRes(int obs, double eT){
  double res = fResVsET[obs][0].Eval(eT);
  return res;
}

void  TopObjectResolutionCalc::operator()(TopElectron& obj){
  //for the moment, only as a function of eT
  obj.setResPinv(  this->getObsRes(0,obj.et()) );	//,fabs(obj.eta()))  );
  obj.setResD(	   this->getObsRes(1,obj.et()) );	//,fabs(obj.eta()))  );
  obj.setResTheta( this->getObsRes(2,obj.et()) );	//,fabs(obj.eta()))  );   
  obj.setResPhi(   this->getObsRes(3,obj.et()) );	//,fabs(obj.eta()))  );
  obj.setResET(    this->getObsRes(4,obj.et()) );	//,fabs(obj.eta()))  );
  obj.setResEta(   this->getObsRes(5,obj.et()) );	//,fabs(obj.eta()))  ); 
}
void  TopObjectResolutionCalc::operator()(TopMuon& obj){
  //for the moment, only as a function of eT
  obj.setResPinv(  this->getObsRes(0,obj.et()) );	//,fabs(obj.eta()))  );
  obj.setResD(	   this->getObsRes(1,obj.et()) );	//,fabs(obj.eta()))  );
  obj.setResTheta( this->getObsRes(2,obj.et()) );	//,fabs(obj.eta()))  );   
  obj.setResPhi(   this->getObsRes(3,obj.et()) );	//,fabs(obj.eta()))  );
  obj.setResET(    this->getObsRes(4,obj.et()) );	//,fabs(obj.eta()))  );
  obj.setResEta(   this->getObsRes(5,obj.et()) );	//,fabs(obj.eta()))  ); 
}
void  TopObjectResolutionCalc::operator()(TopJet& obj){
  //for the moment, only as a function of eT
  obj.setResPinv(  this->getObsRes(0,obj.et()) );	//,fabs(obj.eta()))  );
  obj.setResD(	   this->getObsRes(1,obj.et()) );	//,fabs(obj.eta()))  );
  obj.setResTheta( this->getObsRes(2,obj.et()) );	//,fabs(obj.eta()))  );   
  obj.setResPhi(   this->getObsRes(3,obj.et()) );	//,fabs(obj.eta()))  );
  obj.setResET(    this->getObsRes(4,obj.et()) );	//,fabs(obj.eta()))  );
  obj.setResEta(   this->getObsRes(5,obj.et()) );	//,fabs(obj.eta()))  ); 
}
void  TopObjectResolutionCalc::operator()(TopMET& obj){
  obj.setResPinv(  this->getObsRes(0,obj.et())  );
  obj.setResD(	   this->getObsRes(1,obj.et())  );
  obj.setResTheta( 1000000.  );   			// Total freedom
  obj.setResPhi(   this->getObsRes(3,obj.et())  );
  obj.setResET(    this->getObsRes(4,obj.et())  );
  obj.setResEta(   1000000.  );    			// Total freedom
}
