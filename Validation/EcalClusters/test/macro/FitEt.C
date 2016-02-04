#include "CrystalBall.C"
#include "effSigma.C"

#include "TChain.h"

TChain* chain = new TChain("energyScale");
void load_noET() {
  chain->Add("../SuperClusters.root");
 //Need to be added more files if there are multiple output root files
}
double fit0, fit1, fit2, fit3, fit4;
double left, right;

void plotEnergy(TString var, TString Eta, int etaBin, bool newFit) {
  
  float mean_value = 0;
  float mean_error = 0;
  float sigma_value = 0;
  float ChiSquare = 0;
  
  int nBin = 600;
  float xMin = 0.8;
  float xMax = 1.1;
  
  gStyle->SetOptFit(1);
  gStyle->SetStatX(0.48);
  gStyle->SetStatY(0.82);
  gStyle->SetStatW(0.2);
  gStyle->SetStatH(0.15);
 
  TString Cut = Eta + "&&em_scType==1";
  Cut = Cut + "&&mc_et>10&&mc_et<300&&!em_isInCrack";
  TH1F *h = new TH1F("h"," Et^{RECO}/Et^{GEN} GeV",nBin, xMin, xMax);
  chain->Draw(var + "/mc_et>>h",Cut);

  if ( newFit ) {
    fit0 = 25;
    fit1 = 0.4;
    fit2 = 3;
    fit3 = 1;
    fit4 = 0.016;

    left = 0.7;
    right = 1.1;
  }
  // This is for BR1/Full ============
  if ( true ) {
    if ( etaBin >= 0 ) {
      right = 1.05;
    }
    if ( etaBin >= 5  ) {
      right = 1.03;
    }
    if ( etaBin >= 10 ) {
      right = 1.03;
    }
  }
  
  if ( false ) {
    if ( etaBin >= 0 ) {
      right = 1.06;
    }
    if ( etaBin >= 20  ) {
      right = 1.05;
    }
    if ( etaBin >= 50 ) {
      right = 1.05;
    }
  }
  TF1* g1 = new TF1("g1",CrystalBall,0.85, 1.2, 5);
  g1->SetParameters(fit0, fit1, fit2, fit3, fit4);
  g1->SetLineColor(4);
  g1->SetLineWidth(3);
  h->Fit(g1,"","",left,right);

  mean_value  = g1->GetParameter(3);
  mean_error  = g1->GetParError(3); 
  sigma_value = g1->GetParameter(4); 
  ChiSquare = g1->GetChisquare();
  
  cout << "Result:\t" << var << "\t" << Eta << "\t" << mean_value << "\t" << mean_error << "\t" << effSigma(h) << endl;

  fit0 = g1->GetParameter(0);
  fit1 = g1->GetParameter(1);
  fit2 = g1->GetParameter(2);
  fit3 = g1->GetParameter(3);
  fit4 = g1->GetParameter(4);
  
  delete g1;
  delete h;
} 

void derivationEnergyCorrection_phiWidth() {

  TString vars[2] = {"emCorr_et", "em_et"};

  for (int i=0; i < 2; ++i) {
    bool newFit = true;
    for (int j=0; j < 145; ++j) {
      TString leftEt = FloatToString(10 + 2*j);
      TString rightEt = FloatToString(10 + 2*(j + 1));

      TString etaCut = "mc_et>" + leftEt + "&&mc_et<" + rightEt;
      plotEnergy(vars[i], etaCut, j, newFit);
      newFit = false;
    }
  } 
}

TString IntToString(int number){
  ostringstream oss;
  oss << number;
  return oss.str();
}

TString FloatToString(float number){
  ostringstream oss;
  oss << number;
  return oss.str();
}


void FitEt(){
  load_noET();
  derivationEnergyCorrection_phiWidth();
  gROOT->ProcessLine(".q");
}

