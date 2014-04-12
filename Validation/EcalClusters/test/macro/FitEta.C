#include "CrystalBall.C"
#include "effSigma.C"

#include "TChain.h"

TChain* chain = new TChain("energyScale");
void load_noET() {
  chain->Add("../SuperClusters.root");
  //Need to be added more files if there are multiple output root files
}

double fit0, fit1, fit2, fit3, fit4;

void plotEnergy(TString var, TString Eta, int etaBin, bool newE) {
  
  float mean_value = 0;
  float mean_error = 0;
  float sigma_value = 0;
  float ChiSquare = 0;
  
  int nBin = 1000;
  float xMin = 0.6;
  float xMax = 1.1;
  
  gStyle->SetOptFit(1);
  gStyle->SetStatX(0.48);
  gStyle->SetStatY(0.82);
  gStyle->SetStatW(0.2);
  gStyle->SetStatH(0.15);
 
  TString Cut = Eta + "&&em_scType==1";
  Cut = Cut + "&&em_et>10&&em_et<300&&!em_isInCrack";
  TH1F *h = new TH1F("h"," Et^{RECO}/Et^{GEN} GeV",nBin, xMin, xMax);
  chain->Draw(var + "/mc_et>>h",Cut);

  if ( newE ) {
    fit0 = 1800;
    fit1 = 0.9;
    fit2 = 5;
    fit3 = 1.000;
    fit4 = 0.006;
  }

  //TF1* g1 = new TF1(cb,"","",Emin, Emax);
  TF1* g1 = new TF1("g1",CrystalBall,0.85, 1.2, 5);
  g1->SetParameters(fit0, fit1, fit2, fit3, fit4);
  g1->SetLineColor(4);
  g1->SetLineWidth(3);
  float max = 1.02;
  if ( etaBin < 3 )
    max = 1.02;
  else if ( etaBin < 5 )
    max = 1.03;
  else
    max = 1.05;
  
  h->Fit(g1,"","",0.6,max);
  cout << endl;
  mean_value  = g1->GetParameter(3);
  mean_error  = g1->GetParError(3); 
  sigma_value = g1->GetParameter(4); 
  ChiSquare = g1->GetChisquare();

  fit0 = g1->GetParameter(0);
  fit1 = g1->GetParameter(1);
  fit2 = g1->GetParameter(2);
  fit3 = g1->GetParameter(3);
  fit4 = g1->GetParameter(4);
  
  cout << "Result:\t" << var << "\t" << Eta << "\t" << mean_value << "\t" << mean_error << "\t" << effSigma(h) << endl;
  
  delete g1;
  delete h;
} 

void derivationEnergyCorrection_phiWidth() {

  // MODULE 1: 0.018 < ETA < 0.423: 0.018, 0.119, 0.220, 0.320, 0.423
  // MODULE 2: 0.461 < ETA < 0.770: 0.461, 0.538, 0.616, 0.693, 0.770 
  // MODULE 3: 0.806 < ETA < 1.127: 0.806, 0.886, 0.967, 1.047, 1.127  
  // MODULE 4: 1.163 < ETA < 1.460: 1.163, 1.237, 1.312, 1.386, 1.460

  TString leftEta[24] = {"0.018", "0.0855", "0.153", "0.2205", "0.288", "0.3555", 
			 "0.461", "0.5125", "0.564", "0.6155", "0.667", "0.7185", 
			 "0.806", "0.8595", "0.913", "0.9665", "1.02",  "1.0735", 
			 "1.163", "1.2125", "1.262", "1.3115", "1.361", "1.4105"};

  TString rightEta[24] = {"0.0855", "0.153", "0.2205", "0.288", "0.3555", "0.423", 
			  "0.5125", "0.564", "0.6155", "0.667", "0.7185", "0.770", 
			  "0.8595", "0.913",  "0.9665", "1.02", "1.0735", "1.127", 
			  "1.2125", "1.262",  "1.3115","1.361","1.4105",  "1.46"};


  TString vars[2] = {"emCorr_et", "em_et"};

  for (int i=0; i < 2; ++i) {
    bool newE = true;
    for (int j=0; j < 24; ++j) {   
      TString etaCut = "abs(em_eta)>" + leftEta[j] + "&&abs(em_eta)<" + rightEta[j];
      plotEnergy(vars[i], etaCut, j, newE);
      newE = false;
    }
  } 
}

void FitEta(){
  load_noET();
  derivationEnergyCorrection_phiWidth();
  gROOT->ProcessLine(".q");
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
