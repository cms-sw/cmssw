#include "TFile.h"
#include "TTree.h"
#include "TText.h"

void SCEnergyPlots (TString inputfile = "SuperClusters.root" )
{
  gROOT->Reset();
  char* rfilename = inputfile;

  TFile *rfile = new TFile(rfilename);
  rfile->ls();
  TTree * mytree = (TTree*)rfile->Get("energyScale");

  cout << "Validation plots for SC Energy Corrections; " << endl;

  int nBins = 200;
  float xMin = 0;
  float xMax = 1.2;

  TCut hybSC = "em_scType==1";
  TCut dynSC = "em_scType==2";
  TCut fmSC  = "em_scType==3";
  
  TCut barrel = "fabs(em_eta)<1.479";
  TCut endcap = "fabs(em_eta)>1.55&&fabs(em_eta)<2.5";
  
  //========== For my ROOT, need to be REMOVED before commiting
  //  TCut barrel = "abs(em_eta)<1.479";
  //  TCut endcap = "abs(em_eta)>1.55&&abs(em_eta)<2.5";

  TString plotEnergy = "emCorr_e/mc_e";
  TString plotEt     = "emCorr_et/mc_et";

  
  //Barrel SuperClusters
  cout << endl;
  cout << "Hybrid Super Clusters in Barrel." << endl;
  
  TCanvas *Hybrid = new TCanvas("Hybrid", "Hybrid", 800, 800);
  Hybrid->Divide(1,2);

  TH1F *h1 = new TH1F("h1","hybridSCEnergy", nBins, xMin, xMax);
  TH1F *h2 = new TH1F("h2","hybridSCEt", nBins, xMin, xMax);

  Hybrid->cd(1);
  mytree->Draw( plotEnergy+">>h1",hybSC&&barrel);
  Hybrid->cd(2);
  mytree->Draw( plotEt+">>h2",hybSC&&barrel);
  Hybrid->Print("Hybrid.eps");
  delete h1; 
  delete h2;
  delete Hybrid;

  //---------------------------
  cout << endl;
  cout << "Dynamic Hybrid Super Clusters in Barrel. " << endl;
  TCanvas *Dynamic = new TCanvas("Dynamic","Dynamic",800,800);
  Dynamic->Divide(1,2);
  
  TH1F *h1 = new TH1F("h1","dynamicHybridSCEnergy",nBins, xMin, xMax);
  TH1F *h2 = new TH1F("h2","dynamicHybridSCEt",nBins, xMin, xMax);
  Dynamic->cd(1);
  mytree->Draw( plotEnergy+">>h1",dynSC&&barrel);
  Dynamic->cd(2);
  mytree->Draw( plotEt+">>h2",dynSC&&barrel);
  Dynamic->Print("Dynamic.eps");
  delete h1;
  delete h2;
  delete Dynamic;
  //-----------------------------
  cout << endl;
  cout << "Fixed Matrix Super Clusters in Endcap. " << endl;
  
  TCanvas *FixedMatrix = new TCanvas("FixedMatrix","FixedMatrix",800,800);
  FixedMatrix->Divide(1,2);
  
  TH1F *h1 = new TH1F("h1","fixedMatrixSCEnergy",nBins, xMin, xMax);
  TH1F *h2 = new TH1F("h2","fixedMatrixSCEt",nBins, xMin, xMax);
  FixedMatrix->cd(1);
  mytree->Draw( plotEnergy+">>h1",fmSC&&endcap);
  FixedMatrix->cd(2);
  mytree->Draw( plotEt+">>h2",fmSC&&endcap);
  FixedMatrix->Print("FixedMatrix.eps");
  delete h1;
  delete h2;
  delete FixedMatrix;
  
  gROOT->ProcessLine(".q");
}
