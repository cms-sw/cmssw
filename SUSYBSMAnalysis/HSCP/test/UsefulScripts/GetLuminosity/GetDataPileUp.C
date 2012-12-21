#include <iostream>
#include "TFile.h"
#include "TH1.h"

//Outputs arrays to put in Analysis_Step3.C
void GetDataPileUp() {
  TFile* File  = new TFile("PileUp.root");
  TFile* File_XSecShiftUp  = new TFile("PileUp_XSecShiftUp.root");
  TFile* File_XSecShiftDown  = new TFile("PileUp_XSecShiftDown.root");

  TH1D* pileup = (TH1D*)File->Get("pileup");
  TH1D* pileup_XSecShiftDown = (TH1D*)File->Get("pileup");
  TH1D* pileup_XSecShiftUp = (TH1D*)File->Get("pileup");

  cout << "const   float TrueDist2012_f[60] = {";
  for (int i=0; i<pileup->GetNbinsX(); i++) {
    if(i!=(pileup->GetNbinsX()-1)) std::cout << pileup->GetBinContent(i+1)/pileup->Integral() << " ,";
    else cout << pileup->GetBinContent(i+1)/pileup->Integral() << "};" << endl;
  }

  cout << "const   float TrueDist2012_XSecShiftUp_f[60] = {";
  for (int i=0; i<pileup_XSecShiftUp->GetNbinsX(); i++) {
    if(i!=(pileup_XSecShiftUp->GetNbinsX()-1)) std::cout << pileup_XSecShiftUp->GetBinContent(i+1)/pileup_XSecShiftUp->Integral() << " ,";
    else cout << pileup_XSecShiftUp->GetBinContent(i+1)/pileup_XSecShiftUp->Integral() << "};" << endl;
  }

  cout << "const   float TrueDist2012_XSecShiftDown_f[60] = {";
  for (int i=0; i<pileup_XSecShiftDown->GetNbinsX(); i++) {
    if(i!=(pileup_XSecShiftDown->GetNbinsX()-1)) std::cout << pileup_XSecShiftDown->GetBinContent(i+1)/pileup_XSecShiftDown->Integral() << " ,";
    else cout << pileup_XSecShiftDown->GetBinContent(i+1)/pileup_XSecShiftDown->Integral() << "};" << endl;
  }
}
