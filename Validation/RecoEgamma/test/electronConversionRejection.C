#if !defined(__CINT__) || defined(__MAKECINT__)
#include <iostream>
#include "TFile.h"
#include "TDirectory.h"
#include "TH1F.h"
#include "TGraphAsymmErrors.h"
#include "TCanvas.h"
#endif

void electronConversionRejection() {

  TFile *fin = new TFile("ElectronConversionRejectionValidation.root");
  TDirectory *dir = (TDirectory*)fin->Get("DQMData");
  
  TCanvas *cpteff = new TCanvas;
  TH1F *hptnum = (TH1F*)dir->Get("elePtPass");
  TH1F *hptden = (TH1F*)dir->Get("elePtAll");
  TGraphAsymmErrors *hpteff = new TGraphAsymmErrors(hptnum,hptden);
  hpteff->SetTitle("Conversion Veto Efficiency");
  hpteff->GetYaxis()->SetTitle("Efficiency");
  hpteff->GetXaxis()->SetTitle("Gsf Electron p_{T} (GeV)");
  hpteff->SetMinimum(0.0);
  hpteff->SetMaximum(1.0);
  hpteff->Draw("APE");
  cpteff->SaveAs("cpteff.eps");
  
  TCanvas *cetaeff = new TCanvas;
  TH1F *hetanum = (TH1F*)dir->Get("eleEtaPass");
  TH1F *hetaden = (TH1F*)dir->Get("eleEtaAll");
  TGraphAsymmErrors *hetaeff = new TGraphAsymmErrors(hetanum,hetaden);
  hetaeff->SetTitle("Conversion Veto Efficiency");
  hetaeff->GetYaxis()->SetTitle("Efficiency");
  hetaeff->GetXaxis()->SetTitle("Gsf Electron #eta");
  hetaeff->SetMinimum(0.0);
  hetaeff->SetMaximum(1.0);
  hetaeff->Draw("APE");
  cetaeff->SaveAs("etaeff.eps");
  
  TCanvas *cphieff = new TCanvas;
  TH1F *hphinum = (TH1F*)dir->Get("elePhiPass");
  TH1F *hphiden = (TH1F*)dir->Get("elePhiAll");
  TGraphAsymmErrors *hphieff = new TGraphAsymmErrors(hphinum,hphiden);
  hphieff->SetTitle("Conversion Veto Efficiency");
  hphieff->GetYaxis()->SetTitle("Efficiency");
  hphieff->GetXaxis()->SetTitle("Gsf Electron #phi");
  hphieff->SetMinimum(0.0);
  hphieff->SetMaximum(1.0);
  hphieff->Draw("APE");  
  cphieff->SaveAs("cphieff.eps");
  
}
