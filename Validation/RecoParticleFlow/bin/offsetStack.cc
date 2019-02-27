#include "TString.h"
#include "TFile.h"
#include "TProfile.h"
#include "TH1.h"
#include "THStack.h"
#include "TCanvas.h"
#include "TStyle.h"
#include "TLegend.h"
#include "TLatex.h"

#include <iostream>
#include <map>

void getHists( std::map<TString, TH1D*>& hists, TFile*& file, TString var, int var_val, float r );
void setStyle();
void setStack( THStack*& stack, std::map<TString, TH1D*>& hists );

int main(int argc, char* argv[]) {

  if (argc < 6) {
    std::cout << "Please provide a file, variable, deltaR value, label, and an out directory." << std::endl;
    std::cout << "You may also include a second file at the end." << std::endl;
    std::cout << "Example: offsetStack file1.root npv 0.4 dataset_label plots" << std::endl;
    std::cout << "Example: offsetStack file1.root npv 0.4 dataset_label plots file2.root" << std::endl;
    return -1;
  }

  TString fname1 = argv[1];
  TString var = argv[2];
  float r = std::stof( argv[3] );
  TString label = argv[4];
  TString outdir = argv[5];

  TFile* file1 = TFile::Open( fname1 );
  if (!file1) {std::cout << "Invalid file1: " << fname1 << std::endl; return -1;}

  TH1F* h = (TH1F*) file1->FindObjectAny( var );
  int avg = h->GetMean()+0.5;

  std::map<TString, TH1D*> hists1 = { {"chm",0}, {"chu",0}, {"nh",0}, {"ne",0}, {"hfh",0}, {"hfe",0}, {"lep",0} };
  getHists( hists1, file1, var, avg, r );

  setStyle();
  TCanvas c("c", "c", 600, 600);

  THStack* stack1 = new THStack( "stack1", Form(";#eta;<Offset Energy_{T}(#pi%.1f^{2})>/<%s> [GeV]", r, var=="npv"?"N_{PV}":"#mu") );
  setStack( stack1, hists1 );
  stack1->Draw("hist");
  TString legPF = "F";

  TLegend* leg = new TLegend(.4,.67,.65,.92);

  //file2 included//
  if (argc > 6) {
    TString fname2 = argv[6];
    TFile* file2 = TFile::Open( fname2 );
    if (!file2) {std::cout << "Invalid file2: " << fname2 << std::endl; return -1;}

    std::map<TString, TH1D*> hists2 = hists1;
    getHists( hists2, file2, var, avg, r );

    THStack* stack2 = new THStack( "stack2", "stack2" );
    setStack( stack2, hists2 );
    stack2->Draw("samepe");

    legPF = "PF";
    leg->SetHeader("#bf{Markers: file2, Histograms: file1}");

    //Draw Markers for EM Deposits and Hadronic Deposits in two separate regions//
    TH1D* hfe_clone = (TH1D*) hists2["hfe"]->Clone("hfe_clone");
    TH1D* hfh_clone = (TH1D*) hists2["hfh"]->Clone("hfh_clone");

    THStack* cloneStack = new THStack( "cloneStack", "cloneStack" );
    cloneStack->Add(hists2["ne"]);
    cloneStack->Add(hfe_clone);
    cloneStack->Add(hists2["nh"]);
    cloneStack->Add(hfh_clone);
    cloneStack->Draw("samepe");

    hists2["ne"] ->SetAxisRange(-2.9,2.9);
    hists2["hfe"]->SetAxisRange(-5,-2.6);
    hists2["nh"] ->SetAxisRange(-2.9,2.9);
    hists2["hfh"]->SetAxisRange(-5,-2.6);
    hists2["chu"]->SetAxisRange(-2.9,2.9);
    hists2["chm"]->SetAxisRange(-2.9,2.9);

    hfe_clone        ->SetAxisRange(2.6,5);
    hfh_clone        ->SetAxisRange(2.6,5);
  }

  leg->SetBorderSize(0);
  leg->SetFillColor(0);
  leg->SetFillStyle(0);
  leg->SetTextSize(0.04);
  leg->SetTextFont(42);

  leg->AddEntry( hists1["ne"],  "Photons",                  legPF );
  leg->AddEntry( hists1["hfe"], "EM Deposits",              legPF );
  leg->AddEntry( hists1["nh"],  "Neutral Hadrons",          legPF );
  leg->AddEntry( hists1["hfh"], "Hadronic Deposits",        legPF );
  leg->AddEntry( hists1["chu"], "Unassoc. Charged Hadrons", legPF );
  leg->AddEntry( hists1["chm"], "Assoc. Charged Hadrons",   legPF );

  leg->Draw();

  TLatex text;
  text.SetNDC();

  text.SetTextSize(0.065);
  text.SetTextFont(61);
  text.DrawLatex(0.2, 0.87, "CMS");

  text.SetTextSize(0.045);
  text.SetTextFont(42);
  label.ReplaceAll( '_', ' ' );
  text.DrawLatex(0.7, 0.96, label);

  label.ReplaceAll( ' ', '_' );
  c.Print( outdir + "/stack_" + label + ".pdf" );
}

void getHists( std::map<TString, TH1D*>& hists, TFile*& file, TString var, int var_val, float r ) {

  for ( auto& pair : hists ) {

    TProfile* p = (TProfile*) file->FindObjectAny( Form("p_offset_eta_%s%i_%s", var.Data(), var_val, pair.first.Data()) );
    pair.second = p->ProjectionX( pair.first );
    pair.second->Scale( r*r / 2 / var_val );

    const double* xbins = p->GetXaxis()->GetXbins()->GetArray();
    for (int i=1, n=p->GetNbinsX(); i<=n; i++) {
      pair.second->SetBinContent( i, pair.second->GetBinContent(i) / (xbins[i]-xbins[i-1]) );
      pair.second->SetBinError( i, pair.second->GetBinError(i) / (xbins[i]-xbins[i-1]) );
    }
  }
}

void setStyle() {
  gStyle->SetPadTopMargin(0.05);
  gStyle->SetPadBottomMargin(0.1);
  gStyle->SetPadLeftMargin(0.16);
  gStyle->SetPadRightMargin(0.02);

  gStyle->SetOptStat(0);
  gStyle->SetOptTitle(0);

  gStyle->SetTitleFont(42, "XYZ");
  gStyle->SetTitleSize(0.05, "XYZ");
  gStyle->SetTitleXOffset(0.9);
  gStyle->SetTitleYOffset(1.4);

  gStyle->SetLabelFont(42, "XYZ");
  gStyle->SetLabelOffset(0.007, "XYZ");
  gStyle->SetLabelSize(0.04, "XYZ");

  gStyle->SetPadTickX(1);
  gStyle->SetPadTickY(1);
}

void setStack( THStack*& stack, std::map<TString, TH1D*>& hists ) {

  stack->Add( hists["ne"] );
  stack->Add( hists["hfe"] );
  stack->Add( hists["nh"] );
  stack->Add( hists["hfh"] );
  stack->Add( hists["chu"] );
  stack->Add( hists["chm"] );

  hists["ne"] ->SetMarkerStyle(kMultiply);
  hists["hfe"]->SetMarkerStyle(kOpenStar);
  hists["nh"] ->SetMarkerStyle(kOpenDiamond);
  hists["hfh"]->SetMarkerStyle(kOpenTriangleUp);
  hists["chu"]->SetMarkerStyle(kOpenCircle);
  hists["chm"]->SetMarkerStyle(kOpenCircle);

  hists["ne"] ->SetFillColor(kBlue);
  hists["hfe"]->SetFillColor(kViolet+2);
  hists["nh"] ->SetFillColor(kGreen);
  hists["hfh"]->SetFillColor(kPink+6);
  hists["chu"]->SetFillColor(kRed-9);
  hists["chm"]->SetFillColor(kRed);

  hists["ne"] ->SetLineColor(kBlack);
  hists["hfe"]->SetLineColor(kBlack);
  hists["nh"] ->SetLineColor(kBlack);
  hists["hfh"]->SetLineColor(kBlack);
  hists["chu"]->SetLineColor(kBlack);
  hists["chm"]->SetLineColor(kBlack);
}
