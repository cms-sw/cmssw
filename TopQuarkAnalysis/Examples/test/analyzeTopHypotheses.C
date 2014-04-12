#include <TCanvas.h>
#include <TFile.h>
#include <TH1F.h>
#include <TLegend.h>
#include <TROOT.h>

#include <vector>
#include <iostream>

void setHistStyles(std::vector<TH1F*>* hists, const bool normalize)
{
  const int lineColors[3] = {kGreen+1, kBlue, kRed};
  const int lineStyles[3] = {1, 3, 1};
  const int fillColors[3] = {kGreen+1, kBlue, 0};
  const int fillStyles[3] = {3554, 0, 0};

  for(unsigned h=0; h<hists[0].size(); h++) {
    hists[0][h]->SetXTitle(hists[0][h]->GetTitle());
    if(normalize)
      hists[0][h]->SetYTitle("a.u.");
    else
      hists[0][h]->SetYTitle("Events");
    hists[0][h]->SetTitle("");
    hists[0][h]->SetStats(kFALSE);
    for(unsigned d=0; d<3; d++) {
      if(normalize)
	hists[d][h]->Scale(1/hists[d][h]->Integral());
      hists[d][h]->SetLineWidth(2);
      hists[d][h]->SetLineColor(lineColors[d]);
      hists[d][h]->SetLineStyle(lineStyles[d]);
      hists[d][h]->SetFillColor(fillColors[d]);
      hists[d][h]->SetFillStyle(fillStyles[d]);
    }
  }
}

void setYmax()
{
  TIter iter(gPad->GetListOfPrimitives());
  TObject *obj;
  TH1 *h_1=0;
  TH1 *h_i=0;
  Bool_t foundfirstHisto = kFALSE;
  while ((obj = (TObject*)iter.Next())) {
    if(obj->InheritsFrom("TH1")) {
      if(foundfirstHisto == kFALSE) {
	h_1 = (TH1*)obj;
	foundfirstHisto = kTRUE;
      }
      else {
	h_i = (TH1*)obj;
	Double_t max_i = h_i->GetMaximum();
	if(max_i > h_1->GetMaximum()) h_1->SetMaximum(1.05 * max_i);
      }
    }
  }
}

void analyzeTopHypotheses()
{
  TFile* file = new TFile("analyzeTopHypothesis.root");

  const bool normalize = true;
  
  gROOT->cd();
  gROOT->SetStyle("Plain");

  const TString dirs[3] = {"analyzeGenMatch",
			   "analyzeMaxSumPtWMass",
			   "analyzeKinFit"};

  std::vector<TH1F*> hists[3];

  TIter iter(((TDirectoryFile*) file->Get(dirs[0]))->GetListOfKeys());
  TObject *obj;
  while((obj = (TObject*)iter.Next())) {
    if(((TDirectoryFile*) file->Get(dirs[0]))->Get(obj->GetName())->InheritsFrom("TH1F")) {
      for(unsigned d=0; d<3; d++)
	hists[d].push_back((TH1F*) file->Get(dirs[d]+"/"+obj->GetName())->Clone());
    }
  }
    
  file->Close();
  delete file;

  setHistStyles(hists, normalize);

  TCanvas* canvas = new TCanvas("canvas", "canvas", 900, 600);
  canvas->Print("analyzeTopHypotheses.ps[");

  TLegend legend(0.6, 0.75, 0.9, 0.9);
  legend.SetFillColor(0);
  legend.AddEntry(hists[0][0], "GenMatch"     , "F");
  legend.AddEntry(hists[1][0], "MaxSumPtWMass", "L");
  legend.AddEntry(hists[2][0], "KinFit"       , "L");

  for(unsigned h=0; h<hists[0].size(); h++) {
    hists[0][h]->Draw();
    if(!((TString)hists[0][h]->GetName()).Contains("genMatch")) {
      hists[1][h]->Draw("same");
      hists[2][h]->Draw("same");
      legend.Draw();
      setYmax();
    }
    gPad->RedrawAxis();
    canvas->Print("analyzeTopHypotheses.ps");
    //canvas->Print((TString)"analyzeTopHypotheses/"+hists[0][h]->GetName()+".eps");
  }

  canvas->Print("analyzeTopHypotheses.ps]");
  delete canvas;
}
