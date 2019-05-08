#include "Validation/RecoParticleFlow/interface/NicePlot.h"

#include <TROOT.h>
#include <TStyle.h>
#include <string>

using namespace std;

Styles::Styles() {
  gROOT->SetStyle("Plain");
  gStyle->SetPalette(1);
  gStyle->SetHistMinimumZero(kTRUE);

  s1 = new Style();

  s1->SetLineWidth(2);
  s1->SetLineColor(1);

  s2 = new Style();

  s2->SetLineWidth(2);
  s2->SetLineColor(4);

  sg1 = new Style();

  sg1->SetMarkerColor(4);
  sg1->SetLineColor(4);
  sg1->SetLineWidth(2);
  sg1->SetMarkerStyle(21);

  sback = new Style();
  sback->SetFillStyle(1001);
  sback->SetFillColor(5);

  spred = new Style();
  spred->SetLineColor(2);
  spred->SetLineWidth(2);
  spred->SetFillStyle(1001);
  spred->SetFillColor(kRed - 8);

  spblue = new Style();
  spblue->SetLineColor(4);
  spblue->SetLineWidth(2);

  sgr1 = new Style();
  sgr1->SetLineWidth(1);
  sgr1->SetLineColor(1);

  sgr2 = new Style();
  sgr2->SetLineWidth(1);
  sgr2->SetLineColor(1);
}

void Styles::FormatHisto(TH1 *h, const Style *s) {
  //  h->SetStats(0);
  h->SetTitle("CMS Preliminary");

  h->GetYaxis()->SetTitleSize(0.06);
  h->GetYaxis()->SetTitleOffset(1.2);
  h->GetXaxis()->SetTitleSize(0.06);
  h->GetYaxis()->SetLabelSize(0.045);
  h->GetXaxis()->SetLabelSize(0.045);

  h->SetLineWidth(s->GetLineWidth());
  h->SetLineColor(s->GetLineColor());
  h->SetFillStyle(s->GetFillStyle());
  h->SetFillColor(s->GetFillColor());
}

void Styles::FormatPad(TPad *pad, bool grid, bool logx, bool logy) {
  pad->SetGridx(grid);
  pad->SetGridy(grid);

  if (logx)
    pad->SetLogx();
  if (logy)
    pad->SetLogy();

  pad->SetBottomMargin(0.14);
  pad->SetLeftMargin(0.15);
  pad->SetRightMargin(0.05);
  pad->Modified();
  pad->Update();
}

void Styles::SavePlot(const char *name, const char *dir) {
  string eps = dir;
  eps += "/";
  eps += name;
  eps += ".eps";
  gPad->SaveAs(eps.c_str());

  string png = dir;
  png += "/";
  png += name;
  png += ".png";
  gPad->SaveAs(png.c_str());
}
