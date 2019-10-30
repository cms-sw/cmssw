#include <TFile.h>
#include <TH1.h>
#include <TH2.h>
#include <TPaveStats.h>
#include <TStyle.h>

#include <cassert>
#include <cstdlib>
#include <sstream>

#include "Validation/RecoParticleFlow/interface/Comparator.h"
#include "Validation/RecoParticleFlow/interface/NicePlot.h"
#include "Validation/RecoParticleFlow/interface/TH2Analyzer.h"

using namespace std;

void Comparator::SetDirs(const char *file0, const char *dir0, const char *file1, const char *dir1) {
  file0_ = new TFile(file0);
  if (file0_->IsZombie())
    exit(1);
  dir0_ = file0_->GetDirectory(dir0);
  if (!dir0_)
    exit(1);

  file1_ = new TFile(file1);
  if (file1_->IsZombie())
    exit(1);
  dir1_ = file1_->GetDirectory(dir1);
  if (!dir1_)
    exit(1);
}

void Comparator::SetStyles(Style *s0, Style *s1, const char *leg0, const char *leg1) {
  s0_ = s0;
  s1_ = s1;

  legend_.Clear();
  legend_.AddEntry(s0_, leg0, "mlf");
  legend_.AddEntry(s1_, leg1, "mlf");
}

void Comparator::DrawSlice(const char *key, int binxmin, int binxmax, Mode mode) {
  static int num = 0;

  ostringstream out0;
  out0 << "h0_2d_" << num;
  ostringstream out1;
  out1 << "h1_2d_" << num;
  num++;

  string name0 = out0.str();
  string name1 = out1.str();

  TH1 *h0 = Histo(key, 0);
  TH1 *h1 = Histo(key, 1);

  TH2 *h0_2d = dynamic_cast<TH2 *>(h0);
  TH2 *h1_2d = dynamic_cast<TH2 *>(h1);

  if (h0_2d->GetNbinsY() == 1 || h1_2d->GetNbinsY() == 1) {
    cerr << key << " is not 2D" << endl;
    return;
  }

  TH1::AddDirectory(false);

  TH1D *h0_slice = h0_2d->ProjectionY(name0.c_str(), binxmin, binxmax, "");
  TH1D *h1_slice = h1_2d->ProjectionY(name1.c_str(), binxmin, binxmax, "");
  TH1::AddDirectory(true);
  Draw(h0_slice, h1_slice, mode);
}

void Comparator::DrawMeanSlice(const char *key, const int rebinFactor, Mode mode) {
  TDirectory *dir = dir1_;
  dir->cd();
  TH2D *h2 = (TH2D *)dir->Get(key);
  TH2Analyzer TH2Ana(h2, rebinFactor);
  TH1D *ha = TH2Ana.Average();

  dir = dir0_;
  dir->cd();
  TH2D *h2b = (TH2D *)dir->Get(key);
  TH2Analyzer TH2Anab(h2b, rebinFactor);
  TH1D *hb = TH2Anab.Average();

  Draw(hb, ha, mode);
}

void Comparator::DrawSigmaSlice(const char *key, const int rebinFactor, Mode mode) {
  TDirectory *dir = dir1_;
  dir->cd();
  TH2D *h2 = (TH2D *)dir->Get(key);
  TH2Analyzer TH2Ana(h2, rebinFactor);
  TH1D *ha = TH2Ana.RMS();

  dir = dir0_;
  dir->cd();
  TH2D *h2b = (TH2D *)dir->Get(key);
  TH2Analyzer TH2Anab(h2b, rebinFactor);
  TH1D *hb = TH2Anab.RMS();

  Draw(hb, ha, mode);
}

void Comparator::DrawGaussSigmaSlice(const char *key, const int rebinFactor, Mode mode) {
  TDirectory *dir = dir1_;
  dir->cd();
  TH2D *h2 = (TH2D *)dir->Get(key);
  TH2Analyzer TH2Ana(h2, rebinFactor);
  TH1D *ha = TH2Ana.SigmaGauss();

  dir = dir0_;
  dir->cd();
  TH2D *h2b = (TH2D *)dir->Get(key);
  TH2Analyzer TH2Anab(h2b, rebinFactor);
  TH1D *hb = TH2Anab.SigmaGauss();

  Draw(hb, ha, mode);
}

Double_t fitFunction_f(Double_t *x, Double_t *par) {
  const Double_t value =
      sqrt(par[0] * par[0] + par[1] * par[1] * (x[0] - par[3]) + par[2] * par[2] * (x[0] - par[3]) * (x[0] - par[3])) /
      x[0];
  return value;
}

void Comparator::DrawGaussSigmaSlice(
    const char *key, const int rebinFactor, const int binxmin, const int binxmax, const bool cst_binning, Mode mode) {
  TDirectory *dir = dir1_;
  dir->cd();
  TH2D *h2 = (TH2D *)dir->Get(key);
  TH2Analyzer TH2Ana(h2, binxmin, binxmax, rebinFactor, cst_binning);
  TH1D *hrms = TH2Ana.RMS();
  TF1 *fitfcndgssrms3 = new TF1("fitfcndgssrms3",
                                fitFunction_f,
                                hrms->GetXaxis()->GetBinLowEdge(1),
                                hrms->GetXaxis()->GetBinUpEdge(hrms->GetNbinsX()),
                                4);
  fitfcndgssrms3->SetNpx(500);
  fitfcndgssrms3->SetLineWidth(3);
  fitfcndgssrms3->SetLineStyle(2);
  fitfcndgssrms3->SetLineColor(4);
  hrms->Fit(fitfcndgssrms3, "0R");

  TH1D *ha = TH2Ana.SigmaGauss();
  TF1 *fitfcndgsse3 = new TF1("fitfcndgsse3",
                              fitFunction_f,
                              hrms->GetXaxis()->GetBinLowEdge(1),
                              hrms->GetXaxis()->GetBinUpEdge(hrms->GetNbinsX()),
                              4);
  fitfcndgsse3->SetNpx(500);
  fitfcndgsse3->SetLineWidth(3);
  fitfcndgsse3->SetLineStyle(1);
  fitfcndgsse3->SetLineColor(4);
  ha->Fit(fitfcndgsse3, "0R");

  dir = dir0_;
  dir->cd();
  TH2D *h2b = (TH2D *)dir->Get(key);
  TH2Analyzer TH2Anab(h2b, binxmin, binxmax, rebinFactor, cst_binning);
  TH1D *hrmsb = TH2Anab.RMS();
  TF1 *fitfcndgssrmsb3 = new TF1("fitfcndgssrmsb3",
                                 fitFunction_f,
                                 hrms->GetXaxis()->GetBinLowEdge(1),
                                 hrms->GetXaxis()->GetBinUpEdge(hrms->GetNbinsX()),
                                 4);
  fitfcndgssrmsb3->SetNpx(500);
  fitfcndgssrmsb3->SetLineWidth(3);
  fitfcndgssrmsb3->SetLineStyle(2);
  fitfcndgssrmsb3->SetLineColor(2);
  hrmsb->Fit(fitfcndgssrmsb3, "0R");

  TH1D *hb = TH2Anab.SigmaGauss();
  TF1 *fitfcndgsseb3 = new TF1("fitfcndgsseb3",
                               fitFunction_f,
                               hrms->GetXaxis()->GetBinLowEdge(1),
                               hrms->GetXaxis()->GetBinUpEdge(hrms->GetNbinsX()),
                               4);
  fitfcndgsseb3->SetNpx(500);
  fitfcndgsseb3->SetLineWidth(3);
  fitfcndgsseb3->SetLineStyle(1);
  fitfcndgsseb3->SetLineColor(2);
  hb->Fit(fitfcndgsseb3, "0R");

  Draw(hb, ha, mode);
  // Draw(hrms,ha,mode);
  // Draw(ha,ha,mode);
  fitfcndgssrms3->Draw("same");
  fitfcndgsse3->Draw("same");
  fitfcndgssrmsb3->Draw("same");
  fitfcndgsseb3->Draw("same");
}

void Comparator::DrawGaussSigmaOverMeanXSlice(
    const char *key, const int rebinFactor, const int binxmin, const int binxmax, const bool cst_binning, Mode mode) {
  TDirectory *dir = dir1_;
  dir->cd();
  TH2D *h2 = (TH2D *)dir->Get(key);
  TH2Analyzer TH2Ana(h2, binxmin, binxmax, rebinFactor, cst_binning);
  TH1D *hrms = TH2Ana.RMS();

  TH1D *meanXslice = TH2Ana.MeanX();
  // for( int i=1; i<=meanXslice->GetNbinsX(); ++i) {
  //  std::cout << "meanXslice->GetBinContent(" << i << ") = "
  //	      << meanXslice->GetBinContent(i) << std::endl;
  //  std::cout << "meanXslice->GetBinError(" << i << ") = "
  //	      << meanXslice->GetBinError(i) << std::endl;
  //}
  // Draw(meanXslice,meanXslice,mode);
  hrms->Divide(meanXslice);

  TF1 *fitXfcndgssrms3 = new TF1("fitXfcndgssrms3",
                                 fitFunction_f,
                                 hrms->GetXaxis()->GetBinLowEdge(1),
                                 hrms->GetXaxis()->GetBinUpEdge(hrms->GetNbinsX()),
                                 4);
  fitXfcndgssrms3->SetNpx(500);
  fitXfcndgssrms3->SetLineWidth(3);
  fitXfcndgssrms3->SetLineStyle(2);
  fitXfcndgssrms3->SetLineColor(4);
  hrms->Fit(fitXfcndgssrms3, "0R");

  TH1D *ha = TH2Ana.SigmaGauss();
  ha->Divide(meanXslice);
  TF1 *fitXfcndgsse3 = new TF1("fitXfcndgsse3",
                               fitFunction_f,
                               ha->GetXaxis()->GetBinLowEdge(1),
                               ha->GetXaxis()->GetBinUpEdge(ha->GetNbinsX()),
                               4);
  fitXfcndgsse3->SetNpx(500);
  fitXfcndgsse3->SetLineWidth(3);
  fitXfcndgsse3->SetLineStyle(1);
  fitXfcndgsse3->SetLineColor(4);
  ha->Fit(fitXfcndgsse3, "0R");

  dir = dir0_;
  dir->cd();
  TH2D *h2b = (TH2D *)dir->Get(key);
  TH2Analyzer TH2Anab(h2b, binxmin, binxmax, rebinFactor, cst_binning);
  TH1D *hrmsb = TH2Anab.RMS();
  hrmsb->Divide(meanXslice);
  TF1 *fitXfcndgssrmsb3 = new TF1("fitXfcndgssrmsb3",
                                  fitFunction_f,
                                  hrmsb->GetXaxis()->GetBinLowEdge(1),
                                  hrmsb->GetXaxis()->GetBinUpEdge(hrmsb->GetNbinsX()),
                                  4);
  fitXfcndgssrmsb3->SetNpx(500);
  fitXfcndgssrmsb3->SetLineWidth(3);
  fitXfcndgssrmsb3->SetLineStyle(2);
  fitXfcndgssrmsb3->SetLineColor(2);
  hrmsb->Fit(fitXfcndgssrmsb3, "0R");

  TH1D *hb = TH2Anab.SigmaGauss();
  hb->Divide(meanXslice);
  TF1 *fitXfcndgsseb3 = new TF1("fitXfcndgsseb3",
                                fitFunction_f,
                                hb->GetXaxis()->GetBinLowEdge(1),
                                hb->GetXaxis()->GetBinUpEdge(hb->GetNbinsX()),
                                4);
  fitXfcndgsseb3->SetNpx(500);
  fitXfcndgsseb3->SetLineWidth(3);
  fitXfcndgsseb3->SetLineStyle(1);
  fitXfcndgsseb3->SetLineColor(2);
  hb->Fit(fitXfcndgsseb3, "0R");

  Draw(hb, ha, mode);
  // Draw(hrms,ha,mode);
  // Draw(ha,ha,mode);
  fitXfcndgssrms3->Draw("same");
  fitXfcndgsse3->Draw("same");
  fitXfcndgssrmsb3->Draw("same");
  fitXfcndgsseb3->Draw("same");
}

void Comparator::DrawGaussSigmaOverMeanSlice(const char *key, const char *key2, const int rebinFactor, Mode mode) {
  TDirectory *dir = dir1_;
  dir->cd();

  TH2D *h2_b = (TH2D *)dir->Get(key2);
  TH2Analyzer TH2Ana_b(h2_b, rebinFactor);
  TH1D *meanslice = TH2Ana_b.Average();

  TH2D *h2 = (TH2D *)dir->Get(key);
  TH2Analyzer TH2Ana(h2, rebinFactor);

  TH1D *ha = TH2Ana.SigmaGauss();
  ha->Divide(meanslice);

  dir = dir0_;
  dir->cd();
  TH2D *h2b = (TH2D *)dir->Get(key);
  TH2Analyzer TH2Anab(h2b, rebinFactor);

  TH2D *h2b_b = (TH2D *)dir->Get(key2);
  TH2Analyzer TH2Anab_b(h2b_b, rebinFactor);
  TH1D *meansliceb = TH2Anab_b.Average();

  TH1D *hb = TH2Anab.SigmaGauss();
  hb->Divide(meansliceb);

  Draw(hb, ha, mode);
  // Draw(meansliceb,meanslice,mode);
}

void Comparator::Draw(const char *key, Mode mode) {
  TH1::AddDirectory(false);
  TH1 *h0 = Histo(key, 0);
  TH1 *h1 = (TH1 *)Histo(key, 1)->Clone("h1");

  TH1::AddDirectory(true);
  Draw(h0, h1, mode);
}

void Comparator::Draw(const char *key0, const char *key1, Mode mode) {
  TH1 *h0 = nullptr;
  TH1 *h1 = nullptr;
  if (mode != EFF) {
    h0 = Histo(key0, 0);
    h1 = Histo(key1, 1);
  } else {
    h0 = Histo(key0, 0);
    TH1 *h0b = Histo(key1, 0);
    h1 = Histo(key0, 1);
    TH1 *h1b = Histo(key1, 1);
    if (rebin_ > 1) {
      h0->Rebin(rebin_);
      h1->Rebin(rebin_);
      h0b->Rebin(rebin_);
      h1b->Rebin(rebin_);
    }
    if (resetAxis_) {
      h0->GetXaxis()->SetRangeUser(xMin_, xMax_);
      h1->GetXaxis()->SetRangeUser(xMin_, xMax_);
      h0b->GetXaxis()->SetRangeUser(xMin_, xMax_);
      h1b->GetXaxis()->SetRangeUser(xMin_, xMax_);
    }

    h0b->Sumw2();
    h0->Sumw2();
    h0->Divide(h0, h0b, 1., 1., "B");
    h1b->Sumw2();
    h1->Sumw2();
    h1->Divide(h1, h1b, 1., 1., "B");
  }
  Draw(h0, h1, mode);
}

TH1 *Comparator::Histo(const char *key, unsigned dirIndex) {
  if (dirIndex > 1U) {  // dirIndex >= 0, since dirIndex is unsigned
    cerr << "bad dir index: " << dirIndex << endl;
    return nullptr;
  }
  TDirectory *dir = nullptr;
  if (dirIndex == 0)
    dir = dir0_;
  if (dirIndex == 1)
    dir = dir1_;
  assert(dir);

  dir->cd();

  TH1 *h = (TH1 *)dir->Get(key);
  if (!h)
    cerr << "no key " << key << " in directory " << dir->GetName() << endl;
  return h;
}

void Comparator::Draw(TH1 *h0, TH1 *h1, Mode mode) {
  if (!(h0 && h1)) {
    cerr << "invalid histo" << endl;
    return;
  }

  TH1::AddDirectory(false);
  h0_ = (TH1 *)h0->Clone("h0_");
  h1_ = (TH1 *)h1->Clone("h1_");
  TH1::AddDirectory(true);

  // unsetting the title, since the title of projections
  // is still the title of the 2d histo
  // and this is better anyway
  h0_->SetTitle("");
  h1_->SetTitle("");

  // h0_->SetStats(1);
  // h1_->SetStats(1);

  if (mode != EFF) {
    if (rebin_ > 1) {
      h0_->Rebin(rebin_);
      h1_->Rebin(rebin_);
    }
    if (resetAxis_) {
      h0_->GetXaxis()->SetRangeUser(xMin_, xMax_);
      h1_->GetXaxis()->SetRangeUser(xMin_, xMax_);
    }
  }

  if (mode != GRAPH) {
    TPaveStats *ptstats = new TPaveStats(0.7385057, 0.720339, 0.9396552, 0.8792373, "brNDC");
    ptstats->SetName("stats");
    ptstats->SetBorderSize(1);
    ptstats->SetLineColor(2);
    ptstats->SetFillColor(10);
    ptstats->SetTextAlign(12);
    ptstats->SetTextColor(2);
    ptstats->SetOptStat(1111);
    ptstats->SetOptFit(0);
    ptstats->Draw();
    h0_->GetListOfFunctions()->Add(ptstats);
    ptstats->SetParent(h0_->GetListOfFunctions());

    // std::cout << "FL: h0_->GetMean() = " << h0_->GetMean() << std::endl;
    // std::cout << "FL: h0_->GetRMS() = " << h0_->GetRMS() << std::endl;
    // std::cout << "FL: h1_->GetMean() = " << h1_->GetMean() << std::endl;
    // std::cout << "FL: h1_->GetRMS() = " << h1_->GetRMS() << std::endl;
    // std::cout << "FL: test2" << std::endl;
    TPaveStats *ptstats2 = new TPaveStats(0.7399425, 0.529661, 0.941092, 0.6885593, "brNDC");
    ptstats2->SetName("stats");
    ptstats2->SetBorderSize(1);
    ptstats2->SetLineColor(4);
    ptstats2->SetFillColor(10);
    ptstats2->SetTextAlign(12);
    ptstats2->SetTextColor(4);
    TText *text = ptstats2->AddText("h1_");
    text->SetTextSize(0.03654661);

    std::ostringstream oss3;
    oss3 << h1_->GetEntries();
    const std::string txt_entries = "Entries = " + oss3.str();
    text = ptstats2->AddText(txt_entries.c_str());
    std::ostringstream oss;
    oss << h1_->GetMean();
    const std::string txt_mean = "Mean  = " + oss.str();
    text = ptstats2->AddText(txt_mean.c_str());
    std::ostringstream oss2;
    oss2 << h1_->GetRMS();
    const std::string txt_rms = "RMS  = " + oss2.str();
    text = ptstats2->AddText(txt_rms.c_str());
    ptstats2->SetOptStat(1111);
    ptstats2->SetOptFit(0);
    ptstats2->Draw();
    h1_->GetListOfFunctions()->Add(ptstats2);
    ptstats2->SetParent(h1_->GetListOfFunctions());
  } else {
    TPaveStats *ptstats = new TPaveStats(0.0, 0.0, 0.0, 0.0, "brNDC");
    ptstats->Draw();
    h0_->GetListOfFunctions()->Add(ptstats);
    ptstats->SetParent(h0_->GetListOfFunctions());
  }

  float min = -999.;
  float max = +999.;
  switch (mode) {
    case SCALE:
      h1_->Scale(h0_->GetEntries() / h1_->GetEntries());
      break;
    case NORMAL:
      if (s0_)
        Styles::FormatHisto(h0_, s0_);
      if (s1_)
        Styles::FormatHisto(h1_, s1_);

      if (h1_->GetMaximum() > h0_->GetMaximum()) {
        h0_->SetMaximum(h1_->GetMaximum() * 1.15);
      }

      h0_->Draw();
      h1_->Draw("same");

      break;
    case EFF:
      if (s0_)
        Styles::FormatHisto(h0_, s0_);
      if (s1_)
        Styles::FormatHisto(h1_, s1_);

      // rather arbitrary but useful
      max = h0_->GetMaximum();
      if (h1_->GetMaximum() > max)
        max = h1_->GetMaximum();
      if (max > 0.8)
        max = 1;

      max *= 1.1;

      min = h0_->GetMinimum();
      if (h1_->GetMinimum() < min)
        min = h1_->GetMinimum();
      if (min > 0.2)
        min = 0.;

      min *= 0.8;
      h0_->SetMaximum(max);
      h0_->SetMinimum(min);

      h0_->Draw("E");
      h1_->Draw("Esame");
      break;
    case GRAPH:
      if (s0_)
        Styles::FormatHisto(h0_, s0_);
      if (s1_)
        Styles::FormatHisto(h1_, s1_);

      if (h1_->GetMaximum() > h0_->GetMaximum()) {
        h0_->SetMaximum(h1_->GetMaximum() * 1.15);
      }
      if (h1_->GetMinimum() < h0_->GetMinimum()) {
        h0_->SetMinimum(h1_->GetMinimum() * 1.15);
      }
      h0_->SetMarkerStyle(21);
      h0_->SetMarkerColor(2);
      h1_->SetMarkerStyle(21);
      h1_->SetMarkerColor(4);

      h0_->Draw("E1");
      h1_->Draw("E1same");
      break;
    case RATIO:
      h0_->Sumw2();
      h1_->Sumw2();
      h0_->Divide(h1_);
      if (s0_)
        Styles::FormatHisto(h0_, s0_);
      h0_->Draw();
      break;
    default:
      break;
  }
}
