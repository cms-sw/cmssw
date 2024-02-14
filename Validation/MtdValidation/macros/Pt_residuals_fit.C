// -*- C -*-
//
// 'Resolution in track Pt' macro
//
// \author 12/2023 - Raffaele Delli Gatti

#include "Riostream.h"
#include "TFile.h"
#include "TDirectoryFile.h"
#include "TTree.h"
#include "TCanvas.h"
#include "TH1D.h"
#include "TStyle.h"
#include "TLegend.h"
#include "TLatex.h"
#include "TString.h"
#include "RooRealVar.h"
#include "RooCrystalBall.h"
#include "RooAddPdf.h"
#include "RooDataHist.h"
#include "RooPlot.h"
#include "RooFitResult.h"
using namespace std;
using namespace RooFit;

// Funtion for fitting to data with roofit library (defined below)
// ---------------------------------------------------------------
void fit_to_data(TH1D* histogram, TString name_file);

// Main function
//--------------
void residuals_fit() {
  // Open the root file to read
  // --------------------------

  TFile* file_DQM = TFile::Open("./DQM_V0001_R000000001__Global__CMSSW_X_Y_Z__RECO.root", "READ");
  // and take its directories
  TDirectoryFile* dir_DQMData = (TDirectoryFile*)file_DQM->Get("DQMData");
  if (!dir_DQMData)
    cout << "Cannot find dir_DQMData" << endl;
  TDirectoryFile* dir_Run1 = (TDirectoryFile*)dir_DQMData->Get("Run 1");
  if (!dir_Run1)
    cout << "Cannot find dir_Run1" << endl;
  TDirectoryFile* dir_MTD = (TDirectoryFile*)dir_Run1->Get("MTD");
  if (!dir_MTD)
    cout << "Cannot find dir_MTD" << endl;
  TDirectoryFile* dir_Runsum = (TDirectoryFile*)dir_MTD->Get("Run summary");
  if (!dir_Runsum)
    cout << "Cannot find dir_Runsum" << endl;
  TDirectoryFile* dir_Tracks = (TDirectoryFile*)dir_Runsum->Get("Tracks");
  if (!dir_Tracks)
    cout << "Cannot find dir_Tracks" << endl;

  // Take the trees with the method Get()
  TH1D* h_TrackMatchedTPBTLPtRatioGen = (TH1D*)dir_Tracks->Get("TrackMatchedTPBTLPtRatioGen");
  TH1D* h_TrackMatchedTPBTLPtRatioMtd = (TH1D*)dir_Tracks->Get("TrackMatchedTPBTLPtRatioMtd");
  TH1D* h_TrackMatchedTPBTLPtResMtd = (TH1D*)dir_Tracks->Get("TrackMatchedTPBTLPtResMtd");

  TH1D* h_TrackMatchedTPETLPtRatioGen = (TH1D*)dir_Tracks->Get("TrackMatchedTPETLPtRatioGen");
  TH1D* h_TrackMatchedTPETLPtRatioMtd = (TH1D*)dir_Tracks->Get("TrackMatchedTPETLPtRatioMtd");
  TH1D* h_TrackMatchedTPETLPtResMtd = (TH1D*)dir_Tracks->Get("TrackMatchedTPETLPtResMtd");

  TH1D* h_TrackMatchedTPETL2PtRatioGen = (TH1D*)dir_Tracks->Get("TrackMatchedTPETL2PtRatioGen");
  TH1D* h_TrackMatchedTPETL2PtRatioMtd = (TH1D*)dir_Tracks->Get("TrackMatchedTPETL2PtRatioMtd");
  TH1D* h_TrackMatchedTPETL2PtResMtd = (TH1D*)dir_Tracks->Get("TrackMatchedTPETL2PtResMtd");

  // Fit to data with the function fit_to_data
  //------------------------------------------
  fit_to_data(h_TrackMatchedTPBTLPtRatioGen, "BTLPtRatioGen_fit.pdf");
  fit_to_data(h_TrackMatchedTPETLPtRatioGen, "ETLPtRatioGen_fit.pdf");
  fit_to_data(h_TrackMatchedTPETL2PtRatioGen, "ETL2PtRatioGen_fit.pdf");

  fit_to_data(h_TrackMatchedTPBTLPtRatioMtd, "BTLPtRatioMtd_fit.pdf");
  fit_to_data(h_TrackMatchedTPETLPtRatioMtd, "ETLPtRatioMtd_fit.pdf");
  fit_to_data(h_TrackMatchedTPETL2PtRatioMtd, "ETL2PtRatioMtd_fit.pdf");

  fit_to_data(h_TrackMatchedTPBTLPtResMtd, "BTLPtResMtd_fit.pdf");
  fit_to_data(h_TrackMatchedTPETLPtResMtd, "ETLPtResMtd_fit.pdf");
  fit_to_data(h_TrackMatchedTPETL2PtResMtd, "ETL2PtResMtd_fit.pdf");

  return;
}

void fit_to_data(TH1D* h_TrackMatchedTP, TString str) {
  // Fit to data using roofit library
  // --------------------------------

  Double_t bin_width = h_TrackMatchedTP->GetBinWidth(0);
  Double_t range_min = h_TrackMatchedTP->GetXaxis()->GetBinLowEdge(1);
  Double_t range_max =
      h_TrackMatchedTP->GetXaxis()->GetBinLowEdge(h_TrackMatchedTP->GetNbinsX()) + h_TrackMatchedTP->GetBinWidth(0);

  // Observable
  RooRealVar x_res("x res", "", range_min, range_max);

  // Import data from histogram
  RooDataHist* h_ = new RooDataHist("h_", "h_", x_res, h_TrackMatchedTP);

  // Parameters
  RooRealVar mean("mean", "mean", h_TrackMatchedTP->GetMean(), range_min, range_max);
  RooRealVar sigmaL("sigmaL", "sigmaL", 0.05, 0., 5.);
  RooRealVar sigmaR("sigmaR", "sigmaR", 0.05, 0., 5.);
  RooRealVar alphaL("alphaL", "alphaL", 1., 0., 5.);
  RooRealVar alphaR("alphaR", "alphaR", 1., 0., 5.);
  RooRealVar nL("NL", "NL", 5., 0., 100.);
  RooRealVar nR("NR", "NR", 5., 0., 100.);

  // Build a double sided crystall ball PDF
  RooCrystalBall* pdf = new RooCrystalBall("pdf", "pdf", x_res, mean, sigmaL, sigmaR, alphaL, nL, alphaR, nR);
  // Construct a signal PDF
  RooRealVar nsig("nsig", "#signal events", h_TrackMatchedTP->GetEntries(), 0., h_TrackMatchedTP->GetEntries() * 2);
  RooAddPdf* model = new RooAddPdf("model", "model", {*pdf}, {nsig});

  // The PDF fit to that data set using an un-binned maximum likelihood fit
  // Then the data are visualized with the PDF overlaid

  // Perform extended ML fit of PDF to data and save results in a pointer
  RooFitResult* r1 = model->fitTo(*h_, Save());

  // Retrieve values from the fit
  Double_t mean_fit = mean.getVal();
  Double_t err_mean = mean.getError();
  Double_t sigmaR_fit = sigmaR.getVal();
  Double_t err_sigmaR = sigmaR.getError();
  Double_t sigmaL_fit = sigmaL.getVal();
  Double_t err_sigmaL = sigmaL.getError();

  // Compute resolution as half-width of the interval containing 68% of all entries (including overflows), centered around the MPV of the residuals
  // ----------------------------------------------------------------------------------------------------------------------------------------------

  Double_t res = 0;
  Double_t min = 0;
  double_t integral = 0;
  for (Int_t i = 1; i < h_TrackMatchedTP->GetNbinsX() / 2; i++) {
    Int_t bin_mean = (mean_fit - range_min) / bin_width;
    double_t int_norm = h_TrackMatchedTP->Integral(bin_mean - i, bin_mean + i) / h_TrackMatchedTP->Integral();
    if (int_norm - 0.68 < min)
      res = i * bin_width;
  }
  cout << "Resolution = " << res << " +- " << bin_width << endl;

  // Create a RooPlot to draw on
  //----------------------------
  // We don't manage the memory of the returned pointer
  // Instead we let it leak such that the plot still exists at the end of the macro and we can take a look at it
  RooPlot* xresframe = x_res.frame();

  // Plot data and PDF overlaid
  h_->plotOn(xresframe, MarkerSize(0.8), Name("histogram"));
  model->plotOn(xresframe, LineColor(kBlue), LineWidth(3), Name("model"));
  // In the previous lines, name is needed for the legend

  auto legend_res = new TLegend(0.65, 0.8, 0.8, 0.9);
  gStyle->SetLegendTextSize(0.033);
  TLatex* header = new TLatex();
  header->SetTextSize(0.035);

  TCanvas* c1 = new TCanvas("c1", "c1", 600, 500);
  c1->cd();
  xresframe->GetXaxis()->SetTitle(h_TrackMatchedTP->GetXaxis()->GetTitle());
  xresframe->Draw();
  if (str.Contains("BTL") && str.Contains("Mtd"))
    legend_res->AddEntry("histogram", "BTL", "PLerr");
  else if (str.Contains("BTL") && str.Contains("Gen"))
    legend_res->AddEntry("histogram", "GenTrack (barrel)", "PLerr");
  else if (str.Contains("ETLPt") && str.Contains("Mtd"))
    legend_res->AddEntry("histogram", "ETL (one hit)", "PLerr");
  else if (str.Contains("ETLPt") && str.Contains("Gen"))
    legend_res->AddEntry("histogram", "GenTrack (end-caps)", "PLerr");
  else if (str.Contains("ETL2") && str.Contains("Mtd"))
    legend_res->AddEntry("histogram", "ETL (2 hits)", "PLerr");
  else if (str.Contains("ETL2") && str.Contains("Gen"))
    legend_res->AddEntry("histogram", "GenTrack (end-caps)", "PLerr");
  legend_res->AddEntry("model", "DSCB Fit", "L");
  legend_res->Draw("same");
  header->DrawLatexNDC(0.12, 0.96, "MC Simulation");
  header->DrawLatexNDC(0.81, 0.96, "#sqrt{s} = 14 TeV");
  header->DrawLatexNDC(0.66, 0.75, TString::Format("#mu = %.4f #pm %.4f", mean_fit, err_mean));
  header->DrawLatexNDC(0.66, 0.71, TString::Format("#sigma_{R} = %.4f #pm %.4f", sigmaR_fit, err_sigmaR));
  header->DrawLatexNDC(0.66, 0.67, TString::Format("#sigma_{L} = %.4f #pm %.4f", sigmaL_fit, err_sigmaL));
  if (str.Contains("Ratio"))
    header->DrawLatexNDC(0.66, 0.63, TString::Format("#sigma (0.68) = %.3f #pm %.3f", res, bin_width));
  if (str.Contains("Ratio"))
    header->DrawLatexNDC(0.66, 0.59, TString::Format("#chi^{2}/ndf = %.2f", xresframe->chiSquare()));
  if (str.Contains("Res"))
    header->DrawLatexNDC(0.66, 0.63, TString::Format("#chi^{2}/ndf = %.2f", xresframe->chiSquare()));
  c1->Print(str);
}
