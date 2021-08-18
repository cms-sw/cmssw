// =============================================================================
//
//  This macro reads the histograms of time residuals of the BTL and ETL
//  UncalibratedRecHits from the DQM root file and produces time resolution
//  plots vs the hit amplitude and the hit |eta|.
//
//  Usage:
//     root -l make_resPlot.C\(\"DQM_filename.root\"\)
//
//  NB: In order to have the UncalibratedRecHits histograms filled,
//      the MTD validation has to be run the flags:
//
//      process.btlLocalReco.UncalibRecHitsPlots = cms.bool(True);
//      process.etlLocalReco.UncalibRecHitsPlots = cms.bool(True);
//
// =============================================================================

#include <iostream>
#include "TStyle.h"
#include "TFile.h"
#include "TString.h"
#include "TCanvas.h"
#include "TH1F.h"
#include "TFitResult.h"
#include "TLegend.h"

// =============================================================================
//  configuration

const Float_t minBinContent = 20.;

// --- BTL

const Int_t nBinsQ_BTL = 20;
const Float_t binWidthQ_BTL = 30.;  // [pC]
const Int_t nBinsQEta_BTL = 3;
const Float_t binsQEta_BTL[4] = {0., 0.65, 1.15, 1.55};

const Int_t nBinsEta_BTL = 31;
const Float_t binWidthEta_BTL = 0.05;
const Int_t nBinsEtaQ_BTL = 7;
const Float_t binsEtaQ_BTL[8] = {0., 30., 60., 90., 120., 150., 360., 600.};

// --- ETL

const Int_t nBinsQ_ETL = 20;
const Float_t binWidthQ_ETL = 1.3;  // [MIP]

const Int_t nBinsEta_ETL = 26;
const Float_t binWidthEta_ETL = 0.05;
const Float_t etaMin_ETL = 1.65;

// =============================================================================

TCanvas* c[10];

TH1F* h_TimeResQ_BTL[nBinsQ_BTL];
TH1F* h_TimeResQEta_BTL[nBinsQ_BTL][nBinsQEta_BTL];
TH1F* h_TimeResEta_BTL[nBinsEta_BTL];
TH1F* h_TimeResEtaQ_BTL[nBinsEta_BTL][nBinsEtaQ_BTL];

TH1F* g_TimeResQ_BTL;
TH1F* g_TimeResQEta_BTL[nBinsQEta_BTL];
TH1F* g_TimeResEta_BTL;
TH1F* g_TimeResEtaQ_BTL[nBinsEtaQ_BTL];

TH1F* h_TimeResQ_ETL[2][nBinsQ_ETL];
TH1F* h_TimeResEta_ETL[2][nBinsEta_ETL];

TH1F* g_TimeResQ_ETL[2];
TH1F* g_TimeResEta_ETL[2];

void makeTimeResPlots(const TString DQMfilename = "DQM_V0001_UNKNOWN_R000000001.root") {
  gStyle->SetOptStat(kFALSE);

  // --- Histograms booking

  g_TimeResQ_BTL = new TH1F("g_TimeResQ_BTL", "BTL time resolution vs Q", nBinsQ_BTL, 0., nBinsQ_BTL * binWidthQ_BTL);

  for (Int_t ih = 0; ih < nBinsQEta_BTL; ++ih) {
    TString hname = Form("g_TimeResQEta_BTL_%d", ih);
    TString htitle =
        Form("BTL time resolution vs Q (%4.2f < |#eta_{hit}| < %4.2f)", binsQEta_BTL[ih], binsQEta_BTL[ih + 1]);
    g_TimeResQEta_BTL[ih] = new TH1F(hname, htitle, nBinsQ_BTL, 0., nBinsQ_BTL * binWidthQ_BTL);
  }

  g_TimeResEta_BTL = new TH1F(
      "g_TimeResEta_BTL", "BTL time resolution vs |#eta_{hit}|", nBinsEta_BTL, 0., nBinsEta_BTL * binWidthEta_BTL);

  for (Int_t ih = 0; ih < nBinsEtaQ_BTL; ++ih) {
    TString hname = Form("g_TimeResEtaQ_BTL_%d", ih);
    TString htitle =
        Form("BTL time resolution vs |#eta_{hit}| (%4.2f < Q < %4.2f)", binsEtaQ_BTL[ih], binsEtaQ_BTL[ih + 1]);
    g_TimeResEtaQ_BTL[ih] = new TH1F(hname, htitle, nBinsEta_BTL, 0., nBinsEta_BTL * binWidthEta_BTL);
  }

  for (Int_t iside = 0; iside < 2; ++iside) {
    TString hname = Form("g_TimeResQ_ETL_%d", iside);
    g_TimeResQ_ETL[iside] =
        new TH1F(hname, "ETL time resolution vs amplitude", nBinsQ_ETL, 0., nBinsQ_ETL * binWidthQ_ETL);

    hname = Form("g_TimeResEta_ETL_%d", iside);
    g_TimeResEta_ETL[iside] = new TH1F(hname,
                                       "ETL time resolution vs |#eta_{hit}|",
                                       nBinsEta_ETL,
                                       etaMin_ETL,
                                       etaMin_ETL + nBinsEta_ETL * binWidthEta_ETL);
  }

  std::cout << " Processing file " << DQMfilename.Data() << " ... " << std::endl;
  TFile* input_file = new TFile(DQMfilename.Data());

  // ---------------------------------------------------------------------------
  // BTL time resolution vs Q

  for (Int_t iq = 0; iq < nBinsQ_BTL; ++iq) {
    TString hname = Form("DQMData/Run 1/MTD/Run summary/BTL/LocalReco/TimeResQ_%d", iq);
    h_TimeResQ_BTL[iq] = (TH1F*)input_file->Get(hname);

    if (h_TimeResQ_BTL[iq]->GetEntries() < minBinContent)
      continue;

    Float_t low_limit = h_TimeResQ_BTL[iq]->GetMean() - 2. * h_TimeResQ_BTL[iq]->GetRMS();
    Float_t up_limit = h_TimeResQ_BTL[iq]->GetMean() + 3 * h_TimeResQ_BTL[iq]->GetRMS();

    TFitResultPtr fit_res = h_TimeResQ_BTL[iq]->Fit("gaus", "SERQ0", "", low_limit, up_limit);

    if (fit_res->Status() != 0) {
      std::cout << " *** ERROR: fit failed for the histogram" << h_TimeResQ_BTL[iq]->GetName() << std::endl;
      continue;
    }

    g_TimeResQ_BTL->SetBinContent(iq + 1, fit_res->GetParams()[2]);
    g_TimeResQ_BTL->SetBinError(iq + 1, fit_res->GetErrors()[2]);

  }  // iq loop

  // ---------------------------------------------------------------------------
  // BTL time resolution vs Q in bind of |eta|

  for (Int_t iq = 0; iq < nBinsQ_BTL; ++iq) {
    for (Int_t ieta = 0; ieta < nBinsQEta_BTL; ++ieta) {
      TString hname = Form("DQMData/Run 1/MTD/Run summary/BTL/LocalReco/TimeResQvsEta_%d_%d", iq, ieta);
      h_TimeResQEta_BTL[iq][ieta] = (TH1F*)input_file->Get(hname);

      if (h_TimeResQEta_BTL[iq][ieta]->GetEntries() < minBinContent)
        continue;

      Float_t low_limit = h_TimeResQEta_BTL[iq][ieta]->GetMean() - 2. * h_TimeResQEta_BTL[iq][ieta]->GetRMS();
      Float_t up_limit = h_TimeResQEta_BTL[iq][ieta]->GetMean() + 3 * h_TimeResQEta_BTL[iq][ieta]->GetRMS();

      TFitResultPtr fit_res = h_TimeResQEta_BTL[iq][ieta]->Fit("gaus", "SERQ0", "", low_limit, up_limit);

      if (fit_res->Status() != 0) {
        std::cout << " *** ERROR: fit failed for the histogram" << h_TimeResQEta_BTL[iq][ieta]->GetName() << std::endl;
        continue;
      }

      g_TimeResQEta_BTL[ieta]->SetBinContent(iq + 1, fit_res->GetParams()[2]);
      g_TimeResQEta_BTL[ieta]->SetBinError(iq + 1, fit_res->GetErrors()[2]);

    }  // ieta loop

  }  // iq loop

  // ---------------------------------------------------------------------------
  // BTL time resolution vs |eta|

  for (Int_t ieta = 0; ieta < nBinsEta_BTL; ++ieta) {
    TString hname = Form("DQMData/Run 1/MTD/Run summary/BTL/LocalReco/TimeResEta_%d", ieta);
    h_TimeResEta_BTL[ieta] = (TH1F*)input_file->Get(hname);

    if (h_TimeResEta_BTL[ieta]->GetEntries() < minBinContent)
      continue;

    Float_t low_limit = h_TimeResEta_BTL[ieta]->GetMean() - 2. * h_TimeResEta_BTL[ieta]->GetRMS();
    Float_t up_limit = h_TimeResEta_BTL[ieta]->GetMean() + 3 * h_TimeResEta_BTL[ieta]->GetRMS();

    TFitResultPtr fit_res = h_TimeResEta_BTL[ieta]->Fit("gaus", "SERQ0", "", low_limit, up_limit);

    if (fit_res->Status() != 0) {
      std::cout << " *** ERROR: fit failed for the histogram" << h_TimeResEta_BTL[ieta]->GetName() << std::endl;
      continue;
    }

    g_TimeResEta_BTL->SetBinContent(ieta + 1, fit_res->GetParams()[2]);
    g_TimeResEta_BTL->SetBinError(ieta + 1, fit_res->GetErrors()[2]);

  }  // ieta loop

  // ---------------------------------------------------------------------------
  // BTL time resolution vs |eta| in bind of Q

  for (Int_t ieta = 0; ieta < nBinsEta_BTL; ++ieta) {
    for (Int_t iq = 0; iq < nBinsEtaQ_BTL; ++iq) {
      TString hname = Form("DQMData/Run 1/MTD/Run summary/BTL/LocalReco/TimeResEtavsQ_%d_%d", ieta, iq);
      h_TimeResEtaQ_BTL[ieta][iq] = (TH1F*)input_file->Get(hname);

      if (h_TimeResEtaQ_BTL[ieta][iq]->GetEntries() < minBinContent)
        continue;

      Float_t low_limit = h_TimeResEtaQ_BTL[ieta][iq]->GetMean() - 2. * h_TimeResEtaQ_BTL[ieta][iq]->GetRMS();
      Float_t up_limit = h_TimeResEtaQ_BTL[ieta][iq]->GetMean() + 3 * h_TimeResEtaQ_BTL[ieta][iq]->GetRMS();

      TFitResultPtr fit_res = h_TimeResEtaQ_BTL[ieta][iq]->Fit("gaus", "SERQ0", "", low_limit, up_limit);

      if (fit_res->Status() != 0) {
        std::cout << " *** ERROR: fit failed for the histogram" << h_TimeResEtaQ_BTL[ieta][iq]->GetName() << std::endl;
        continue;
      }

      g_TimeResEtaQ_BTL[iq]->SetBinContent(ieta + 1, fit_res->GetParams()[2]);
      g_TimeResEtaQ_BTL[iq]->SetBinError(ieta + 1, fit_res->GetErrors()[2]);

    }  // iq loop

  }  // ieta loop

  // ---------------------------------------------------------------------------
  // ETL time resolution vs amplitude and |eta|

  for (Int_t iside = 0; iside < 2; ++iside) {
    for (Int_t iq = 0; iq < nBinsQ_ETL; ++iq) {
      TString hname = Form("DQMData/Run 1/MTD/Run summary/ETL/LocalReco/TimeResQ_%d_%d", iside, iq);
      h_TimeResQ_ETL[iside][iq] = (TH1F*)input_file->Get(hname);

      if (h_TimeResQ_ETL[iside][iq]->GetEntries() < minBinContent)
        continue;

      Float_t low_limit = h_TimeResQ_ETL[iside][iq]->GetMean() - 2. * h_TimeResQ_ETL[iside][iq]->GetRMS();
      Float_t up_limit = h_TimeResQ_ETL[iside][iq]->GetMean() + 3 * h_TimeResQ_ETL[iside][iq]->GetRMS();

      TFitResultPtr fit_res = h_TimeResQ_ETL[iside][iq]->Fit("gaus", "SERQ0", "", low_limit, up_limit);

      if (fit_res->Status() != 0) {
        std::cout << " *** ERROR: fit failed for the histogram" << h_TimeResQ_ETL[iside][iq]->GetName() << std::endl;
        continue;
      }

      g_TimeResQ_ETL[iside]->SetBinContent(iq + 1, fit_res->GetParams()[2]);
      g_TimeResQ_ETL[iside]->SetBinError(iq + 1, fit_res->GetErrors()[2]);

    }  // iq loop

    for (Int_t ieta = 0; ieta < nBinsEta_ETL; ++ieta) {
      TString hname = Form("DQMData/Run 1/MTD/Run summary/ETL/LocalReco/TimeResEta_%d_%d", iside, ieta);
      h_TimeResEta_ETL[iside][ieta] = (TH1F*)input_file->Get(hname);

      if (h_TimeResEta_ETL[iside][ieta]->GetEntries() < minBinContent)
        continue;

      Float_t low_limit = h_TimeResEta_ETL[iside][ieta]->GetMean() - 2. * h_TimeResEta_ETL[iside][ieta]->GetRMS();
      Float_t up_limit = h_TimeResEta_ETL[iside][ieta]->GetMean() + 3 * h_TimeResEta_ETL[iside][ieta]->GetRMS();

      TFitResultPtr fit_res = h_TimeResEta_ETL[iside][ieta]->Fit("gaus", "SERQ0", "", low_limit, up_limit);

      if (fit_res->Status() != 0) {
        std::cout << " *** ERROR: fit failed for the histogram" << h_TimeResEta_ETL[iside][ieta]->GetName()
                  << std::endl;
        continue;
      }

      g_TimeResEta_ETL[iside]->SetBinContent(ieta + 1, fit_res->GetParams()[2]);
      g_TimeResEta_ETL[iside]->SetBinError(ieta + 1, fit_res->GetErrors()[2]);

    }  // ieta loop

  }  // iside loop

  // =============================================================================
  // Draw the histograms

  // --- BTL

  c[0] = new TCanvas("c_0", "BTL time resolution vs Q");
  c[0]->SetGridy(kTRUE);
  g_TimeResQ_BTL->SetTitle("BTL UncalibratedRecHits");
  g_TimeResQ_BTL->SetMarkerStyle(20);
  g_TimeResQ_BTL->SetMarkerColor(4);
  g_TimeResQ_BTL->SetXTitle("hit amplitude [pC]");
  g_TimeResQ_BTL->SetYTitle("#sigma_{T} [ns]");
  g_TimeResQ_BTL->GetYaxis()->SetRangeUser(0., 0.075);
  g_TimeResQ_BTL->Draw("EP");

  c[1] = new TCanvas("c_1", "BTL time resolution vs Q");
  c[1]->SetGridy(kTRUE);
  g_TimeResQEta_BTL[0]->SetTitle("BTL UncalibratedRecHits");
  g_TimeResQEta_BTL[0]->SetXTitle("hit amplitude [pC]");
  g_TimeResQEta_BTL[0]->SetYTitle("#sigma_{T} [ns]");
  g_TimeResQEta_BTL[0]->GetYaxis()->SetRangeUser(0., 0.075);
  g_TimeResQEta_BTL[0]->SetMarkerStyle(20);
  g_TimeResQEta_BTL[0]->SetMarkerColor(1);
  g_TimeResQEta_BTL[0]->Draw("EP");
  g_TimeResQEta_BTL[1]->SetMarkerStyle(20);
  g_TimeResQEta_BTL[1]->SetMarkerColor(4);
  g_TimeResQEta_BTL[1]->Draw("EPSAME");
  g_TimeResQEta_BTL[2]->SetMarkerStyle(20);
  g_TimeResQEta_BTL[2]->SetMarkerColor(2);
  g_TimeResQEta_BTL[2]->Draw("EPSAME");

  TLegend* legend_1 = new TLegend(0.510, 0.634, 0.862, 0.847);
  legend_1->SetBorderSize(0.);
  legend_1->SetFillStyle(0);
  for (Int_t ih = 0; ih < nBinsQEta_BTL; ++ih)
    legend_1->AddEntry(
        g_TimeResQEta_BTL[ih], Form("%4.2f #leq |#eta_{hit}| < %4.2f", binsQEta_BTL[ih], binsQEta_BTL[ih + 1]), "LP");
  legend_1->Draw();

  c[2] = new TCanvas("c_2", "BTL time resolution vs eta");
  c[2]->SetGridy(kTRUE);
  g_TimeResEta_BTL->SetTitle("BTL UncalibratedRecHits");
  g_TimeResEta_BTL->SetMarkerStyle(20);
  g_TimeResEta_BTL->SetMarkerColor(4);
  g_TimeResEta_BTL->SetXTitle("|#eta_{hit}|");
  g_TimeResEta_BTL->SetYTitle("#sigma_{T} [ns]");
  g_TimeResEta_BTL->GetYaxis()->SetRangeUser(0., 0.075);
  g_TimeResEta_BTL->Draw("EP");

  c[3] = new TCanvas("c_3", "BTL time resolution vs eta");
  c[3]->SetGridy(kTRUE);
  g_TimeResEtaQ_BTL[1]->SetTitle("BTL UncalibratedRecHits");
  g_TimeResEtaQ_BTL[1]->SetXTitle("|#eta_{hit}|");
  g_TimeResEtaQ_BTL[1]->SetYTitle("#sigma_{T} [ns]");
  g_TimeResEtaQ_BTL[1]->GetYaxis()->SetRangeUser(0., 0.075);
  g_TimeResEtaQ_BTL[1]->SetMarkerStyle(20);
  g_TimeResEtaQ_BTL[1]->SetMarkerColor(1);
  g_TimeResEtaQ_BTL[1]->Draw("EP");

  TLegend* legend_3 = new TLegend(0.649, 0.124, 0.862, 0.328);
  legend_3->SetBorderSize(0.);
  legend_3->SetFillStyle(0);
  for (Int_t ih = 1; ih < nBinsEtaQ_BTL; ++ih) {
    if (ih > 1) {
      g_TimeResEtaQ_BTL[ih]->SetMarkerStyle(20);
      g_TimeResEtaQ_BTL[ih]->SetMarkerColor(ih);
      g_TimeResEtaQ_BTL[ih]->Draw("EPSAME");
    }

    legend_3->AddEntry(
        g_TimeResEtaQ_BTL[ih], Form("%4.0f #leq Q_{hit} < %4.0f pC", binsEtaQ_BTL[ih], binsEtaQ_BTL[ih + 1]), "LP");
  }

  legend_3->Draw();

  // --- ETL

  c[4] = new TCanvas("c_4", "ETL time resolution vs amplitude");
  c[4]->SetGridy(kTRUE);
  g_TimeResQ_ETL[0]->SetTitle("ETL UncalibratedRecHits");
  g_TimeResQ_ETL[0]->SetXTitle("hit amplitude [MIP]");
  g_TimeResQ_ETL[0]->SetYTitle("#sigma_{T} [ns]");
  g_TimeResQ_ETL[0]->GetYaxis()->SetRangeUser(0., 0.075);
  g_TimeResQ_ETL[0]->SetMarkerStyle(20);
  g_TimeResQ_ETL[0]->SetMarkerColor(4);
  g_TimeResQ_ETL[0]->Draw("EP");
  g_TimeResQ_ETL[1]->SetMarkerStyle(20);
  g_TimeResQ_ETL[1]->SetMarkerColor(2);
  g_TimeResQ_ETL[1]->Draw("EPSAME");

  TLegend* legend_4 = new TLegend(0.673, 0.655, 0.872, 0.819);
  legend_4->SetBorderSize(0.);
  legend_4->SetFillStyle(0);
  legend_4->AddEntry(g_TimeResQ_ETL[0], "ETL-", "LP");
  legend_4->AddEntry(g_TimeResQ_ETL[1], "ETL+", "LP");
  legend_4->DrawClone();

  c[5] = new TCanvas("c_5", "ETL time resolution vs |eta|");
  c[5]->SetGridy(kTRUE);
  g_TimeResEta_ETL[0]->SetTitle("ETL UncalibratedRecHits");
  g_TimeResEta_ETL[0]->SetXTitle("|#eta_{hit}|");
  g_TimeResEta_ETL[0]->SetYTitle("#sigma_{T} [ns]");
  g_TimeResEta_ETL[0]->GetYaxis()->SetRangeUser(0., 0.075);
  g_TimeResEta_ETL[0]->SetMarkerStyle(20);
  g_TimeResEta_ETL[0]->SetMarkerColor(4);
  g_TimeResEta_ETL[0]->Draw("EP");
  g_TimeResEta_ETL[1]->SetMarkerStyle(20);
  g_TimeResEta_ETL[1]->SetMarkerColor(2);
  g_TimeResEta_ETL[1]->Draw("EPSAME");

  legend_4->DrawClone();
}
