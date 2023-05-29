//////////////////////////////////////////////////////////////////////////////
//
// Usage:
// .L MakeECPlots.C+g
//
//   To make plot from a file created using EcalSimHitStudy
//     makeECPlots(fname, text, prefix, save);
//
//   where
//     fname   std::string   Name of the ROOT file ("runWithGun_Fix.root")
//     text    std::string   Text written in the Canvas
//                           ("Fixed Depth Calculation")
//     prefix  std::string   Text added to the name of the Canvas ("Fix")
//     save    bool          Flag to save the canvas as a gif file (false)
//
//   To compare plots from 2 different runs with EcalSimHitStudy with
//   "old" and "new" settings
//      comparePlots(dirname, tex, mom, ratio, fname, save=false)
//
//   where
//     dirname std::string   Name of the directory ("EcalSimHitStudy")
//     text    std::string   Postfix to the histogram name ("All")
//     mom     int           Momentum of the single particle used which
//                           is also a part of the file name (10)
//     ratio   bool          Shows both distributions or plot the ratio (false)
//     fname   std:string    Prefix of the file name ("elec")
//     save    bool          Flag to save the canvas as a gif file (false)
//
//////////////////////////////////////////////////////////////////////////////

#include <TROOT.h>
#include <TChain.h>
#include <TFile.h>
#include <TH1D.h>
#include <TH2D.h>
#include <TProfile.h>
#include <TProfile2D.h>
#include <TFitResult.h>
#include <TFitResultPtr.h>
#include <TStyle.h>
#include <TCanvas.h>
#include <TLegend.h>
#include <TPaveStats.h>
#include <TPaveText.h>
#include <vector>
#include <string>
#include <iomanip>
#include <iostream>
#include <fstream>

void makeECPlots(std::string fname = "runWithGun_Fix.root",
                 std::string text = "Fixed Depth Calculation",
                 std::string prefix = "Fix",
                 bool save = false) {
  std::string name[4] = {"ECLL_EB", "ECLL_EBref", "ECLL_EE", "ECLL_EERef"};
  double xrnglo[4] = {1200.0, 1200.0, 3100.0, 3100.0};
  double xrnghi[4] = {1600.0, 1600.0, 3600.0, 3600.0};
  std::string xtitl[4] = {"R_{Cyl} (mm)", "R_{Cyl} (mm)", "|z| (mm)", "|z| (mm)"};
  std::string ytitl[4] = {"# X_{0} (*100)", "# X_{0} (*100)", "# X_{0} (*100)", "# X_{0} (*100)"};
  std::string title[4] = {"EB (Normal)", "EB (Reflected)", "EE (Normal)", "EE (Reflected)"};

  gStyle->SetCanvasBorderMode(0);
  gStyle->SetCanvasColor(kWhite);
  gStyle->SetPadColor(kWhite);
  gStyle->SetFillColor(kWhite);
  gStyle->SetOptStat(11110);
  TFile *file = new TFile(fname.c_str());
  if (file) {
    char namep[100];
    for (int k = 0; k < 4; ++k) {
      TH2D *hist(0);
      for (int i = 0; i < 4; ++i) {
        if (i == 0)
          sprintf(namep, "%s", name[k].c_str());
        else
          sprintf(namep, "%s;%d", name[k].c_str(), i);
        hist = (TH2D *)file->FindObjectAny(name[k].c_str());
        std::cout << namep << " read out at " << hist << std::endl;
        if (hist != 0) {
          std::cout << "Entries " << hist->GetEntries() << std::endl;
          if (hist->GetEntries() > 0)
            break;
        }
      }
      if (hist != 0) {
        sprintf(namep, "%s%s", name[k].c_str(), prefix.c_str());
        TCanvas *pad = new TCanvas(namep, namep, 500, 500);
        pad->SetRightMargin(0.10);
        pad->SetTopMargin(0.10);
        hist->GetYaxis()->SetTitle(ytitl[k].c_str());
        hist->GetXaxis()->SetTitle(xtitl[k].c_str());
        hist->SetTitle(title[k].c_str());
        hist->GetXaxis()->SetRangeUser(xrnglo[k], xrnghi[k]);
        hist->GetYaxis()->SetTitleOffset(1.4);
        hist->Draw();
        pad->Update();
        TPaveStats *st1 = (TPaveStats *)hist->GetListOfFunctions()->FindObject("stats");
        if (st1 != NULL) {
          st1->SetY1NDC(0.70);
          st1->SetY2NDC(0.90);
          st1->SetX1NDC(0.65);
          st1->SetX2NDC(0.90);
        }
        TPaveText *txt1 = new TPaveText(0.50, 0.60, 0.90, 0.65, "blNDC");
        txt1->SetFillColor(0);
        txt1->AddText(text.c_str());
        pad->Update();
        if (save) {
          sprintf(namep, "c_%s%s.gif", name[k].c_str(), prefix.c_str());
          pad->Print(namep);
        }
      }
    }
  }
}

std::vector<TCanvas *> comparePlots(std::string dirname = "EcalSimHitStudy",
                                    std::string text = "All",
                                    int mom = 10,
                                    bool ratio = false,
                                    std::string fname = "elec",
                                    bool save = false) {
  std::vector<TCanvas *> tcvs;
  gStyle->SetCanvasBorderMode(0);
  gStyle->SetCanvasColor(kWhite);
  gStyle->SetPadColor(kWhite);
  gStyle->SetFillColor(kWhite);
  gStyle->SetOptTitle(0);
  gStyle->SetOptFit(0);
  if (ratio)
    gStyle->SetOptStat(0);
  else
    gStyle->SetOptStat(1100);

  std::string tags[2] = {"Old", "New"};
  int color[2] = {2, 4};
  int marker[2] = {20, 21};
  int style[2] = {1, 2};
  int rebin[16] = {50, 50, 50, 50, 2, 2, 2, 2, 2, 2, 20, 20, 20, 20, 20, 20};
  int type[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  int edgex[16] = {0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0};
  std::string name1[16] = {"Etot0",
                           "Etot1",
                           "EtotG0",
                           "EtotG1",
                           "r1by250",
                           "r1by251",
                           "r1by90",
                           "r1by91",
                           "r9by250",
                           "r9by251",
                           "sEtaEta0",
                           "sEtaEta1",
                           "sEtaPhi0",
                           "sEtaPhi1",
                           "sPhiPhi0",
                           "sPhiPhi1"};
  char name[100];
  TFile *file[2];
  TDirectory *dir[2];
  for (int i = 0; i < 2; ++i) {
    sprintf(name, "%s%d%s.root", fname.c_str(), mom, tags[i].c_str());
    file[i] = new TFile(name);
    dir[i] = (TDirectory *)file[i]->FindObjectAny(dirname.c_str());
  }
  for (int k = 0; k < 16; ++k) {
    TH1D *hist[2];
    sprintf(name, "%s", name1[k].c_str());
    for (int i = 0; i < 2; ++i) {
      hist[i] = (TH1D *)dir[i]->FindObjectAny(name);
      if (hist[i] != 0) {
        hist[i]->GetXaxis()->SetLabelOffset(0.005);
        hist[i]->GetXaxis()->SetTitleOffset(1.00);
        hist[i]->GetYaxis()->SetLabelOffset(0.005);
        hist[i]->GetYaxis()->SetTitleOffset(1.20);
        hist[i]->SetMarkerStyle(marker[i]);
        hist[i]->SetMarkerColor(color[i]);
        hist[i]->SetLineColor(color[i]);
        hist[i]->SetLineStyle(style[i]);
        hist[i]->SetLineWidth(2);
      }
    }
    if (hist[0] != 0 && hist[1] != 0) {
      double ytop(0.90), dy(0.05);
      double xmin = (edgex[k] == 0) ? 0.65 : 0.11;
      double xmin1 = (edgex[k] == 0) ? 0.55 : 0.11;
      double ymax = ratio ? (ytop - 0.01) : (ytop - 2 * dy - 0.01);
      double ymin = ratio ? (ymax - 0.045) : (ymax - 0.09);
      TLegend *legend = new TLegend(xmin1, ymin, xmin1 + 0.35, ymax);
      legend->SetFillColor(kWhite);
      if (ratio) {
        sprintf(name, "c_R%sE%d%s", name1[k].c_str(), mom, text.c_str());
      } else {
        sprintf(name, "c_%sE%d%s", name1[k].c_str(), mom, text.c_str());
      }
      TCanvas *pad = new TCanvas(name, name, 700, 500);
      pad->SetRightMargin(0.10);
      pad->SetTopMargin(0.10);
      if (type[k] != 0)
        pad->SetLogy();
      if (ratio) {
        int nbin = hist[0]->GetNbinsX();
        int nbinR = nbin / rebin[k];
        double xlow = hist[0]->GetXaxis()->GetBinLowEdge(1);
        double xhigh = hist[0]->GetXaxis()->GetBinLowEdge(nbin) + hist[0]->GetXaxis()->GetBinWidth(nbin);
        ;
        sprintf(name, "%sRatio", name1[k].c_str());
        TH1D *histr = new TH1D(name, hist[0]->GetTitle(), nbinR, xlow, xhigh);
        sprintf(name, "Ratio (%s/%s)", tags[0].c_str(), tags[1].c_str());
        histr->GetXaxis()->SetTitle(hist[0]->GetXaxis()->GetTitle());
        histr->GetYaxis()->SetTitle(name);
        histr->GetXaxis()->SetLabelOffset(0.005);
        histr->GetXaxis()->SetTitleOffset(1.00);
        histr->GetYaxis()->SetLabelOffset(0.005);
        histr->GetYaxis()->SetTitleOffset(1.20);
        histr->GetYaxis()->SetRangeUser(0.0, 2.0);
        histr->SetMarkerStyle(marker[0]);
        histr->SetMarkerColor(color[0]);
        histr->SetLineColor(color[0]);
        histr->SetLineStyle(style[0]);
        for (int j = 0; j < nbinR; ++j) {
          double tnum(0), tden(0), rnum(0), rden(0);
          for (int i = 0; i < rebin[k]; ++i) {
            int ib = j * rebin[k] + i + 1;
            tnum += hist[0]->GetBinContent(ib);
            tden += hist[1]->GetBinContent(ib);
            rnum += ((hist[0]->GetBinError(ib)) * (hist[0]->GetBinError(ib)));
            rden += ((hist[1]->GetBinError(ib)) * (hist[1]->GetBinError(ib)));
          }
          if (tden > 0 && tnum > 0) {
            double rat = tnum / tden;
            double err = rat * sqrt((rnum / (tnum * tnum)) + (rden / (tden * tden)));
            histr->SetBinContent(j + 1, rat);
            histr->SetBinError(j + 1, err);
          }
        }
        histr->Draw();
        sprintf(name, "%d GeV Electron (%s)", mom, text.c_str());
        legend->AddEntry(histr, name, "lp");
        pad->Update();
        TLine *line = new TLine(xlow, 1.0, xhigh, 1.0);
        line->SetLineColor(color[1]);
        line->SetLineWidth(2);
        line->SetLineStyle(2);
        line->Draw("same");
        pad->Update();
      } else {
        double mean[2], error[2];
        for (int i = 0; i < 2; ++i) {
          if (rebin[k] > 1)
            hist[i]->Rebin(rebin[k]);
          if (i == 0)
            hist[i]->Draw("hist");
          else
            hist[i]->Draw("sameshist");
          pad->Update();
          sprintf(name, "%d GeV Electron (%s %s)", mom, text.c_str(), tags[i].c_str());
          legend->AddEntry(hist[i], name, "lp");
          TPaveStats *st1 = (TPaveStats *)hist[i]->GetListOfFunctions()->FindObject("stats");
          if (st1 != NULL) {
            st1->SetLineColor(color[i]);
            st1->SetTextColor(color[i]);
            st1->SetY1NDC(ytop - dy);
            st1->SetY2NDC(ytop);
            st1->SetX1NDC(xmin);
            st1->SetX2NDC(xmin + 0.25);
            ytop -= dy;
          }
          double entries = hist[i]->GetEntries();
          mean[i] = hist[i]->GetMean();
          error[i] = (hist[i]->GetRMS()) / sqrt(entries);
          std::cout << text << ":" << hist[i]->GetName() << " V " << tags[i] << " Mean " << mean[i] << " RMS "
                    << hist[i]->GetRMS() << " Error " << error[i] << std::endl;
        }
        double diff = 0.5 * (mean[0] - mean[1]) / (mean[0] + mean[1]);
        double ddiff =
            (sqrt((mean[0] * error[1]) * (mean[0] * error[1]) + (mean[1] * error[0]) * (mean[1] * error[0])) /
             ((mean[0] * mean[0]) + (mean[1] * mean[1])));
        double sign = std::abs(diff) / ddiff;
        std::cout << "Difference " << diff << " +- " << ddiff << " Significance " << sign << std::endl;
        pad->Modified();
        pad->Update();
      }
      if (ratio) {
      }
      legend->Draw("same");
      pad->Modified();
      pad->Update();
      tcvs.push_back(pad);
      if (save) {
        sprintf(name, "%s.pdf", pad->GetName());
        pad->Print(name);
      }
    }
  }
  return tcvs;
}
