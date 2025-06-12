//////////////////////////////////////////////////////////////////////////////
//
// Usage:
// .L MakeHitStudyPlots.C+g
//
//   To make plot of various quantities which are created by CaloSimHitStudy
//   from one or two different settings
//
//     makeHitStudyPlots(file1, tag1, file2, tag2, todomin, todomax, gtitle,
//                       ratio, save, dirnm)
//
//   To make plots of ratios of hits produced with identical geometry and
//   generator level files created by HGCalHitCheck
//
//      makeDDDvsDD4hepPlots(dirnm, inType, geometry, layer, ratio, save)
//
//   To make plots of digitization variables from digitiasation campaigns
//   with FullSim_Signal+FullSim_PU vs FasrSim_Signal+FastSim_PU vs
//   FullSim_Signal+FastSim_PU
//
//      makeDigiStudyPlots(tag, todomin, todomax, save)
//
//   where (for makeHitStudyPlots)
//     file1    std::string   Name of the first ROOT file [old/analRun3.root]
//     file2    std::string   Name of the second ROOT file [new/analRun3.root]
//     tag1     std::string   Tag for the first file [Bug]
//     tag2     std::string   Tag for the second file [Fix]
//     gtitle   std::string   Overall Titile [""]
//     todomin  int           Minimum type # of histograms to be plotted [0]
//     todomax  int           Maximum type # of histograms to be plotted [0]
//     dirnm    std::string   Name of the directory [CaloSimHitStudy]
//
//   where (for makeHitStudyPlots)
//     dirnm    std::string   Directory name (EE/HEF/HEB)
//     inType   std::string   Name of the input data (Muon/MinBias)
//     geometry std::string   Tag for the geometry (D98/D99)
//     layer    int           Layer number (if 0; all layers combined)
//     ratio    bool          if the ratio to be plotted [true]
//                            (works when both files are active)
//
//   where (for makeHitStudyPlots)
//     tag      std::string   Detectr type (EC/HC)
//     todomin  int           Minimum type # of histograms to be plotted [0]
//     todomax  int           Maximum type # of histograms to be plotted [0]
//
//   where (common to all macros)
//     save     bool          If the canvas is to be saved as jpg file [false]
//
//////////////////////////////////////////////////////////////////////////////

#include <TCanvas.h>
#include <TChain.h>
#include <TFile.h>
#include <TFitResult.h>
#include <TFitResultPtr.h>
#include <TH1D.h>
#include <TH2D.h>
#include <TLegend.h>
#include <TPaveStats.h>
#include <TPaveText.h>
#include <TProfile.h>
#include <TROOT.h>
#include <TStyle.h>
#include <vector>
#include <string>
#include <iomanip>
#include <iostream>
#include <fstream>

void makeHitStudyPlots(std::string file1 = "uncorr/analRun3.root",
                       std::string file2 = "corr/analRun3.root",
                       std::string tag1 = "Bug",
                       std::string tag2 = "Fix",
                       std::string gtitle = "",
                       int todomin = 0,
                       int todomax = 0,
                       bool ratio = true,
                       bool save = false,
                       std::string dirnm = "CaloSimHitStudy") {
  const int plots = 20;
  std::string names[plots] = {"Etot",   "Hit",     "EtotG",  "Time",    "EdepTk", "Edep", "HitHigh",
                              "HitLow", "HitMu",   "HitTk",  "TimeAll", "TimeTk", "eta",  "phi",
                              "EdepEM", "EdepHad", "EneInc", "EtaInc",  "PhiInc", "PtInc"};
  int numb[plots] = {9, 9, 9, 9, 16, 9, 1, 1, 1, 16, 9, 16, 9, 9, 9, 9, 1, 1, 1, 1};
  int rebin[plots] = {10, 10, 10, 10, 1, 10, 10, 10, 10, 10, 10, 10, 2, 4, 10, 10, 1, 1, 1, 1};
  bool debug(false);

  gStyle->SetCanvasBorderMode(0);
  gStyle->SetCanvasColor(kWhite);
  gStyle->SetPadColor(kWhite);
  gStyle->SetFillColor(kWhite);
  if (ratio)
    gStyle->SetOptStat(0);
  else
    gStyle->SetOptStat(1110);
  TFile* file[2];
  int nfile(0);
  std::string tag(""), tags[2];
  if (file1 != "") {
    file[nfile] = new TFile(file1.c_str());
    if (file[nfile]) {
      tags[nfile] = tag1;
      ++nfile;
      tag += tag1;
    }
  }
  if (file2 != "") {
    file[nfile] = new TFile(file2.c_str());
    if (file[nfile]) {
      tags[nfile] = tag2;
      ++nfile;
      tag += tag2;
    }
  }
  if ((todomin < 0) || (todomin >= plots))
    todomin = 0;
  if (todomax < todomin) {
    todomax = todomin;
  } else if (todomax >= plots) {
    todomax = plots - 1;
  }
  std::cout << "Use " << nfile << " files from " << file1 << " and " << file2 << " and look for " << todomin << ":"
            << todomax << std::endl;
  for (int todo = todomin; todo <= todomax; ++todo) {
    for (int i1 = 0; i1 < numb[todo]; ++i1) {
      double y1(0.90), dy(0.12);
      double y2 = y1 - dy * nfile - 0.01;
      TLegend* leg = (ratio) ? (new TLegend(0.10, 0.86, 0.90, 0.90)) : (new TLegend(0.65, y2 - nfile * 0.04, 0.90, y2));
      leg->SetBorderSize(1);
      leg->SetFillColor(10);
      TH1D* hist0[nfile];
      char name[100], namec[100];
      if (numb[todo] == 1) {
        sprintf(name, "%s", names[todo].c_str());
        sprintf(namec, "c_%s%s%s", names[todo].c_str(), tag.c_str(), gtitle.c_str());
      } else {
        sprintf(name, "%s%d", names[todo].c_str(), i1);
        sprintf(namec, "c_%s%d%s%s", names[todo].c_str(), i1, tag.c_str(), gtitle.c_str());
      }
      TCanvas* pad = new TCanvas(namec, namec, 500, 500);
      for (int ifile = 0; ifile < nfile; ++ifile) {
        TDirectory* dir = (TDirectory*)file[ifile]->FindObjectAny(dirnm.c_str());
        hist0[ifile] = static_cast<TH1D*>(dir->FindObjectAny(name));
        if (debug)
          std::cout << name << " read out at " << hist0[ifile] << " for " << tags[ifile] << std::endl;
      }
      if (!ratio) {
        int first(0);
        for (int ifile = 0; ifile < nfile; ++ifile) {
          TH1D* hist(hist0[ifile]);
          if (hist != nullptr) {
            hist->SetLineColor(first + 1);
            hist->SetLineStyle(first + 1);
            hist->GetYaxis()->SetTitleOffset(1.4);
            if (rebin[todo] > 1)
              hist->Rebin(rebin[todo]);
            hist->SetTitle(gtitle.c_str());
            if (first == 0) {
              pad = new TCanvas(namec, namec, 500, 500);
              pad->SetRightMargin(0.10);
              pad->SetTopMargin(0.10);
              pad->SetLogy();
              hist->Draw();
            } else {
              hist->Draw("sames");
            }
            leg->AddEntry(hist, tags[ifile].c_str(), "lp");
            pad->Update();
            ++first;
            TPaveStats* st = ((TPaveStats*)hist->GetListOfFunctions()->FindObject("stats"));
            if (st != NULL) {
              st->SetLineColor(first);
              st->SetTextColor(first);
              st->SetY1NDC(y1 - dy);
              st->SetY2NDC(y1);
              st->SetX1NDC(0.65);
              st->SetX2NDC(0.90);
              y1 -= dy;
            }
            pad->Modified();
            pad->Update();
            leg->Draw("same");
            pad->Update();
            if (save) {
              sprintf(name, "%s.pdf", pad->GetName());
              pad->Print(name);
            }
          }
        }
      } else {
        if (nfile == 2) {
          int nbin = hist0[0]->GetNbinsX();
          int nbinR = nbin / rebin[todo];
          double xlow = hist0[0]->GetXaxis()->GetBinLowEdge(1);
          double xhigh = hist0[0]->GetXaxis()->GetBinLowEdge(nbin) + hist0[0]->GetXaxis()->GetBinWidth(nbin);
          if (numb[todo] == 1) {
            sprintf(name, "%sRatio", names[todo].c_str());
            sprintf(namec, "c_%sRatio%s%s", names[todo].c_str(), tag.c_str(), gtitle.c_str());
          } else {
            sprintf(name, "%s%dRatio", names[todo].c_str(), i1);
            sprintf(namec, "c_%s%dRatio%s%s", names[todo].c_str(), i1, tag.c_str(), gtitle.c_str());
          }
          pad = new TCanvas(namec, namec, 500, 500);
          TH1D* histr = new TH1D(name, hist0[0]->GetTitle(), nbinR, xlow, xhigh);
          sprintf(name, "Ratio (%s/%s)", tags[0].c_str(), tags[1].c_str());
          histr->SetTitle(gtitle.c_str());
          histr->GetXaxis()->SetTitle(hist0[0]->GetXaxis()->GetTitle());
          histr->GetYaxis()->SetTitle(name);
          histr->GetXaxis()->SetLabelOffset(0.005);
          histr->GetXaxis()->SetTitleOffset(1.00);
          histr->GetYaxis()->SetLabelOffset(0.005);
          histr->GetYaxis()->SetTitleOffset(1.20);
          histr->GetYaxis()->SetRangeUser(0.0, 2.0);
          double sumNum(0), sumDen(0), maxDev(0);
          for (int j = 0; j < nbinR; ++j) {
            double tnum(0), tden(0), rnum(0), rden(0);
            for (int i = 0; i < rebin[todo]; ++i) {
              int ib = j * rebin[todo] + i + 1;
              tnum += hist0[0]->GetBinContent(ib);
              tden += hist0[1]->GetBinContent(ib);
              rnum += ((hist0[0]->GetBinError(ib)) * (hist0[0]->GetBinError(ib)));
              rden += ((hist0[1]->GetBinError(ib)) * (hist0[1]->GetBinError(ib)));
            }
            if (tden > 0 && tnum > 0) {
              double rat = tnum / tden;
              double err = rat * sqrt((rnum / (tnum * tnum)) + (rden / (tden * tden)));
              histr->SetBinContent(j + 1, rat);
              histr->SetBinError(j + 1, err);
              double temp1 = (rat > 1.0) ? 1.0 / rat : rat;
              double temp2 = (rat > 1.0) ? err / (rat * rat) : err;
              sumNum += (fabs(1 - temp1) / (temp2 * temp2));
              sumDen += (1.0 / (temp2 * temp2));
              if (fabs(1 - temp1) > maxDev)
                maxDev = fabs(1 - temp1);
            }
          }
          histr->Draw();
          sprintf(name, "%s vs %s", tag1.c_str(), tag2.c_str());
          leg->AddEntry(histr, name, "lp");
          leg->Draw("same");
          pad->Update();
          TLine* line = new TLine(xlow, 1.0, xhigh, 1.0);
          line->SetLineColor(2);
          line->SetLineWidth(2);
          line->SetLineStyle(2);
          line->Draw("same");
          pad->Modified();
          pad->Update();
          sumNum = (sumDen > 0) ? (sumNum / sumDen) : 0;
          sumDen = (sumDen > 0) ? 1.0 / sqrt(sumDen) : 0;
          if (sumNum == 0)
            sumDen = 0;
          std::cout << tag1 << " vs " << tag2 << " " << hist0[0]->GetXaxis()->GetTitle() << " Mean deviation " << sumNum
                    << " +- " << sumDen << " maximum " << maxDev << std::endl;
          if (save) {
            sprintf(name, "%s.pdf", pad->GetName());
            pad->Print(name);
          }
        }
      }
    }
  }
}

void makeDDDvsDD4hepPlots(std::string dirnm = "EE",
                          std::string inType = "Muon",
                          std::string geometry = "D98",
                          int layer = 0,
                          bool ratio = true,
                          bool save = false) {
  const int plots = 3;
  std::string types[2] = {"DDD", "DD4hep"};
  std::string plotf[plots] = {"L", "F", "P"};
  std::string plotp[plots] = {"All", "Full|SiPM 2mm", "Partial|SiPM 4mm"};
  int rebins[2] = {4, 20};
  double xmaxs[2] = {1000, 5000};
  bool debug(false);

  gStyle->SetCanvasBorderMode(0);
  gStyle->SetCanvasColor(kWhite);
  gStyle->SetPadColor(kWhite);
  gStyle->SetFillColor(kWhite);
  if (ratio)
    gStyle->SetOptStat(0);
  else
    gStyle->SetOptStat(1110);
  TFile* file[2];
  int nfile(0);
  std::string tag(""), tags[2], filex[2];
  for (int i = 0; i < 2; ++i) {
    filex[i] = types[i] + geometry + inType + ".root";
    file[nfile] = new TFile(filex[i].c_str());
    if (file[nfile]) {
      tags[nfile] = types[i];
      ++nfile;
      tag += tags[nfile];
    }
  }
  char name[80], nameD[80], title[80];
  int rebin = (inType == "Muon") ? rebins[0] : rebins[1];
  double xmax = (inType == "Muon") ? xmaxs[0] : xmaxs[1];
  sprintf(nameD, "hgcalHitCheck%s", dirnm.c_str());
  sprintf(title, "%s vs %s for %s", types[0].c_str(), types[1].c_str(), inType.c_str());
  std::cout << "Use " << nfile << " files from " << filex[0] << " and " << filex[1] << " and look for " << plots
            << " plots in " << nameD << " with rebin " << rebin << " Max " << xmax << std::endl;
  for (int i = 0; i < plots; ++i) {
    if (layer == 0)
      sprintf(name, "Hits%s", plotf[i].c_str());
    else
      sprintf(name, "Hits%s%d", plotf[i].c_str(), layer);
    double y1(0.90), dy(0.12);
    double y2 = y1 - dy * nfile - 0.01;
    TLegend* leg = (ratio) ? (new TLegend(0.40, 0.86, 0.90, 0.90)) : (new TLegend(0.65, y2 - nfile * 0.04, 0.90, y2));
    leg->SetBorderSize(1);
    leg->SetFillColor(10);
    TH1D* hist0[nfile];
    for (int ifile = 0; ifile < nfile; ++ifile) {
      TDirectory* dir = (TDirectory*)file[ifile]->FindObjectAny(nameD);
      hist0[ifile] = static_cast<TH1D*>(dir->FindObjectAny(name));
      if (debug)
        std::cout << name << " read out at " << hist0[ifile] << " for " << tags[ifile] << std::endl;
    }
    char namec[160];
    if (!ratio) {
      sprintf(namec, "c_%s%s%s%s", geometry.c_str(), inType.c_str(), dirnm.c_str(), name);
      TCanvas* pad;
      int first(0);
      for (int ifile = 0; ifile < nfile; ++ifile) {
        TH1D* hist(hist0[ifile]);
        if (hist != nullptr) {
          if (rebin > 1)
            hist->Rebin(rebin);
          hist->SetTitle(title);
          hist->SetLineColor(first + 1);
          hist->SetLineStyle(first + 1);
          hist->GetYaxis()->SetTitleOffset(1.4);
          hist->GetXaxis()->SetRangeUser(0, xmax);
          hist->GetXaxis()->SetTitleSize(0.025);
          hist->GetXaxis()->SetTitleOffset(1.2);
          hist->SetMarkerStyle(first + 20);
          hist->SetMarkerColor(first + 1);
          hist->SetMarkerSize(0.7);
          if (first == 0) {
            pad = new TCanvas(namec, namec, 500, 500);
            pad->SetRightMargin(0.10);
            pad->SetTopMargin(0.10);
            pad->SetLogy();
            hist->Draw();
          } else {
            hist->Draw("sames");
          }
          leg->AddEntry(hist, tags[ifile].c_str(), "lp");
          pad->Update();
          ++first;
          TPaveStats* st = ((TPaveStats*)hist->GetListOfFunctions()->FindObject("stats"));
          if (st != NULL) {
            st->SetLineColor(first);
            st->SetTextColor(first);
            st->SetY1NDC(y1 - dy);
            st->SetY2NDC(y1);
            st->SetX1NDC(0.65);
            st->SetX2NDC(0.90);
            y1 -= dy;
          }
          pad->Modified();
          pad->Update();
          leg->Draw("same");
          pad->Update();
        }
        if (save) {
          sprintf(name, "%s.pdf", pad->GetName());
          pad->Print(name);
        }
      }
    } else if (nfile == 2) {
      sprintf(namec, "cR_%s%s%s%s", geometry.c_str(), inType.c_str(), dirnm.c_str(), name);
      TCanvas* pad = new TCanvas(namec, namec, 500, 500);
      int nbin = hist0[0]->GetNbinsX();
      int nbinR = nbin / rebin;
      double xlow = hist0[0]->GetXaxis()->GetBinLowEdge(1);
      double xhigh = xmax;
      TH1D* histr = new TH1D(name, hist0[0]->GetTitle(), nbinR, xlow, xhigh);
      histr->SetTitle(title);
      if (layer == 0)
        sprintf(name, "Number of hits (%s)", plotp[i].c_str());
      else
        sprintf(name, "Number of hits in Layer %d (%s)", layer, plotp[i].c_str());
      histr->GetXaxis()->SetTitle(name);
      sprintf(name, "Ratio (%s/%s)", tags[0].c_str(), tags[1].c_str());
      histr->GetYaxis()->SetTitle(name);
      histr->GetXaxis()->SetLabelOffset(0.005);
      histr->GetXaxis()->SetTitleOffset(1.30);
      histr->GetXaxis()->SetTitleSize(0.036);
      histr->GetYaxis()->SetLabelOffset(0.005);
      histr->GetYaxis()->SetTitleOffset(1.20);
      histr->GetYaxis()->SetTitleSize(0.036);
      histr->GetYaxis()->SetRangeUser(0.0, 5.0);
      histr->SetMarkerStyle(20);
      histr->SetMarkerColor(1);
      histr->SetMarkerSize(0.7);
      double sumNum(0), sumDen(0), maxDev(0);
      for (int j = 0; j < nbinR; ++j) {
        double tnum(0), tden(0), rnum(0), rden(0);
        for (int i = 0; i < rebin; ++i) {
          int ib = j * rebin + i + 1;
          tnum += hist0[0]->GetBinContent(ib);
          tden += hist0[1]->GetBinContent(ib);
          rnum += ((hist0[0]->GetBinError(ib)) * (hist0[0]->GetBinError(ib)));
          rden += ((hist0[1]->GetBinError(ib)) * (hist0[1]->GetBinError(ib)));
        }
        if (tden > 0 && tnum > 0) {
          double rat = tnum / tden;
          double err = rat * sqrt((rnum / (tnum * tnum)) + (rden / (tden * tden)));
          histr->SetBinContent(j + 1, rat);
          histr->SetBinError(j + 1, err);
          double temp1 = (rat > 1.0) ? 1.0 / rat : rat;
          double temp2 = (rat > 1.0) ? err / (rat * rat) : err;
          sumNum += (fabs(1 - temp1) / (temp2 * temp2));
          sumDen += (1.0 / (temp2 * temp2));
          if (fabs(1 - temp1) > maxDev)
            maxDev = fabs(1 - temp1);
        }
      }
      histr->Draw();
      if (layer == 0)
        sprintf(name, "%s %s", dirnm.c_str(), plotp[i].c_str());
      else
        sprintf(name, "%s (Layer %d) %s", dirnm.c_str(), layer, plotp[i].c_str());
      leg->AddEntry(histr, name, "lp");
      leg->Draw("same");
      pad->Update();
      TLine* line = new TLine(xlow, 1.0, xhigh, 1.0);
      line->SetLineColor(2);
      line->SetLineWidth(2);
      line->SetLineStyle(2);
      line->Draw("same");
      pad->Modified();
      pad->Update();
      sumNum = (sumDen > 0) ? (sumNum / sumDen) : 0;
      sumDen = (sumDen > 0) ? 1.0 / sqrt(sumDen) : 0;
      if (sumNum == 0)
        sumDen = 0;
      std::cout << tags[0] << " vs " << tags[1] << " " << hist0[0]->GetXaxis()->GetTitle() << " Mean deviation "
                << sumNum << " +- " << sumDen << " maximum " << maxDev << std::endl;
      if (save) {
        sprintf(name, "%s.pdf", pad->GetName());
        pad->Print(name);
      }
    }
  }
}

void makeDigiStudyPlots(std::string tag = "HC", int todomin = 0, int todomax = 11, bool save = false) {
  const int nFiles = 3, ndetEC = 2, ndetHC = 3;
  std::string files[nFiles] = {
      "FullSimSignalwithFullSimPU", "FullSimSignalwithFastSimPU", "FastSimSignalwithFastSimPU"};
  std::string tags[nFiles] = {"Full+Full", "Ful+Fast", "Fast+Fast"};
  int color[nFiles] = {1, 2, 4};
  int lstyl[nFiles] = {1, 2, 4};
  std::string detsHC[ndetHC] = {"HB", "HE", "HF"};
  std::string detsEC[ndetEC] = {"Barrel", "Endcap"};
  std::string pretagEC = "EcalDigiTask";
  std::string pretagHC = "HcalDigiTask";
  const int plots = 16;
  std::string nameEC[plots] = {"ADC pulse 01 Gain 12",
                               "ADC pulse 02 Gain 12",
                               "ADC pulse 07 Gain 12",
                               "ADC pulse 04 Gain 12",
                               "ADC pulse 05 Gain 12",
                               "ADC pulse 06 Gain 12",
                               "ADC pulse 07 Gain 12",
                               "ADC pulse 08 Gain 12",
                               "analog pulse 01",
                               "analog pulse 02",
                               "analog pulse 03",
                               "analog pulse 04",
                               "analog pulse 05",
                               "analog pulse 06",
                               "analog pulse 07",
                               "analog pulse 08"};
  std::string nameHC[plots] = {"Ndigis",
                               "depths",
                               "post_SOI_frac",
                               "signal_amplitude",
                               "ADCO_adc_depth1",
                               "ADCO_adc_depth2",
                               "ADCO_adc_depth3",
                               "ADCO_adc_depth4",
                               "signal_amplitude_depth1",
                               "signal_amplitude_depth2",
                               "signal_amplitude_depth3",
                               "signal_amplitude_depth4",
                               "all_amplitudes_vs_bin_1D_depth1",
                               "all_amplitudes_vs_bin_1D_depth2",
                               "all_amplitudes_vs_bin_1D_depth3",
                               "all_amplitudes_vs_bin_1D_depth4"};
  int rebinEC[plots] = {10, 10, 10, 10, 10, 10, 10, 10, 2, 2, 2, 2, 2, 2, 2, 2};
  int rebinHC[plots] = {10, 1, 10, 10, 1, 1, 1, 1, 10, 10, 10, 10, 1, 1, 1, 1};
  double xlowEC[plots] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  double xlowHC[plots] = {3000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  double xhighEC[plots] = {1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 50, 50, 50, 50, 50, 50, 50, 50};
  double xhighHC[plots] = {5000, 10, 2, 8000, 20, 20, 20, 20, 8000, 8000, 8000, 8000, 10, 10, 10, 10};
  bool debug(true);

  std::string dirnm = (tag == "EC") ? "ecalDigiStudy" : "hcalDigiStudy";
  gStyle->SetCanvasBorderMode(0);
  gStyle->SetCanvasColor(kWhite);
  gStyle->SetPadColor(kWhite);
  gStyle->SetFillColor(kWhite);
  gStyle->SetOptStat(1110);
  if ((todomin < 0) || (todomin >= plots))
    todomin = 0;
  if (todomax < todomin) {
    todomax = todomin;
  } else if (todomax >= plots) {
    todomax = plots - 1;
  }
  TFile* file[nFiles];
  char fname[100];
  int ok(0);
  for (int i = 0; i < nFiles; ++i) {
    sprintf(fname, "%s%s.root", tag.c_str(), files[i].c_str());
    file[i] = new TFile(fname);
    if (file[i]) {
      ++ok;
      std::cout << fname << " opened successfully" << std::endl;
    } else {
      std::cout << fname << " cannot be found" << std::endl;
    }
  }
  if (ok == nFiles) {
    int ndet = (tag == "EC") ? ndetEC : ndetHC;
    char name[100], namec[100];
    for (int i1 = todomin; i1 <= todomax; ++i1) {
      for (int i2 = 0; i2 < ndet; ++i2) {
        if (tag == "EC") {
          sprintf(name, "%s %s %s", pretagEC.c_str(), detsEC[i2].c_str(), nameEC[i1].c_str());
          sprintf(namec, "c_%s_%s_%s", pretagEC.c_str(), detsEC[i2].c_str(), nameEC[i1].c_str());
        } else {
          sprintf(name, "%s_%s_%s", pretagHC.c_str(), nameHC[i1].c_str(), detsHC[i2].c_str());
          sprintf(namec, "c_%s_%s_%s", pretagHC.c_str(), nameHC[i1].c_str(), detsHC[i2].c_str());
        }
        TH1D* hist0[nFiles];
        for (int i3 = 0; i3 < nFiles; ++i3) {
          TDirectory* dir = (TDirectory*)file[i3]->FindObjectAny(dirnm.c_str());
          hist0[i3] = static_cast<TH1D*>(dir->FindObjectAny(name));
          if (debug)
            std::cout << name << " read out at " << hist0[i3] << " for " << tags[i3] << std::endl;
        }
        TCanvas* pad = new TCanvas(namec, namec, 500, 500);
        int first(0);
        double y1(0.90), dy(0.12);
        double y2 = y1 - dy * nFiles - 0.01;
        TLegend* leg = new TLegend(0.65, y2 - nFiles * 0.04, 0.90, y2);
        for (int i3 = 0; i3 < nFiles; ++i3) {
          TH1D* hist(hist0[i3]);
          if (debug)
            std::cout << i3 << " Tag " << tags[i3] << " hiist " << hist << std::endl;
          if (hist != nullptr) {
            hist->SetLineColor(color[i3]);
            hist->SetLineStyle(lstyl[i3]);
            hist->GetYaxis()->SetTitleOffset(1.4);
            std::string title = hist->GetTitle();
            hist->GetXaxis()->SetTitle(title.c_str());
            hist->SetTitle("");
            if (tag == "EC") {
              if (rebinEC[i1] > 1)
                hist->Rebin(rebinEC[i1]);
              hist->GetXaxis()->SetRangeUser(xlowEC[i1], xhighEC[i1]);
            } else {
              if (rebinHC[i1] > 1)
                hist->Rebin(rebinHC[i1]);
              hist->GetXaxis()->SetRangeUser(xlowHC[i1], xhighHC[i1]);
            }
            if (first == 0) {
              pad = new TCanvas(namec, namec, 500, 500);
              pad->SetRightMargin(0.10);
              pad->SetTopMargin(0.10);
              /*
	      if (tag == "EC") 
		pad->SetLogy();
	      */
              hist->Draw();
            } else {
              hist->Draw("sames");
            }
            leg->AddEntry(hist, tags[i3].c_str(), "lp");
            pad->Update();
            ++first;
            TPaveStats* st = ((TPaveStats*)hist->GetListOfFunctions()->FindObject("stats"));
            if (st != NULL) {
              st->SetLineColor(color[i3]);
              st->SetTextColor(color[i3]);
              st->SetY1NDC(y1 - dy);
              st->SetY2NDC(y1);
              st->SetX1NDC(0.65);
              st->SetX2NDC(0.90);
              y1 -= dy;
            }
            pad->Modified();
            pad->Update();
            leg->Draw("same");
            pad->Update();
          }
        }
        if (save) {
          sprintf(name, "%s.pdf", pad->GetName());
          pad->Print(name);
        }
      }
    }
  }
}
