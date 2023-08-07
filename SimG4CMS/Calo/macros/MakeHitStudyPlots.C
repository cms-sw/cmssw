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
//
//   where (common to both macros)
//     ratio    bool          if the ratio to be plotted [true]
//                            (works when both files are active)
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
  std::string names[plots] = {"Etot",       "Hit",  "EtotG",  "Time", "EdepTk",
			      "Edep",   "HitHigh", "HitLow", "HitMu",  "HitTk",
			      "TimeAll", "TimeTk",    "eta",   "phi", "EdepEM",
                              "EdepHad", "EneInc", "EtaInc", "PhiInc", "PtInc"};
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
