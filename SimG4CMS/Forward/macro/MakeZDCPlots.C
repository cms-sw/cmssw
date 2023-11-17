//////////////////////////////////////////////////////////////////////////////
//
// Usage:
// .L MakeZDCPlots.C+g
//
//   To make plot of various quantities which are created by ZDCSimHitStudy
//   from upto three different settings
//
//     makeHitStudyPlots(file1, tag1, file2, tag2, file3, tag3, todomin,
//                       todomax, gtitle, ratio, save, dirnm)
//
//   where (for makeHitStudyPlots)
//     fileN    std::string   Name of the Nth ROOT file
//                            The fist must exist; blank means beng ignored
//     tag1     std::string   Tag for the Nth file
//     todomin  int           Minimum type # of histograms to be plotted [0]
//     todomax  int           Maximum type # of histograms to be plotted [5]
//     gtitle   std::string   Overall Titile [""]
//     ratio    bool          if the ratio to be plotted [true]
//                            (with respect to the first file)
//     save     bool          If the canvas is to be saved [false]
//     dirnm    std::string   Name of the directory [zdcSimHitStudy]
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

void makeHitStudyPlots(std::string file1 = "Forward.root",
                       std::string tag1 = "New",
                       std::string file2 = "Standard.root",
                       std::string tag2 = "Old",
                       std::string file3 = "",
                       std::string tag3 = "",
                       std::string gtitle = "",
                       int todomin = 0,
                       int todomax = 5,
                       bool ratio = true,
                       bool save = false,
                       std::string dirnm = "zdcSimHitStudy") {
  const int plots = 6;
  std::string names[plots] = {"ETot", "ETotT", "Edep", "Hits", "Indx", "Time"};
  int logy[plots] = {1, 1, 1, 0, 1, 1};
  int rebin[plots] = {10, 10, 10, 2, 1, 10};
  int xmax[plots] = {5, 5, 3, 5, 2, 2};
  int colors[5] = {1, 2, 4, 6, 46};
  int marker[5] = {20, 21, 22, 23, 24};
  int styles[5] = {1, 2, 3, 4, 5};
  bool debug(false);

  gStyle->SetCanvasBorderMode(0);
  gStyle->SetCanvasColor(kWhite);
  gStyle->SetPadColor(kWhite);
  gStyle->SetFillColor(kWhite);
  if (ratio) {
    gStyle->SetOptStat(10);
    gStyle->SetOptFit(1);
  } else {
    gStyle->SetOptStat(1110);
  }
  TFile* file[3];
  int nfile(0);
  std::string tag(""), tags[3];
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
  if (file3 != "") {
    file[nfile] = new TFile(file3.c_str());
    if (file[nfile]) {
      tags[nfile] = tag3;
      ++nfile;
      tag += tag3;
    }
  }
  if ((todomin < 0) || (todomin >= plots))
    todomin = 0;
  if (todomax < todomin) {
    todomax = todomin;
  } else if (todomax >= plots) {
    todomax = plots - 1;
  }
  std::cout << "Use " << nfile << " files from " << file1 << "," << file2 << " and " << file3 << " and look for "
            << todomin << ":" << todomax << std::endl;
  for (int todo = todomin; todo <= todomax; ++todo) {
    double y1(0.90), dy(0.12);
    double y2 = y1 - dy * nfile - 0.01;
    double y3 = y1 - 0.04 * (nfile - 1);
    TLegend* leg = (ratio) ? (new TLegend(0.10, y3, 0.65, y1)) : (new TLegend(0.65, y2 - nfile * 0.04, 0.90, y2));
    leg->SetBorderSize(1);
    leg->SetFillColor(10);
    TH1D* hist0[nfile];
    for (int i1 = 0; i1 < nfile; ++i1) {
      TDirectory* dir = static_cast<TDirectory*>(file[i1]->FindObjectAny(dirnm.c_str()));
      hist0[i1] = static_cast<TH1D*>(dir->FindObjectAny(names[todo].c_str()));
      if (debug)
        std::cout << "Reads histogram " << hist0[i1]->GetName() << " for " << hist0[i1]->GetTitle() << " from "
                  << file[i1] << std::endl;
    }
    std::string title = hist0[0]->GetTitle();
    if (debug)
      std::cout << "Histogram title " << title << std::endl;
    int istart = (ratio) ? 1 : 0;
    char name[100], namec[100];
    std::vector<TH1D*> hist1;
    for (int i1 = istart; i1 < nfile; ++i1) {
      if (ratio) {
        int nbin = hist0[0]->GetNbinsX();
        int nbinR = nbin / rebin[todo];
        double xlow = hist0[0]->GetXaxis()->GetBinLowEdge(1);
        double xhigh = hist0[0]->GetXaxis()->GetBinLowEdge(nbin) + hist0[0]->GetXaxis()->GetBinWidth(nbin);
        sprintf(name, "%s%sVs%sRatio", names[todo].c_str(), tags[0].c_str(), tags[i1].c_str());
        TH1D* histr = new TH1D(name, hist0[0]->GetTitle(), nbinR, xlow, xhigh);
        histr->SetTitle(gtitle.c_str());
        histr->GetXaxis()->SetTitle(title.c_str());
        histr->GetYaxis()->SetTitle(name);
        histr->GetXaxis()->SetLabelOffset(0.005);
        histr->GetXaxis()->SetTitleOffset(1.00);
        histr->GetYaxis()->SetLabelOffset(0.005);
        histr->GetYaxis()->SetTitleOffset(1.20);
        histr->GetYaxis()->SetRangeUser(0.0, xmax[todo]);
        histr->SetLineColor(colors[i1 - 1]);
        histr->SetLineStyle(styles[i1 - 1]);
        histr->SetMarkerStyle(marker[i1 - 1]);
        histr->SetMarkerColor(colors[i1 - 1]);
        double sumNum(0), sumDen(0), maxDev(0), xlow1(-1), xhigh1(xlow);
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
            if (xlow1 < 0)
              xlow1 = hist0[0]->GetXaxis()->GetBinLowEdge(j * rebin[todo] + 1);
            xhigh1 = hist0[0]->GetXaxis()->GetBinLowEdge((j + 1) * rebin[todo]) +
                     hist0[0]->GetXaxis()->GetBinWidth((j + 1) * rebin[todo]);
            double rat = tnum / tden;
            double err = rat * sqrt((rnum / (tnum * tnum)) + (rden / (tden * tden)));
            if (debug)
              std::cout << "Bin " << j << " Ratio " << rat << " +- " << err << std::endl;
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
        histr->Fit("pol0", "+QRWLS", "", xlow1, xhigh1);
        sumNum = (sumDen > 0) ? (sumNum / sumDen) : 0;
        sumDen = (sumDen > 0) ? 1.0 / sqrt(sumDen) : 0;
        sprintf(name, "%sVs%sRatio", tags[0].c_str(), tags[i1].c_str());
        if (sumNum == 0)
          sumDen = 0;
        sprintf(name, "%s vs %s (%6.3f +- %6.3f)", tags[0].c_str(), tags[i1].c_str(), sumNum, sumDen);
        hist1.push_back(histr);
        leg->AddEntry(histr, name, "lp");
      } else {
        hist1.push_back(hist0[i1]);
        hist1.back()->SetTitle(gtitle.c_str());
        hist1.back()->GetXaxis()->SetTitle(title.c_str());
        hist1.back()->GetYaxis()->SetTitle("Events");
        hist1.back()->GetXaxis()->SetLabelOffset(0.005);
        hist1.back()->GetXaxis()->SetTitleOffset(1.00);
        hist1.back()->GetYaxis()->SetLabelOffset(0.005);
        hist1.back()->GetYaxis()->SetTitleOffset(1.40);
        hist1.back()->SetLineColor(colors[i1]);
        hist1.back()->SetLineStyle(styles[i1]);
        leg->AddEntry(hist1.back(), tags[i1].c_str(), "lp");
      }
    }
    sprintf(namec, "c_%s", hist1[0]->GetName());
    if (debug)
      std::cout << namec << " Canvas made for " << hist1[0]->GetName() << " with " << hist1.size() << " plots"
                << std::endl;
    TCanvas* pad = new TCanvas(namec, namec, 500, 500);
    pad->SetRightMargin(0.10);
    pad->SetTopMargin(0.10);
    if ((!ratio) && (logy[todo] > 0))
      pad->SetLogy(1);
    else
      pad->SetLogy(0);
    for (unsigned int i1 = 0; i1 < hist1.size(); ++i1) {
      if ((!ratio) && (rebin[todo] > 1))
        hist1[i1]->Rebin(rebin[todo]);
      if (i1 == 0) {
        hist1[i1]->Draw();
      } else {
        hist1[i1]->Draw("sames");
      }
      if (debug)
        std::cout << "Drawing histograms for " << hist1[i1]->GetName() << ":" << i1 << " in canvas " << pad->GetName()
                  << std::endl;
      pad->Update();
      double dy0 = (ratio) ? 0.75 * dy : dy;
      TPaveStats* st = ((TPaveStats*)hist1[i1]->GetListOfFunctions()->FindObject("stats"));
      if (st != NULL) {
        st->SetFillColor(kWhite);
        st->SetLineColor(colors[i1]);
        st->SetTextColor(colors[i1]);
        st->SetY1NDC(y1 - dy0);
        st->SetY2NDC(y1);
        st->SetX1NDC(0.65);
        st->SetX2NDC(0.90);
        y1 -= dy0;
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
}
