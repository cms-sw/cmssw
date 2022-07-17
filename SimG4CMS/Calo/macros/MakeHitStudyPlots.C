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

void makeHitStudyPlots(std::string file1 = "analRun3Old.root",
                       std::string tag1 = "Old",
                       std::string file2 = "analRun3New.root",
                       std::string tag2 = "New",
                       int todo = 0,
                       std::string dirnm = "CaloSimHitStudy",
                       bool save = false) {
  std::string names[18] = {"Edep",
                           "EdepEM",
                           "EdepHad",
                           "EdepTk",
                           "Etot",
                           "EtotG",
                           "Hit",
                           "HitHigh",
                           "HitLow",
                           "HitMu",
                           "HitTk",
                           "Time",
                           "TimeAll",
                           "TimeTk",
                           "EneInc",
                           "EtaInc",
                           "PhiInc",
                           "PtInc"};
  int numb[18] = {9, 9, 9, 16, 9, 9, 9, 1, 1, 1, 16, 9, 9, 16, 1, 1, 1, 1};
  int rebin[18] = {10, 10, 10, 1, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 1, 1, 1, 1};

  gStyle->SetCanvasBorderMode(0);
  gStyle->SetCanvasColor(kWhite);
  gStyle->SetPadColor(kWhite);
  gStyle->SetFillColor(kWhite);
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
  std::cout << "Use " << nfile << " files from " << file1 << " and " << file2 << std::endl;
  for (int i1 = 0; i1 < numb[todo]; ++i1) {
    int first(0);
    double y1(0.90), dy(0.12);
    double y2 = y1 - dy * nfile - 0.01;
    TLegend* leg = new TLegend(0.74, y2 - nfile * 0.04, 0.89, y2);
    leg->SetBorderSize(1);
    leg->SetFillColor(10);
    TCanvas* pad;
    for (int ifile = 0; ifile < nfile; ++ifile) {
      TDirectory* dir = (TDirectory*)file[ifile]->FindObjectAny(dirnm.c_str());
      char name[100], namec[100];
      if (numb[todo] == 1) {
        sprintf(name, "%s", names[todo].c_str());
        sprintf(namec, "%s%s", names[todo].c_str(), tag.c_str());
      } else {
        sprintf(name, "%s%d", names[todo].c_str(), i1);
        sprintf(namec, "%s%d%s", names[todo].c_str(), i1, tag.c_str());
      }
      TH1D* hist = static_cast<TH1D*>(dir->FindObjectAny(name));
      std::cout << name << " read out at " << hist << " for " << tags[ifile] << std::endl;
      if (hist != nullptr) {
        hist->SetLineColor(first + 1);
        hist->SetLineStyle(first + 1);
        hist->GetYaxis()->SetTitleOffset(1.4);
        if (rebin[todo] > 1)
          hist->Rebin(rebin[todo]);
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
          sprintf(name, "c_%s.pdf", pad->GetName());
          pad->Print(name);
        }
      }
    }
  }
}
