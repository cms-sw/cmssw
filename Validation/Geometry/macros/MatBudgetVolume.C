///////////////////////////////////////////////////////////////////////////////
//
// etaPhiPlot(fileName, plot, drawLeg, ifEta, maxEta, tag)
//      Make the plots of integrated interaction/radiation/step lengths as a
//      function of eta or phi
// fileName (TString)     Name of the input ROOT file ("matbdg_run3.root")
// plot     (std::string) Type of plot: intl/radl/step ("intl")
// drawLeg  (bool)        Flag to show the legend or not (true)
// ifEta    (bool)        Draw as a function of eta or phi (true)
// maxEta   (double)      Maximum value of x-axis: if -1 use default (5.2)
// tag      (string)      Tag to be added to the name of the canvas ("Run3")
//
// etaPhi2DPlot(fileName, plot, drawLeg, maxEta, tag)
//      Make the 2-D plots as a function of eta and phi with same parameter
//      meanings as those of *etaPhiPlot*
//
// etaPhiPlotComp(fileName1, fileName2, plot, ifEta, tag, txt, debug)
//      Compares material budget plots from 2 different files
//
// etaPhiPlotComp4(filePreFix, tag, plot, ifEta, debug)
//      Compares material budget plots from 4 different files:
//      dddXML, dd4hepXML, dddDB, dd4hepDB
//
// filePreFix (std::string) Prefix to all 4 file names which will be followed
//                          by one of dddXML/dd4hepXML/dddDB/dd4hepDB strings
//                          and finally with *tag* and ".root"
// txt        (std::string) Part of the y-title coming after #frac for the plot
//                          ("{DDD}/{DD4Hep}")
//
///////////////////////////////////////////////////////////////////////////////

// include files
#include <TCanvas.h>
#include <TChain.h>
#include <TFile.h>
#include <TGraphErrors.h>
#include <TH1D.h>
#include <TH2D.h>
#include <THStack.h>
#include <TLegend.h>
#include <TPaveStats.h>
#include <TPaveText.h>
#include <TProfile.h>
#include <TProfile2D.h>
#include <TROOT.h>
#include <TStyle.h>
#include <vector>
#include <string>
#include <iomanip>
#include <iostream>
#include <fstream>

void etaPhiPlot(TString fileName = "matbdg_run3.root",
                std::string plot = "intl",
                bool drawLeg = true,
                bool ifEta = true,
                double maxEta = 5.2,
                std::string tag = "Run3");
void etaPhi2DPlot(TString fileName = "matbdg_run3.root",
                  std::string plot = "intl",
                  bool drawLeg = true,
                  double maxEta = 5.2,
                  std::string tag = "Run3");
void etaPhiPlotComp(TString fileName1 = "matbdg_run3.root",
                    TString fileName2 = "matbdg_run3_dd4hep.root",
                    std::string plot = "intl",
                    bool ifEta = true,
                    std::string tag = "Run3",
                    std::string txt = "{DDD}/{DD4Hep}",
                    bool debug = false);
void etaPhiPlotComp4(std::string filePreFix = "files/matbdgRun3",
                     std::string tag = "pre6",
                     std::string plot = "radl",
                     bool ifEta = true,
                     bool debug = false);
void setStyle();

const int nlay = 13;
const int ngrp = 9;
int nlayers[ngrp] = {5, 1, 1, 1, 1, 1, 1, 1, 1};
int nflayer[ngrp] = {0, 5, 6, 7, 8, 9, 10, 11, 12};
int colorLay[nlay] = {2, 2, 2, 2, 2, 3, 5, 4, 8, 6, 3, 7, 1};
int styleLay[nlay] = {20, 20, 20, 20, 20, 21, 22, 23, 24, 25, 26, 27, 30};
int legends[nlay] = {1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1};
std::string title[nlay] = {
    "Beam Pipe", "", "", "", "", "Tracker", "ECAL", "HCAL", "HGCAL", "HF", "Magnet", "MUON", "Forward"};
std::string names[nlay] = {
    "BEAM", "BEAM1", "BEAM2", "BEAM3", "BEAM4", "Tracker", "ECAL", "HCal", "CALOEC", "VCAL", "MGNT", "MUON", "OQUA"};

void etaPhiPlot(TString fileName, std::string plot, bool drawLeg, bool ifEta, double maxEta, std::string tag) {
  TFile *hcalFile = new TFile(fileName);
  hcalFile->cd("materialBudgetVolumeAnalysis");
  setStyle();
  gStyle->SetOptTitle(0);

  std::string xtit = "#eta";
  std::string ztit = "Eta";
  std::string ytit = "none";
  double xh = 0.90;
  if (plot == "radl") {
    ytit = "Material Budget (X_{0})";
  } else if (plot == "step") {
    ytit = "Material Budget (Step Length)";
    xh = 0.95;
  } else {
    plot = "intl";
    ytit = "Material Budget (#lambda)";
  }
  if (!ifEta) {
    xtit = "#phi";
    ztit = "Phi";
  }

  TLegend *leg = new TLegend(0.84, 0.69, 0.99, 0.99);
  leg->SetBorderSize(1);
  leg->SetFillColor(10);
  leg->SetMargin(0.25);
  leg->SetTextSize(0.028);

  char hname[20], titlex[50];
  sprintf(hname, "%s%s%s", plot.c_str(), ztit.c_str(), names[0].c_str());
  TProfile *prof;
  gDirectory->GetObject(hname, prof);
  int nb = prof->GetNbinsX();
  double xlow = prof->GetBinLowEdge(1);
  double xhigh = prof->GetBinLowEdge(nb) + prof->GetBinWidth(nb);
  THStack *hs = new THStack("hs", "");
  for (int ii = 0; ii < nlay; ++ii) {
    sprintf(hname, "%s%s%s", plot.c_str(), ztit.c_str(), names[ii].c_str());
    TProfile *prof;
    gDirectory->GetObject(hname, prof);
    sprintf(hname, "%s%s%sH", plot.c_str(), ztit.c_str(), names[ii].c_str());
    TH1D *hist = new TH1D(hname, "", nb, xlow, xhigh);
    for (int k = 1; k <= nb; ++k) {
      double cont = prof->GetBinContent(k);
      hist->SetBinContent(k, cont);
    }
    hist->SetLineColor(colorLay[ii]);
    hist->SetFillColor(colorLay[ii]);
    if (ifEta && maxEta > 0)
      hist->GetXaxis()->SetRangeUser(-maxEta, maxEta);
    hs->Add(hist);
    if (legends[ii] > 0) {
      sprintf(titlex, "%s", title[ii].c_str());
      leg->AddEntry(hist, titlex, "lf");
    }
  }

  std::string cname = "c_" + plot + ztit + tag;
  TCanvas *cc1 = new TCanvas(cname.c_str(), cname.c_str(), 700, 600);
  if (xh > 0.91) {
    cc1->SetLeftMargin(0.15);
    cc1->SetRightMargin(0.05);
  } else {
    cc1->SetLeftMargin(0.10);
    cc1->SetRightMargin(0.10);
  }

  hs->Draw("");
  if (drawLeg)
    leg->Draw("sames");
  hs->GetXaxis()->SetTitle(xtit.c_str());
  hs->GetYaxis()->SetTitle(ytit.c_str());
  if (xh > 0.91) {
    hs->GetYaxis()->SetTitleOffset(2.0);
  } else {
    hs->GetYaxis()->SetTitleOffset(1.2);
  }
  cc1->Modified();
}

void etaPhi2DPlot(TString fileName, std::string plot, bool drawLeg, double maxEta, std::string tag) {
  TFile *hcalFile = new TFile(fileName);
  hcalFile->cd("materialBudgetVolumeAnalysis");
  setStyle();
  gStyle->SetOptTitle(1);

  std::string xtit = "#eta";
  std::string ytit = "#phi";
  std::string ztit = "none";
  if (plot == "radl") {
    ztit = "Material Budget (X_{0})";
  } else if (plot == "step") {
    ztit = "Material Budget (Step Length)";
  } else {
    plot = "intl";
    ztit = "Material Budget (#lambda)";
  }

  TLegend *leg = new TLegend(0.84, 0.69, 0.99, 0.99);
  leg->SetBorderSize(1);
  leg->SetFillColor(10);
  leg->SetMargin(0.25);
  leg->SetTextSize(0.028);

  char hname[20], titlex[50];
  sprintf(hname, "%sEtaPhi%s", plot.c_str(), names[0].c_str());
  TProfile2D *prof;
  gDirectory->GetObject(hname, prof);
  int nx = prof->GetXaxis()->GetNbins();
  double xlow = prof->GetXaxis()->GetBinLowEdge(1);
  double xhigh = prof->GetXaxis()->GetBinUpEdge(nx);
  int ny = prof->GetYaxis()->GetNbins();
  double ylow = prof->GetYaxis()->GetBinLowEdge(1);
  double yhigh = prof->GetYaxis()->GetBinUpEdge(ny);
  std::cout << hname << " X " << nx << ":" << xlow << ":" << xhigh << " Y " << ny << ":" << ylow << ":" << yhigh
            << std::endl;
  THStack *hs = new THStack("hs", ztit.c_str());
  for (int ii = 0; ii < nlay; ++ii) {
    sprintf(hname, "%sEtaPhi%s", plot.c_str(), names[ii].c_str());
    gDirectory->GetObject(hname, prof);
    sprintf(hname, "%sEtaPhi%sH", plot.c_str(), names[ii].c_str());
    TH2D *hist = new TH2D(hname, "", nx, xlow, xhigh, ny, ylow, yhigh);
    for (int kx = 1; kx <= nx; ++kx) {
      for (int ky = 1; ky <= ny; ++ky) {
        double cont = prof->GetBinContent(kx, ky);
        hist->SetBinContent(kx, ky, cont);
      }
    }
    hist->SetLineColor(colorLay[ii]);
    hist->SetFillColor(colorLay[ii]);
    if (maxEta > 0)
      hist->GetXaxis()->SetRangeUser(-maxEta, maxEta);
    hs->Add(hist);
    if (legends[ii] > 0) {
      sprintf(titlex, "%s", title[ii].c_str());
      leg->AddEntry(hist, titlex, "lf");
    }
  }

  std::string cname = "c_" + plot + "EtaPhi" + tag;
  TCanvas *cc1 = new TCanvas(cname.c_str(), cname.c_str(), 700, 600);
  cc1->SetLeftMargin(0.10);
  cc1->SetRightMargin(0.10);

  hs->Draw("");
  if (drawLeg)
    leg->Draw("sames");
  hs->GetXaxis()->SetTitle(xtit.c_str());
  hs->GetYaxis()->SetTitle(ytit.c_str());
  hs->GetYaxis()->SetTitleOffset(1.2);
  cc1->Modified();
}

void etaPhiPlotComp(
    TString fileName1, TString fileName2, std::string plot, bool ifEta, std::string tag, std::string txt, bool debug) {
  setStyle();
  gStyle->SetOptTitle(0);
  TFile *file1 = new TFile(fileName1);
  TFile *file2 = new TFile(fileName2);
  if ((file1 != nullptr) && (file2 != nullptr)) {
    TDirectory *dir1 = (TDirectory *)(file1->FindObjectAny("materialBudgetVolumeAnalysis"));
    TDirectory *dir2 = (TDirectory *)(file2->FindObjectAny("materialBudgetVolumeAnalysis"));
    TLegend *leg = new TLegend(0.84, 0.69, 0.99, 0.99);
    leg->SetBorderSize(1);
    leg->SetFillColor(10);
    leg->SetMargin(0.25);
    leg->SetTextSize(0.028);

    std::string xtit = "#eta";
    std::string ztit = "Eta";
    char ytit[40];
    if (plot == "radl") {
      sprintf(ytit, "#frac%s for MB (X_{0})", txt.c_str());
    } else if (plot == "step") {
      sprintf(ytit, "#frac%s for MB (Step Length)", txt.c_str());
    } else {
      plot = "intl";
      sprintf(ytit, "#frac%s for MB (#lambda)", txt.c_str());
    }
    if (!ifEta) {
      xtit = "#phi";
      ztit = "Phi";
    }

    std::vector<TGraphErrors *> graphs;
    std::vector<int> index;
    char hname[20], titlex[50];
    int nb(0);
    double xlow(0), xhigh(0);
    for (int i = 0; i < ngrp; ++i) {
      std::vector<double> xx0, yy1, yy2, dy1, dy2;
      for (int j = 0; j < nlayers[i]; ++j) {
        int ii = nflayer[i] + j;
        sprintf(hname, "%s%s%s", plot.c_str(), ztit.c_str(), names[ii].c_str());
        TProfile *prof1, *prof2;
        dir1->GetObject(hname, prof1);
        dir2->GetObject(hname, prof2);
        if ((prof1 != nullptr) && (prof2 != nullptr)) {
          int nb = prof1->GetNbinsX();
          for (int k = 1; k <= nb; ++k) {
            yy1.push_back(prof1->GetBinContent(k));
            yy2.push_back(prof2->GetBinContent(k));
            dy1.push_back(prof1->GetBinError(k));
            dy2.push_back(prof2->GetBinError(k));
            xx0.push_back(prof1->GetBinLowEdge(k) + prof1->GetBinWidth(k));
          }
        }
      }
      std::vector<double> xx, yy, dx, dy;
      int ii = nflayer[i];
      double sumNum(0), sumDen(0);
      for (unsigned int k = 0; k < xx0.size(); ++k) {
        if ((yy1[k] > 0) && (yy2[k] > 0)) {
          double rat = yy1[k] / yy2[k];
          double drt = rat * sqrt((dy1[k] / yy1[k]) * (dy1[k] / yy1[k]) + (dy2[k] / yy2[k]) * (dy2[k] / yy2[k]));
          xx.push_back(xx0[k]);
          dx.push_back(0);
          yy.push_back(rat);
          dy.push_back(drt);
          if (debug) {
            std::cout << title[ii] << " [" << (xx.size() - 1) << "] " << xx0[k] << " Ratio " << rat << " +- " << drt
                      << std::endl;
          }
          double temp1 = (rat > 1.0) ? 1.0 / rat : rat;
          double temp2 = (rat > 1.0) ? drt / (rat * rat) : drt;
          sumNum += (fabs(1.0 - temp1) / (temp2 * temp2));
          sumDen += (1.0 / (temp2 * temp2));
        }
      }
      sumNum = (sumDen > 0) ? (sumNum / sumDen) : 0;
      sumDen = (sumDen > 0) ? 1.0 / sqrt(sumDen) : 0;
      std::cout << "Mean deviation for " << title[ii] << "  " << sumNum << " +- " << sumDen << std::endl;
      if (xx.size() > 0) {
        TGraphErrors *graph = new TGraphErrors(xx.size(), &xx[0], &yy[0], &dx[0], &dy[0]);
        graph->SetLineColor(colorLay[ii]);
        graph->SetFillColor(colorLay[ii]);
        graph->SetMarkerStyle(styleLay[ii]);
        sprintf(titlex, "%s", title[ii].c_str());
        leg->AddEntry(graph, titlex, "lep");
        graphs.push_back(graph);
        if (nb == 0) {
          sprintf(hname, "%s%s%s", plot.c_str(), ztit.c_str(), names[0].c_str());
          TProfile *prof;
          dir1->GetObject(hname, prof);
          nb = prof->GetNbinsX();
          xlow = prof->GetBinLowEdge(1);
          xhigh = prof->GetBinLowEdge(nb) + prof->GetBinWidth(nb);
        }
      }
    }
    if (graphs.size() > 0) {
      std::string cname = "c_" + plot + ztit + "Ratio" + tag;
      TCanvas *cc1 = new TCanvas(cname.c_str(), cname.c_str(), 700, 600);
      cc1->SetLeftMargin(0.10);
      cc1->SetRightMargin(0.10);
      TH1F *vFrame = cc1->DrawFrame(xlow, 0.5, xhigh, 1.5);
      vFrame->GetXaxis()->SetRangeUser(xlow, xhigh);
      vFrame->GetXaxis()->SetLabelSize(0.03);
      vFrame->GetXaxis()->SetTitleSize(0.035);
      vFrame->GetXaxis()->SetTitleOffset(0.4);
      vFrame->GetXaxis()->SetTitle(xtit.c_str());
      vFrame->GetYaxis()->SetRangeUser(0.9, 1.1);
      vFrame->GetYaxis()->SetLabelSize(0.03);
      vFrame->GetYaxis()->SetTitleSize(0.035);
      vFrame->GetYaxis()->SetTitleOffset(1.3);
      vFrame->GetYaxis()->SetTitle(ytit);
      for (unsigned int i = 0; i < graphs.size(); ++i)
        graphs[i]->Draw("P");
      leg->Draw("sames");
      cc1->Modified();
    }
  }
}

void etaPhiPlotComp4(std::string filePreFix, std::string tag, std::string plot, bool ifEta, bool debug) {
  setStyle();
  gStyle->SetOptTitle(0);
  const int files = 4;
  std::string nametype[files] = {"dddXML", "dd4hepXML", "dddDB", "dd4hepDB"};
  int colortype[files] = {1, 2, 4, 6};
  TFile *file[files];
  char fname[40];
  bool ok(true);
  for (int k1 = 0; k1 < files; ++k1) {
    sprintf(fname, "%s%s%s.root", filePreFix.c_str(), nametype[k1].c_str(), tag.c_str());
    file[k1] = new TFile(fname);
    if (file[k1] == nullptr)
      ok = false;
  }
  if (ok) {
    TDirectory *dir[files];
    for (int k1 = 0; k1 < files; ++k1) {
      dir[k1] = (TDirectory *)(file[k1]->FindObjectAny("materialBudgetVolumeAnalysis"));
    }
    TLegend *leg = new TLegend(0.84, 0.69, 0.99, 0.99);
    leg->SetBorderSize(1);
    leg->SetFillColor(10);
    leg->SetMargin(0.25);
    leg->SetTextSize(0.028);

    std::string xtit = "#eta";
    std::string ztit = "Eta";
    std::string ytit = "none";
    if (plot == "radl") {
      ytit = "#frac{Sample}{dddXML} for MB (X_{0})";
    } else if (plot == "step") {
      ytit = "#frac{Sample}{dddXML} for MB (Step Length)";
    } else {
      plot = "intl";
      ytit = "#frac{Sample}{dddXML} for MB (#lambda)";
    }
    if (!ifEta) {
      xtit = "#phi";
      ztit = "Phi";
    }

    std::vector<TGraphErrors *> graphs;
    std::vector<int> index;
    char hname[20], titlex[50];
    int nb(0);
    double xlow(0), xhigh(0);
    for (int i = 0; i < ngrp; ++i) {
      std::vector<double> xx0, yy0[files], dy0[files];
      for (int j = 0; j < nlayers[i]; ++j) {
        int ii = nflayer[i] + j;
        sprintf(hname, "%s%s%s", plot.c_str(), ztit.c_str(), names[ii].c_str());
        TProfile *prof[files];
        bool okf(true);
        for (int k1 = 0; k1 < files; ++k1) {
          dir[k1]->GetObject(hname, prof[k1]);
          if (dir[k1] == nullptr)
            okf = false;
        }
        if (okf) {
          int nb = prof[0]->GetNbinsX();
          for (int k = 1; k <= nb; ++k) {
            xx0.push_back(prof[0]->GetBinLowEdge(k) + prof[0]->GetBinWidth(k));
            for (int k1 = 0; k1 < files; ++k1) {
              yy0[k1].push_back(prof[k1]->GetBinContent(k));
              dy0[k1].push_back(prof[k1]->GetBinError(k));
            }
          }
        }
      }
      int ii = nflayer[i];
      for (int k1 = 1; k1 < files; ++k1) {
        std::vector<double> xx, yy, dx, dy;
        double sumNum(0), sumDen(0), maxtmp(0), maxDev(0), dmaxDev(0);
        for (unsigned int k = 0; k < xx0.size(); ++k) {
          if ((yy0[0][k] > 0) && (yy0[k1][k] > 0)) {
            double rat = yy0[k1][k] / yy0[0][k];
            double drt = rat * sqrt((dy0[k1][k] / yy0[k1][k]) * (dy0[k1][k] / yy0[k1][k]) +
                                    (dy0[0][k] / yy0[0][k]) * (dy0[0][k] / yy0[0][k]));
            xx.push_back(xx0[k]);
            dx.push_back(0);
            yy.push_back(rat);
            dy.push_back(drt);
            if (debug) {
              std::cout << nametype[k1] << ":" << title[ii] << " [" << (xx.size() - 1) << "] " << xx0[k] << " Ratio "
                        << rat << " +- " << drt << std::endl;
            }
            double temp1 = (rat > 1.0) ? 1.0 / rat : rat;
            double temp2 = (rat > 1.0) ? drt / (rat * rat) : drt;
            double temp0 = (fabs(1.0 - temp1) / (temp2 * temp2));
            sumNum += temp0;
            sumDen += (1.0 / (temp2 * temp2));
            if (temp0 >= maxtmp) {
              maxtmp = temp0;
              maxDev = fabs(1.0 - temp1);
              dmaxDev = temp2;
            }
          }
        }
        sumNum = (sumDen > 0) ? (sumNum / sumDen) : 0;
        sumDen = (sumDen > 0) ? 1.0 / sqrt(sumDen) : 0;
        std::cout << title[ii] << " in " << nametype[k1] << " Mean " << sumNum << " +- " << sumDen << " Max " << maxDev
                  << " +- " << dmaxDev << std::endl;
        if (xx.size() > 0) {
          TGraphErrors *graph = new TGraphErrors(xx.size(), &xx[0], &yy[0], &dx[0], &dy[0]);
          graph->SetLineColor(colortype[k1]);
          graph->SetFillColor(colorLay[ii]);
          graph->SetMarkerStyle(styleLay[ii]);
          if (k1 == 1) {
            sprintf(titlex, "%s", title[ii].c_str());
            leg->AddEntry(graph, titlex, "lep");
          }
          graphs.push_back(graph);
          if (nb == 0) {
            sprintf(hname, "%s%s%s", plot.c_str(), ztit.c_str(), names[0].c_str());
            TProfile *prof;
            dir[0]->GetObject(hname, prof);
            nb = prof->GetNbinsX();
            xlow = prof->GetBinLowEdge(1);
            xhigh = prof->GetBinLowEdge(nb) + prof->GetBinWidth(nb);
          }
        }
      }
    }
    if (graphs.size() > 0) {
      std::string cname = "c_" + plot + ztit + "Ratio" + tag;
      TCanvas *cc1 = new TCanvas(cname.c_str(), cname.c_str(), 700, 600);
      cc1->SetLeftMargin(0.10);
      cc1->SetRightMargin(0.10);
      TH1F *vFrame = cc1->DrawFrame(xlow, 0.5, xhigh, 1.5);
      vFrame->GetXaxis()->SetRangeUser(xlow, xhigh);
      vFrame->GetXaxis()->SetLabelSize(0.03);
      vFrame->GetXaxis()->SetTitleSize(0.035);
      vFrame->GetXaxis()->SetTitleOffset(0.4);
      vFrame->GetXaxis()->SetTitle(xtit.c_str());
      vFrame->GetYaxis()->SetRangeUser(0.9, 1.1);
      vFrame->GetYaxis()->SetLabelSize(0.03);
      vFrame->GetYaxis()->SetTitleSize(0.035);
      vFrame->GetYaxis()->SetTitleOffset(1.3);
      vFrame->GetYaxis()->SetTitle(ytit.c_str());
      for (unsigned int i = 0; i < graphs.size(); ++i)
        graphs[i]->Draw("P");
      leg->Draw("sames");
      cc1->Modified();
      double ymx = 0.68;
      for (int k1 = 1; k1 < files; ++k1) {
        TPaveText *txt1 = new TPaveText(0.84, ymx - 0.03, 0.99, ymx, "blNDC");
        txt1->SetFillColor(0);
        sprintf(fname, "%s", nametype[k1].c_str());
        txt1->AddText(fname);
        ((TText *)txt1->GetListOfLines()->Last())->SetTextColor(colortype[k1]);
        txt1->Draw();
        ymx -= 0.03;
      }
    }
  }
}

void setStyle() {
  gStyle->SetCanvasBorderMode(0);
  gStyle->SetCanvasColor(kWhite);
  gStyle->SetPadColor(kWhite);
  gStyle->SetFrameBorderMode(0);
  gStyle->SetFrameBorderSize(1);
  gStyle->SetFrameFillColor(0);
  gStyle->SetFrameFillStyle(0);
  gStyle->SetFrameLineColor(1);
  gStyle->SetFrameLineStyle(1);
  gStyle->SetFrameLineWidth(1);
  gStyle->SetOptStat(0);
  gStyle->SetLegendBorderSize(1);
  gStyle->SetTitleColor(0);
  gStyle->SetTitleOffset(2.5, "Y");
}
