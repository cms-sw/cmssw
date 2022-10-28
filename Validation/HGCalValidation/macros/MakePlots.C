//###########################################################################
//
//   void noseRecHitPlots(fname, dirnm, tag, save)
//        Plots histograms created by recHitStudy for the Nose detector
//   void hgcalStudyPlots(fname, type, tag, dtype, save, debug)
//        Plots histograms created by SimHit/Digi/RecHit Study for HGCal
//   void hgcalGeomCheckPlots(fname, tag, dtype, save, debug)
//        Plots histograms created by HGCGeometryCheck
//   void hgcalBHValidPlots(fname, tag, dtype, save, debug)
//        Plots histograms created by HGCalBHValidation
//   void hgcalSiliconAnalysisPlots(fname, tag, dtype, save, debug)
//        Plots histograms created by HGCalSiliconValidation
//   void hgcalMissedHitPlots(fname, tag, dtype, save, debug)
//        Plots histograms created by HGCalMissedRecHit
//   void ehcalPlots(fname, dirnm, tag, dets, save, debug)
//        Plots histograms created by EcalSimHitStudy
//
//   where
//     fanme     (std::string)  Input ROOT file name
//                              ("hfnRecHitD31tt.root" for noseRecHitPlots,
//                               "roots/hgcSimHitD83tt.root" hgcalStudyPlots,
//                               "roots/hgcGeomCheckD83.root" hgcalGeomCheck,
//                               "roots/hgcBHValidD83.root" hgcalBHValid,
//                               "roots/hgcSilValidD86.root" hgcalSiliconAnalysis,
//                               "missedRecHitD88.root" hgcalMissedHitPlots,
//                               "ecalHitdd4hep.root" ehcalPlot)
//     dirnm     (std::string)  Directory name
//                              ("hfnoseRecHitStudy" for noseRecHitPlots
//                               "hgcGeomCheck" for hgcalGeomCheck,
//                               "hgcalBHAnalysis" for hgcalBHValid,
//                               "hgcMissingRecHit" for hgcalMissedHitPlots,
//                               "EcalSimHitStudy" for ehcalPlot)
//     type      (int)          Type: 0 SimHit; 1 Digi; 2 RecHit (0)
//     tag       (std::string)  Name of the tag for the canvas name
//                              ("HFNose" for noseRecHitPlots,
//                               "SimHitD83" for hgcalStudyPlots,
//                               "GeomChkD83" for hgcalGeomCheck,
//                               "BHValidD83" for hgcalBHValid,
//                               "SilValidD86" for hgcalSiliconAnalysisPlots,
//                               "MissedHitD88" for hgcalMissedHitPlots,
//                               "DD4Hep" for ehcalPlots)
//     dtype     (std::string)  Data type added for canvas name
//                              ("ttbar D83" for hgcalStudyPlots,
//                               "#mu D83" for hgcalGeomCheck, hgcalBHValid',
//                               "ttbar D86" for hgcalSiliconAnalysisPlots,
//                               "#mu D88" for hgcalMissedHitPlots)
//     save      (bool)         Flag to save the canvas (false)
//     debug     (bool)         Debug flag (false)
//
//   In addition there is a tree analysis class to study the ROOT tree
//   created by HGCHitAnalysis code. To invoke this, one needs to do
//   the following steps:
//   .L MakePlots.C+g
//   hgchits c1(fname);
//   c1.Loop();
//   c1.saveHistos(outFile);
//
//###########################################################################
//
#include <TROOT.h>
#include <TChain.h>
#include <TFile.h>
#include <TH1D.h>
#include <TH2D.h>
#include <TProfile.h>
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

void noseRecHitPlots(std::string fname = "hfnRecHitD31tt.root",
                     std::string dirnm = "hfnoseRecHitStudy",
                     std::string tag = "HFNose",
                     bool save = false) {
  int nums[2] = {5, 2};
  int layers(8);
  std::string name1[5] = {
      "Energy_Layer", "HitOccupancy_Minus_layer", "HitOccupaancy_Plus_layer", "EtaPhi_Minus_Layer", "EtaPhi_Plus_Layer"};
  std::string name2[2] = {"EtaPhi", "RZ"};
  std::string detName = "HGCalHFNoseSensitive";
  int type1[5] = {1, 1, 1, 2, 2};
  int rebin[5] = {10, 1, 1, 1, 1};
  int type2[2] = {2, 2};
  std::string xtitl1[5] = {"Energy (GeV)", "Hits", "Hits", "#eta", "#eta"};
  std::string ytitl1[5] = {" ", " ", " ", "#phi", "#phi"};
  std::string xtitl2[2] = {"#eta", "z (cm)"};
  std::string ytitl2[2] = {"#phi", "R (cm)"};

  gStyle->SetCanvasBorderMode(0);
  gStyle->SetCanvasColor(kWhite);
  gStyle->SetPadColor(kWhite);
  gStyle->SetFillColor(kWhite);
  gStyle->SetOptStat(111110);
  TFile *file = new TFile(fname.c_str());
  if (file) {
    TDirectory *dir = (TDirectory *)file->FindObjectAny(dirnm.c_str());
    char name[100], title[100];
    for (int i1 = 0; i1 < 2; ++i1) {
      int kk = (i1 == 0) ? layers : 1;
      for (int i2 = 0; i2 < nums[i1]; ++i2) {
        for (int k = 0; k < kk; ++k) {
          int type(0);
          if (i1 == 0) {
            sprintf(name, "%s_%d", name1[i2].c_str(), k + 1);
            sprintf(title, "%s (Layer %d)", tag.c_str(), k + 1);
            type = type1[i2];
          } else {
            sprintf(name, "%s_%s", name2[i2].c_str(), detName.c_str());
            sprintf(title, "Simulation of %s", tag.c_str());
            type = type2[i2];
          }
          TH1D *hist1(nullptr);
          TH2D *hist2(nullptr);
          if (type == 1)
            hist1 = (TH1D *)dir->FindObjectAny(name);
          else
            hist2 = (TH2D *)dir->FindObjectAny(name);
          if ((hist1 != nullptr) || (hist2 != nullptr)) {
            TCanvas *pad = new TCanvas(name, name, 500, 500);
            pad->SetRightMargin(0.10);
            pad->SetTopMargin(0.10);
            if (type == 1) {
              if (i1 == 0) {
                hist1->GetYaxis()->SetTitle(ytitl1[i2].c_str());
                hist1->GetXaxis()->SetTitle(xtitl1[i2].c_str());
                hist1->Rebin(rebin[i2]);
              } else {
                hist1->GetYaxis()->SetTitle(ytitl2[i2].c_str());
                hist1->GetXaxis()->SetTitle(xtitl2[i2].c_str());
              }
              hist1->SetTitle(title);
              hist1->GetYaxis()->SetTitleOffset(1.2);
              pad->SetLogy();
              hist1->Draw();
            } else {
              if (i1 == 0) {
                hist2->GetYaxis()->SetTitle(ytitl1[i2].c_str());
                hist2->GetXaxis()->SetTitle(xtitl1[i2].c_str());
              } else {
                hist2->GetYaxis()->SetTitle(ytitl2[i2].c_str());
                hist2->GetXaxis()->SetTitle(xtitl2[i2].c_str());
              }
              hist2->GetYaxis()->SetTitleOffset(1.2);
              hist2->SetMarkerStyle(20);
              hist2->SetMarkerSize(0.1);
              hist2->SetTitle(title);
              hist2->Draw();
            }
            pad->Update();
            TPaveStats *st1 = ((hist1 != nullptr) ? ((TPaveStats *)hist1->GetListOfFunctions()->FindObject("stats"))
                                                  : ((TPaveStats *)hist2->GetListOfFunctions()->FindObject("stats")));
            if (st1 != NULL) {
              st1->SetY1NDC(0.70);
              st1->SetY2NDC(0.90);
              st1->SetX1NDC(0.65);
              st1->SetX2NDC(0.90);
            }
            pad->Modified();
            pad->Update();
            if (save) {
              sprintf(name, "c_%s%s.jpg", tag.c_str(), pad->GetName());
              pad->Print(name);
            }
          }
        }
      }
    }
  }
}

void hgcalStudyPlots(std::string fname = "roots/hgcSimHitD83tt.root",
                     int type = 0,
                     std::string tag = "SimHitD83",
                     std::string dtype = "ttbar D83",
                     bool save = false,
                     bool debug = false) {
  int ndir[3] = {1, 3, 3};
  std::string dirnm[3][3] = {{"hgcalSimHitStudy", "", ""},
                             {"hgcalDigiStudyEE", "hgcalDigiStudyHEF", "hgcalDigiStudyHEB"},
                             {"hgcalRecHitStudyEE", "hgcalRecHitStudyFH", "hgcalRecHitStudyBH"}};
  std::string name0[4] = {"HGCal EE", "HGCal HE Silicon", "HGCal HE Scintillator", "HGCal"};
  int nname[3] = {4, 1, 1};
  std::string name1[4] = {
      "HGCalEESensitive", "HGCalHESiliconSensitive", "HGCalHEScintillatorSensitive", "AllDetectors"};
  int nhist[3] = {4, 2, 2};
  int nhtype[3][4] = {{1, 1, 2, 2}, {1, 2, 0, 0}, {2, 2, 0, 0}};
  int yax[3][4] = {{1, 1, 0, 0}, {1, 0, 0, 0}, {0, 0, 0, 0}};
  std::string name2[3][4] = {
      {"E_", "T_", "RZ_", "EtaPhi_"}, {"Charge_", "RZ_", " ", " "}, {"RZ_", "EtaPhi_", " ", " "}};
  std::string xtitl[3][4] = {
      {"Energy (GeV)", "Time (ns)", "Z (mm)", "#phi"}, {"Charge", "Z (cm)", " ", " "}, {"Z (cm)", "#phi", " ", " "}};
  std::string ytitl[3][4] = {
      {"Hits", "Hits", "R (mm)", "#eta"}, {"Hits", "R (cm)", " ", " "}, {"R (cm)", "#eta", " ", " "}};
  double xmax[3][4] = {{0.2, 20.0, -1, -1}, {25.0, -1, -1, -1}, {-1, -1, -1, -1}};

  gStyle->SetCanvasBorderMode(0);
  gStyle->SetCanvasColor(kWhite);
  gStyle->SetPadColor(kWhite);
  gStyle->SetFillColor(kWhite);
  gStyle->SetOptStat(111110);
  if (debug)
    std::cout << "File " << fname << " Type " << type << ":" << name0[type] << " Tags " << tag << " : " << dtype
              << " Save " << save << "\n";
  TFile *file = new TFile(fname.c_str());
  if (file) {
    for (int it = 0; it < ndir[type]; ++it) {
      char dirx[100];
      sprintf(dirx, "%s", dirnm[type][it].c_str());
      TDirectory *dir = (TDirectory *)file->FindObjectAny(dirx);
      if (debug)
        std::cout << "Directory " << dirx << " : " << dir << std::endl;
      if (dir) {
        for (int in = 0; in < nname[type]; ++in) {
          for (int ih = 0; ih < nhist[type]; ++ih) {
            char hname[100];
            if (type == 0) {
              sprintf(hname, "%s%s", name2[type][ih].c_str(), name1[in].c_str());
            } else {
              sprintf(hname, "%s%s", name2[type][ih].c_str(), name1[it].c_str());
            }
            TH1D *hist1(nullptr);
            TH2D *hist2(nullptr);
            if (nhtype[type][ih] == 1)
              hist1 = (TH1D *)dir->FindObjectAny(hname);
            else
              hist2 = (TH2D *)dir->FindObjectAny(hname);
            if (debug)
              std::cout << "Hist " << hname << " : " << hist1 << " : " << hist2 << " Xtitle " << xtitl[type][ih]
                        << " Ytitle " << ytitl[type][ih] << " xmax " << xmax[type][ih] << " Type " << nhtype[type][ih]
                        << " yscale " << yax[type][ih] << std::endl;
            if ((hist1 != nullptr) || (hist2 != nullptr)) {
              char name[100], title[100];
              sprintf(name, "%s%s", hname, tag.c_str());
              TCanvas *pad = new TCanvas(name, name, 500, 500);
              pad->SetRightMargin(0.10);
              pad->SetTopMargin(0.10);
              if (type == 0)
                sprintf(title, "%s (%s)", name0[in].c_str(), dtype.c_str());
              else
                sprintf(title, "%s (%s)", name0[it].c_str(), dtype.c_str());
              if (debug)
                std::cout << "Pad " << name << " : " << pad << "\n";
              if (nhtype[type][ih] == 1) {
                hist1->GetYaxis()->SetTitle(ytitl[type][ih].c_str());
                hist1->GetXaxis()->SetTitle(xtitl[type][ih].c_str());
                if (xmax[type][ih] > 0)
                  hist1->GetXaxis()->SetRangeUser(0, xmax[type][ih]);
                if (yax[type][ih] > 0)
                  pad->SetLogy();
                hist1->SetTitle(title);
                hist1->GetYaxis()->SetTitleOffset(1.2);
                hist1->Draw();
              } else {
                hist2->GetYaxis()->SetTitle(ytitl[type][ih].c_str());
                hist2->GetXaxis()->SetTitle(xtitl[type][ih].c_str());
                hist2->GetYaxis()->SetTitleOffset(1.2);
                hist2->SetMarkerStyle(20);
                hist2->SetMarkerSize(0.1);
                hist2->SetTitle(title);
                hist2->Draw();
              }
              pad->Update();
              TPaveStats *st1 = ((hist1 != nullptr) ? ((TPaveStats *)hist1->GetListOfFunctions()->FindObject("stats"))
                                                    : ((TPaveStats *)hist2->GetListOfFunctions()->FindObject("stats")));
              if (st1 != NULL) {
                st1->SetY1NDC(0.70);
                st1->SetY2NDC(0.90);
                st1->SetX1NDC(0.65);
                st1->SetX2NDC(0.90);
              }
              pad->Modified();
              pad->Update();
              if (save) {
                sprintf(name, "c_%s.jpg", pad->GetName());
                pad->Print(name);
              }
            }
          }
        }
      }
    }
  }
}

void hgcalGeomCheckPlots(std::string fname = "roots/hgcGeomCheckD83.root",
                         std::string tag = "GeomChkD83",
                         std::string dtype = "#mu D83",
                         bool statbox = true,
                         bool save = false,
                         bool debug = false) {
  std::string dirnm = "hgcGeomCheck";
  std::string name0[3] = {"HGCal EE", "HGCal HE Silicon", "HGCal HE Scintillator"};
  int nhist = 3;
  int nhtype = 3;
  std::string name2[9] = {"heerVsLayer",
                          "heezVsLayer",
                          "heedzVsZ",
                          "hefrVsLayer",
                          "hefzVsLayer",
                          "hefdzVsZ",
                          "hebrVsLayer",
                          "hebzVsLayer",
                          "hebdzVsZ"};
  std::string xtitl[9] = {"Layer", "Layer", "Z (cm)", "Layer", "Layer", "Z (cm)", "Layer", "Layer", "Z (cm)"};
  std::string ytitl[9] = {
      "R (cm)", "Z (cm)", "#Delta Z (cm)", "R (cm)", "Z (cm)", "#Delta Z (cm)", "R (cm)", "Z (cm)", "#Delta Z (cm)"};

  gStyle->SetCanvasBorderMode(0);
  gStyle->SetCanvasColor(kWhite);
  gStyle->SetPadColor(kWhite);
  gStyle->SetFillColor(kWhite);
  if (statbox)
    gStyle->SetOptStat(111110);
  else
    gStyle->SetOptStat(0);
  if (debug)
    std::cout << "File " << fname << " Tags " << tag << " : " << dtype << " Save " << save << "\n";
  TFile *file = new TFile(fname.c_str());
  if (file) {
    char dirx[100];
    sprintf(dirx, "%s", dirnm.c_str());
    TDirectory *dir = (TDirectory *)file->FindObjectAny(dirx);
    if (debug)
      std::cout << "Directory " << dirx << " : " << dir << std::endl;
    if (dir) {
      for (int in = 0; in < nhtype; ++in) {
        for (int ih = 0; ih < nhist; ++ih) {
          char hname[100];
          int inh = in * nhist + ih;
          sprintf(hname, "%s", name2[inh].c_str());
          TH2D *hist = (TH2D *)dir->FindObjectAny(hname);
          if (debug)
            std::cout << "Hist " << hname << " : " << hist << " Xtitle " << xtitl[inh] << " Ytitle " << ytitl[inh]
                      << std::endl;
          if (hist != nullptr) {
            char name[100], title[100];
            sprintf(name, "%s%s", hname, tag.c_str());
            TCanvas *pad = new TCanvas(name, name, 500, 500);
            pad->SetRightMargin(0.10);
            pad->SetTopMargin(0.10);
            sprintf(title, "%s (%s)", name0[in].c_str(), dtype.c_str());
            if (debug)
              std::cout << "Pad " << name << " : " << pad << "\n";
            hist->GetYaxis()->SetTitle(ytitl[inh].c_str());
            hist->GetXaxis()->SetTitle(xtitl[inh].c_str());
            hist->SetTitle(title);
            hist->GetYaxis()->SetTitleOffset(1.2);
            hist->Draw();
            pad->Update();
            TPaveStats *st1 = (TPaveStats *)hist->GetListOfFunctions()->FindObject("stats");
            if (st1 != NULL) {
              st1->SetY1NDC(0.70);
              st1->SetY2NDC(0.90);
              st1->SetX1NDC(0.65);
              st1->SetX2NDC(0.90);
            }
            pad->Modified();
            pad->Update();
            if (save) {
              sprintf(name, "c_%s.jpg", pad->GetName());
              pad->Print(name);
            }
          }
        }
      }
    }
  }
}

void hgcalBHValidPlots(std::string fname = "roots/hgcBHValidD83.root",
                       std::string tag = "BHValidD83",
                       std::string dtype = "#mu D83",
                       bool save = false,
                       bool debug = false) {
  std::string dirnm = "hgcalBHAnalysis";
  std::string name0 = "HGCal HE Scintillator";
  int nhist = 9;
  std::string name2[9] = {"SimHitEn1",
                          "SimHitEn2",
                          "SimHitLong",
                          "SimHitOccup",
                          "SimHitOccu3",
                          "SimHitTime",
                          "DigiLong",
                          "DigiOccup",
                          "DigiOccu3"};
  std::string xtitl[9] = {"SimHit Energy (GeV)",
                          "SimHit Energy (GeV)",
                          "SimHit Layer #",
                          "SimHit i#eta",
                          "SimHit i#eta",
                          "Digi Layer #",
                          "Digi i#eta",
                          "Digi i#eta"};
  std::string ytitl[9] = {"Hits", "Hits", "Energy Sum (GeV)", "i#phi", "Layer #", "Digi Sum", "i#phi", "Layer #"};
  double xmax[9] = {0.05, 0.20, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0};
  int ihty[9] = {1, 1, 1, 2, 2, 1, 1, 2, 2};
  int iaxty[9] = {1, 1, 0, 0, 0, 1, 0, 0, 0};
  int ibin[10] = {0, 0, 0, 0, 0, 10, 0, 0, 0};
  gStyle->SetCanvasBorderMode(0);
  gStyle->SetCanvasColor(kWhite);
  gStyle->SetPadColor(kWhite);
  gStyle->SetFillColor(kWhite);
  gStyle->SetOptStat(111110);
  if (debug)
    std::cout << "File " << fname << " Tags " << tag << " : " << dtype << " Save " << save << "\n";
  TFile *file = new TFile(fname.c_str());
  if (file) {
    char dirx[100];
    sprintf(dirx, "%s", dirnm.c_str());
    TDirectory *dir = (TDirectory *)file->FindObjectAny(dirx);
    if (debug)
      std::cout << "Directory " << dirx << " : " << dir << std::endl;
    if (dir) {
      for (int ih = 0; ih < nhist; ++ih) {
        char hname[100];
        sprintf(hname, "%s", name2[ih].c_str());
        TH1D *hist1(nullptr);
        TH2D *hist2(nullptr);
        if (ihty[ih] <= 1)
          hist1 = (TH1D *)dir->FindObjectAny(hname);
        else
          hist2 = (TH2D *)dir->FindObjectAny(hname);
        if (debug)
          std::cout << "Hist " << hname << " : " << hist1 << ":" << hist2 << " Xtitle " << xtitl[ih] << " Ytitle "
                    << ytitl[ih] << " xmax " << xmax[ih] << " iaxty " << iaxty[ih] << " ibin " << ibin[ih] << std::endl;
        if ((hist1 != nullptr) || (hist2 != nullptr)) {
          char name[100], title[100];
          sprintf(name, "%s%s", hname, tag.c_str());
          TCanvas *pad = new TCanvas(name, name, 500, 500);
          pad->SetRightMargin(0.10);
          pad->SetTopMargin(0.10);
          sprintf(title, "%s (%s)", name0.c_str(), dtype.c_str());
          if (debug)
            std::cout << "Pad " << name << " : " << pad << "\n";
          if (hist1 != nullptr) {
            hist1->GetYaxis()->SetTitle(ytitl[ih].c_str());
            hist1->GetXaxis()->SetTitle(xtitl[ih].c_str());
            hist1->SetTitle(title);
            if (xmax[ih] > 0)
              hist1->GetXaxis()->SetRangeUser(0, xmax[ih]);
            if (iaxty[ih] > 0)
              pad->SetLogy();
            if (ibin[ih] > 0)
              hist1->Rebin(ibin[ih]);
            hist1->GetYaxis()->SetTitleOffset(1.2);
            hist1->Draw();
          } else {
            hist2->GetYaxis()->SetTitle(ytitl[ih].c_str());
            hist2->GetXaxis()->SetTitle(xtitl[ih].c_str());
            hist2->SetTitle(title);
            hist2->GetYaxis()->SetTitleOffset(1.2);
            hist2->Draw();
          }
          pad->Update();
          TPaveStats *st1 = ((hist1 != nullptr) ? ((TPaveStats *)hist1->GetListOfFunctions()->FindObject("stats"))
                                                : ((TPaveStats *)hist2->GetListOfFunctions()->FindObject("stats")));
          if (st1 != NULL) {
            st1->SetY1NDC(0.70);
            st1->SetY2NDC(0.90);
            st1->SetX1NDC(0.65);
            st1->SetX2NDC(0.90);
          }
          pad->Modified();
          pad->Update();
          if (save) {
            sprintf(name, "c_%s.jpg", pad->GetName());
            pad->Print(name);
          }
        }
      }
    }
  }
}

void hgcalSiliconAnalysisPlots(std::string fname = "roots/hgcSilValidD86.root",
                               std::string tag = "SilValidD86",
                               std::string dtype = "ttbar D86",
                               bool save = false,
                               bool debug = false) {
  std::string dirnm[2] = {"hgcalSiliconAnalysisEE", "hgcalSiliconAnalysisHEF"};
  std::string name0[2] = {"HGCal EE", "HGCal HE Silicon"};
  int nhist = 9;
  std::string name2[9] = {"SimHitEn1",
                          "SimHitEn2",
                          "SimHitLong",
                          "SimHitOccup",
                          "SimHitOccu2",
                          "SimHitTime",
                          "DigiLong",
                          "DigiOccup",
                          "DigiOccu2"};
  std::string xtitl[9] = {"SimHit Energy (GeV)",
                          "SimHit Energy (GeV)",
                          "SimHit Layer #",
                          "SimHit i#eta",
                          "SimHit i#eta",
                          "Digi Layer #",
                          "Digi i#eta",
                          "Digi i#eta"};
  std::string ytitl[9] = {"Hits", "Hits", "Energy Sum (GeV)", "i#phi", "Layer #", "Digi Sum", "i#phi", "Layer #"};
  double xmax[9] = {0.05, 0.20, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0};
  int ihty[9] = {1, 1, 1, 2, 2, 1, 1, 2, 2};
  int iaxty[9] = {1, 1, 0, 0, 0, 1, 0, 0, 0};
  int ibin[10] = {0, 0, 0, 0, 0, 10, 0, 0, 0};
  gStyle->SetCanvasBorderMode(0);
  gStyle->SetCanvasColor(kWhite);
  gStyle->SetPadColor(kWhite);
  gStyle->SetFillColor(kWhite);
  gStyle->SetOptStat(111110);
  if (debug)
    std::cout << "File " << fname << " Tags " << tag << " : " << dtype << " Save " << save << "\n";
  TFile *file = new TFile(fname.c_str());
  if (file) {
    for (int idir = 0; idir < 2; ++idir) {
      char dirx[100];
      sprintf(dirx, "%s", dirnm[idir].c_str());
      TDirectory *dir = (TDirectory *)file->FindObjectAny(dirx);
      if (debug)
        std::cout << "Directory " << dirx << " : " << dir << std::endl;
      if (dir) {
        for (int ih = 0; ih < nhist; ++ih) {
          char hname[100];
          sprintf(hname, "%s", name2[ih].c_str());
          TH1D *hist1(nullptr);
          TH2D *hist2(nullptr);
          if (ihty[ih] <= 1)
            hist1 = (TH1D *)dir->FindObjectAny(hname);
          else
            hist2 = (TH2D *)dir->FindObjectAny(hname);
          if (debug)
            std::cout << "Hist " << hname << " : " << hist1 << ":" << hist2 << " Xtitle " << xtitl[ih] << " Ytitle "
                      << ytitl[ih] << " xmax " << xmax[ih] << " iaxty " << iaxty[ih] << " ibin " << ibin[ih]
                      << std::endl;
          if ((hist1 != nullptr) || (hist2 != nullptr)) {
            char name[100], title[100];
            sprintf(name, "%s%s", hname, tag.c_str());
            TCanvas *pad = new TCanvas(name, name, 500, 500);
            pad->SetRightMargin(0.10);
            pad->SetTopMargin(0.10);
            sprintf(title, "%s (%s)", name0[idir].c_str(), dtype.c_str());
            if (debug)
              std::cout << "Pad " << name << " : " << pad << "\n";
            if (hist1 != nullptr) {
              hist1->GetYaxis()->SetTitle(ytitl[ih].c_str());
              hist1->GetXaxis()->SetTitle(xtitl[ih].c_str());
              hist1->SetTitle(title);
              if (xmax[ih] > 0)
                hist1->GetXaxis()->SetRangeUser(0, xmax[ih]);
              if (iaxty[ih] > 0)
                pad->SetLogy();
              if (ibin[ih] > 0)
                hist1->Rebin(ibin[ih]);
              hist1->GetYaxis()->SetTitleOffset(1.2);
              hist1->Draw();
            } else {
              hist2->GetYaxis()->SetTitle(ytitl[ih].c_str());
              hist2->GetXaxis()->SetTitle(xtitl[ih].c_str());
              hist2->SetTitle(title);
              hist2->GetYaxis()->SetTitleOffset(1.2);
              hist2->Draw();
            }
            pad->Update();
            TPaveStats *st1 = ((hist1 != nullptr) ? ((TPaveStats *)hist1->GetListOfFunctions()->FindObject("stats"))
                                                  : ((TPaveStats *)hist2->GetListOfFunctions()->FindObject("stats")));
            if (st1 != NULL) {
              st1->SetY1NDC(0.70);
              st1->SetY2NDC(0.90);
              st1->SetX1NDC(0.65);
              st1->SetX2NDC(0.90);
            }
            pad->Modified();
            pad->Update();
            if (save) {
              sprintf(name, "c_%s.jpg", pad->GetName());
              pad->Print(name);
            }
          }
        }
      }
    }
  }
}

void hgcalMissedHitPlots(std::string fname = "missedRecHitD88.root",
                         std::string tag = "MissedHitD88",
                         std::string dtype = "#mu D88",
                         bool save = false,
                         bool debug = false) {
  std::string dirnm = "hgcMissingRecHit";
  std::string name0 = "HGCal HE Scintillator";
  const int nhist = 24;
  std::string name2[nhist] = {"GoodDEHGCalEESensitive",
                              "MissDEHGCalEESensitive",
                              "GoodDEHGCalHESiliconSensitive",
                              "MissDEHGCalHESiliconSensitive",
                              "GoodDEHGCalHEScintillatorSensitive",
                              "MissDEHGCalHEScintillatorSensitive",
                              "GoodREHGCalEESensitive",
                              "MissREHGCalEESensitive",
                              "GoodREHGCalHESiliconSensitive",
                              "MissREHGCalHESiliconSensitive",
                              "GoodREHGCalHEScintillatorSensitive",
                              "MissREHGCalHEScintillatorSensitive",
                              "GoodDTHGCalEESensitive",
                              "MissDTHGCalEESensitive",
                              "GoodDTHGCalHESiliconSensitive",
                              "MissDTHGCalHESiliconSensitive",
                              "GoodDTHGCalHEScintillatorSensitive",
                              "MissDTHGCalHEScintillatorSensitive",
                              "GoodRTHGCalEESensitive",
                              "MissRTHGCalEESensitive",
                              "GoodRTHGCalHESiliconSensitive",
                              "MissRTHGCalHESiliconSensitive",
                              "GoodRTHGCalHEScintillatorSensitive",
                              "MissRTHGCalHEScintillatorSensitive"};
  std::string xtitl[nhist] = {"SimHit Energy (GeV)",
                              "SimHit Energy (GeV)",
                              "SimHit Energy (GeV)",
                              "SimHit Energy (GeV)",
                              "SimHit Energy (GeV)",
                              "SimHit Energy (GeV)",
                              "SimHit Energy (GeV)",
                              "SimHit Energy (GeV)",
                              "SimHit Energy (GeV)",
                              "SimHit Energy (GeV)",
                              "SimHit Energy (GeV)",
                              "SimHit Energy (GeV)",
                              "|#eta|",
                              "|#eta|",
                              "|#eta|",
                              "|#eta|",
                              "|#eta|",
                              "|#eta|",
                              "|#eta|",
                              "|#eta|",
                              "|#eta|",
                              "|#eta|",
                              "|#eta|",
                              "|#eta|"};
  std::string ytitl[nhist] = {"Hits", "Hits", "Hits", "Hits", "Hits", "Hits", "Hits", "Hits",
                              "Hits", "Hits", "Hits", "Hits", "Hits", "Hits", "Hits", "Hits",
                              "Hits", "Hits", "Hits", "Hits", "Hits", "Hits", "Hits", "Hits"};
  int ibin[nhist] = {10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10};
  gStyle->SetCanvasBorderMode(0);
  gStyle->SetCanvasColor(kWhite);
  gStyle->SetPadColor(kWhite);
  gStyle->SetFillColor(kWhite);
  gStyle->SetOptStat(111110);
  if (debug)
    std::cout << "File " << fname << " Tags " << tag << " : " << dtype << " Save " << save << "\n";
  TFile *file = new TFile(fname.c_str());
  if (file) {
    char dirx[100];
    sprintf(dirx, "%s", dirnm.c_str());
    TDirectory *dir = (TDirectory *)file->FindObjectAny(dirx);
    if (debug)
      std::cout << "Directory " << dirx << " : " << dir << std::endl;
    if (dir) {
      for (int ih = 0; ih < nhist; ++ih) {
        char hname[100];
        sprintf(hname, "%s", name2[ih].c_str());
        TH1D *hist1 = (TH1D *)dir->FindObjectAny(hname);
        if (debug)
          std::cout << "Hist " << hname << " : " << hist1 << " Xtitle " << xtitl[ih] << " Ytitle " << ytitl[ih]
                    << " ibin " << ibin[ih] << std::endl;
        if (hist1 != nullptr) {
          char name[100], title[100];
          sprintf(name, "%s%s", hname, tag.c_str());
          TCanvas *pad = new TCanvas(name, name, 500, 500);
          pad->SetRightMargin(0.10);
          pad->SetTopMargin(0.10);
          sprintf(title, "%s (%s)", hist1->GetTitle(), dtype.c_str());
          if (debug)
            std::cout << "Pad " << name << " : " << pad << "\n";
          hist1->GetYaxis()->SetTitle(ytitl[ih].c_str());
          hist1->GetXaxis()->SetTitle(xtitl[ih].c_str());
          hist1->SetTitle(title);
          pad->SetLogy();
          if (ibin[ih] > 0)
            hist1->Rebin(ibin[ih]);
          hist1->GetYaxis()->SetTitleOffset(1.2);
          hist1->Draw();
          pad->Update();
          TPaveStats *st1 = ((TPaveStats *)hist1->GetListOfFunctions()->FindObject("stats"));
          if (st1 != NULL) {
            st1->SetY1NDC(0.70);
            st1->SetY2NDC(0.90);
            st1->SetX1NDC(0.65);
            st1->SetX2NDC(0.90);
          }
          pad->Modified();
          pad->Update();
          if (save) {
            sprintf(name, "c_%s.jpg", pad->GetName());
            pad->Print(name);
          }
        }
      }
    }
  }
}

void ehcalPlots(std::string fname = "ecalHitdd4hep.root",
                std::string dirnm = "EcalSimHitStudy",
                std::string tag = "DD4Hep",
                int dets = 2,
                bool save = false,
                bool debug = false) {
  const int nh = 2;
  std::string name[nh] = {"poszp", "poszn"};
  gStyle->SetCanvasBorderMode(0);
  gStyle->SetCanvasColor(kWhite);
  gStyle->SetPadColor(kWhite);
  gStyle->SetFillColor(kWhite);
  gStyle->SetOptStat(10);
  if (debug)
    std::cout << "File " << fname << " Tag " << tag << " Detectors " << dets << " Save " << save << "\n";
  TFile *file = new TFile(fname.c_str());
  if (file) {
    TDirectory *dir = (TDirectory *)file->FindObjectAny(dirnm.c_str());
    if (debug)
      std::cout << "Directory " << dirnm << " : " << dir << std::endl;
    if (dir) {
      for (int id = 0; id < dets; ++id) {
        for (int ih = 0; ih < nh; ++ih) {
          char hname[100];
          sprintf(hname, "%s%d", name[ih].c_str(), id);
          TH2D *hist = (TH2D *)(dir->FindObjectAny(hname));
          if (debug)
            std::cout << "Hist " << hname << " : " << hist << ":" << hist->GetTitle() << " Xtitle "
                      << hist->GetXaxis()->GetTitle() << " Ytitle " << hist->GetYaxis()->GetTitle() << std::endl;
          if (hist != nullptr) {
            char cname[100], title[100];
            sprintf(cname, "%s%s", hname, tag.c_str());
            TCanvas *pad = new TCanvas(cname, cname, 500, 500);
            pad->SetRightMargin(0.10);
            pad->SetTopMargin(0.10);
            sprintf(title, "%s (%s)", hist->GetTitle(), tag.c_str());
            if (debug)
              std::cout << "Pad " << cname << " : " << pad << "\n";
            hist->SetTitle(title);
            hist->SetMarkerStyle(20);
            hist->SetMarkerSize(0.1);
            hist->GetYaxis()->SetTitleOffset(1.3);
            hist->Draw();
            pad->Update();
            TPaveStats *st1 = ((TPaveStats *)hist->GetListOfFunctions()->FindObject("stats"));
            if (st1 != NULL) {
              st1->SetY1NDC(0.85);
              st1->SetY2NDC(0.90);
              st1->SetX1NDC(0.70);
              st1->SetX2NDC(0.90);
            }
            pad->Modified();
            pad->Update();
            if (save) {
              sprintf(cname, "c_%s.jpg", pad->GetName());
              pad->Print(cname);
            }
          }
        }
      }
    }
  }
}

class hgcHits {
public:
  TTree *fChain;
  Int_t fCurrent;

private:
  // Declaration of leaf types
  std::vector<float> *heeRecX;
  std::vector<float> *heeRecY;
  std::vector<float> *heeRecZ;
  std::vector<float> *heeRecEnergy;
  std::vector<float> *hefRecX;
  std::vector<float> *hefRecY;
  std::vector<float> *hefRecZ;
  std::vector<float> *hefRecEnergy;
  std::vector<float> *hebRecX;
  std::vector<float> *hebRecY;
  std::vector<float> *hebRecZ;
  std::vector<float> *hebRecEta;
  std::vector<float> *hebRecPhi;
  std::vector<float> *hebRecEnergy;
  std::vector<float> *heeSimX;
  std::vector<float> *heeSimY;
  std::vector<float> *heeSimZ;
  std::vector<float> *heeSimEnergy;
  std::vector<float> *hefSimX;
  std::vector<float> *hefSimY;
  std::vector<float> *hefSimZ;
  std::vector<float> *hefSimEnergy;
  std::vector<float> *hebSimX;
  std::vector<float> *hebSimY;
  std::vector<float> *hebSimZ;
  std::vector<float> *hebSimEta;
  std::vector<float> *hebSimPhi;
  std::vector<float> *hebSimEnergy;
  std::vector<unsigned int> *heeDetID;
  std::vector<unsigned int> *hefDetID;
  std::vector<unsigned int> *hebDetID;

  // List of branches
  TBranch *b_heeRecX;
  TBranch *b_heeRecY;
  TBranch *b_heeRecZ;
  TBranch *b_heeRecEnergy;
  TBranch *b_hefRecX;
  TBranch *b_hefRecY;
  TBranch *b_hefRecZ;
  TBranch *b_hefRecEnergy;
  TBranch *b_hebRecX;
  TBranch *b_hebRecY;
  TBranch *b_hebRecZ;
  TBranch *b_hebRecEta;
  TBranch *b_hebRecPhi;
  TBranch *b_hebRecEnergy;
  TBranch *b_heeSimX;
  TBranch *b_heeSimY;
  TBranch *b_heeSimZ;
  TBranch *b_heeSimEnergy;
  TBranch *b_hefSimX;
  TBranch *b_hefSimY;
  TBranch *b_hefSimZ;
  TBranch *b_hefSimEnergy;
  TBranch *b_hebSimX;
  TBranch *b_hebSimY;
  TBranch *b_hebSimZ;
  TBranch *b_hebSimEta;
  TBranch *b_hebSimPhi;
  TBranch *b_hebSimEnergy;
  TBranch *b_heeDetID;
  TBranch *b_hefDetID;
  TBranch *b_hebDetID;

public:
  hgcHits(const char *infile);
  virtual ~hgcHits();
  virtual Int_t Cut(Long64_t entry);
  virtual Int_t GetEntry(Long64_t entry);
  virtual Long64_t LoadTree(Long64_t entry);
  virtual void Init(TTree *tree);
  virtual void Loop();
  virtual Bool_t Notify();
  virtual void Show(Long64_t entry = -1);

  void bookHistograms();
  void saveHistos(const char *outfile);

private:
};

hgcHits::hgcHits::hgcHits(const char *infile) {
  TFile *file = new TFile(infile);
  TDirectory *dir = (TDirectory *)(file->FindObjectAny("hgcHitAnalysis"));
  TTree *tree = (TTree *)(dir->FindObjectAny("hgcHits"));
  std::cout << "Attaches tree hgcHits at " << tree << " in file " << infile << std::endl;
  bookHistograms();
  Init(tree);
}

hgcHits::~hgcHits() {
  if (!fChain)
    return;
  delete fChain->GetCurrentFile();
}

Int_t hgcHits::GetEntry(Long64_t entry) {
  // Read contents of entry.
  if (!fChain)
    return 0;
  return fChain->GetEntry(entry);
}

Long64_t hgcHits::LoadTree(Long64_t entry) {
  // Set the environment to read one entry
  if (!fChain)
    return -5;
  Long64_t centry = fChain->LoadTree(entry);
  if (centry < 0)
    return centry;
  if (!fChain->InheritsFrom(TChain::Class()))
    return centry;
  TChain *chain = (TChain *)fChain;
  if (chain->GetTreeNumber() != fCurrent) {
    fCurrent = chain->GetTreeNumber();
    Notify();
  }
  return centry;
}

void hgcHits::Init(TTree *tree) {
  // The Init() function is called when the selector needs to initialize
  // a new tree or chain. Typically here the branch addresses and branch
  // pointers of the tree will be set.
  // It is normally not necessary to make changes to the generated
  // code, but the routine can be extended by the user if needed.
  // Init() will be called many times when running on PROOF
  // (once per file to be processed).

  // Set object pointer
  heeRecX = 0;
  heeRecY = 0;
  heeRecZ = 0;
  heeRecEnergy = 0;
  hefRecX = 0;
  hefRecY = 0;
  hefRecZ = 0;
  hefRecEnergy = 0;
  hebRecX = 0;
  hebRecY = 0;
  hebRecZ = 0;
  hebRecEta = 0;
  hebRecPhi = 0;
  hebRecEnergy = 0;
  heeSimX = 0;
  heeSimY = 0;
  heeSimZ = 0;
  heeSimEnergy = 0;
  hefSimX = 0;
  hefSimY = 0;
  hefSimZ = 0;
  hefSimEnergy = 0;
  hebSimX = 0;
  hebSimY = 0;
  hebSimZ = 0;
  hebSimEta = 0;
  hebSimPhi = 0;
  hebSimEnergy = 0;
  heeDetID = 0;
  hefDetID = 0;
  hebDetID = 0;
  // Set branch addresses and branch pointers
  if (!tree)
    return;
  fChain = tree;
  fCurrent = -1;
  fChain->SetMakeClass(1);

  fChain->SetBranchAddress("heeRecX", &heeRecX, &b_heeRecX);
  fChain->SetBranchAddress("heeRecY", &heeRecY, &b_heeRecY);
  fChain->SetBranchAddress("heeRecZ", &heeRecZ, &b_heeRecZ);
  fChain->SetBranchAddress("heeRecEnergy", &heeRecEnergy, &b_heeRecEnergy);
  fChain->SetBranchAddress("hefRecX", &hefRecX, &b_hefRecX);
  fChain->SetBranchAddress("hefRecY", &hefRecY, &b_hefRecY);
  fChain->SetBranchAddress("hefRecZ", &hefRecZ, &b_hefRecZ);
  fChain->SetBranchAddress("hefRecEnergy", &hefRecEnergy, &b_hefRecEnergy);
  fChain->SetBranchAddress("hebRecX", &hebRecX, &b_hebRecX);
  fChain->SetBranchAddress("hebRecY", &hebRecY, &b_hebRecY);
  fChain->SetBranchAddress("hebRecZ", &hebRecZ, &b_hebRecZ);
  fChain->SetBranchAddress("hebRecEta", &hebRecEta, &b_hebRecEta);
  fChain->SetBranchAddress("hebRecPhi", &hebRecPhi, &b_hebRecPhi);
  fChain->SetBranchAddress("hebRecEnergy", &hebRecEnergy, &b_hebRecEnergy);
  fChain->SetBranchAddress("heeSimX", &heeSimX, &b_heeSimX);
  fChain->SetBranchAddress("heeSimY", &heeSimY, &b_heeSimY);
  fChain->SetBranchAddress("heeSimZ", &heeSimZ, &b_heeSimZ);
  fChain->SetBranchAddress("heeSimEnergy", &heeSimEnergy, &b_heeSimEnergy);
  fChain->SetBranchAddress("hefSimX", &hefSimX, &b_hefSimX);
  fChain->SetBranchAddress("hefSimY", &hefSimY, &b_hefSimY);
  fChain->SetBranchAddress("hefSimZ", &hefSimZ, &b_hefSimZ);
  fChain->SetBranchAddress("hefSimEnergy", &hefSimEnergy, &b_hefSimEnergy);
  fChain->SetBranchAddress("hebSimX", &hebSimX, &b_hebSimX);
  fChain->SetBranchAddress("hebSimY", &hebSimY, &b_hebSimY);
  fChain->SetBranchAddress("hebSimZ", &hebSimZ, &b_hebSimZ);
  fChain->SetBranchAddress("hebSimEta", &hebSimEta, &b_hebSimEta);
  fChain->SetBranchAddress("hebSimPhi", &hebSimPhi, &b_hebSimPhi);
  fChain->SetBranchAddress("hebSimEnergy", &hebSimEnergy, &b_hebSimEnergy);
  fChain->SetBranchAddress("heeDetID", &heeDetID, &b_heeDetID);
  fChain->SetBranchAddress("hefDetID", &hefDetID, &b_hefDetID);
  fChain->SetBranchAddress("hebDetID", &hebDetID, &b_hebDetID);
  Notify();
}

Bool_t hgcHits::Notify() {
  // The Notify() function is called when a new file is opened. This
  // can be either for a new TTree in a TChain or when when a new TTree
  // is started when using PROOF. It is normally not necessary to make changes
  // to the generated code, but the routine can be extended by the
  // user if needed. The return value is currently not used.

  return kTRUE;
}

void hgcHits::Show(Long64_t entry) {
  // Print contents of entry.
  // If entry is not specified, print current entry
  if (!fChain)
    return;
  fChain->Show(entry);
}

Int_t hgcHits::Cut(Long64_t) {
  // This function may be called from Loop.
  // returns  1 if entry is accepted.
  // returns -1 otherwise.
  return 1;
}

void hgcHits::Loop() {
  //   In a ROOT session, you can do:
  //      Root > .L hgcHits.C
  //      Root > hgcHits t
  //      Root > t.GetEntry(12); // Fill t data members with entry number 12
  //      Root > t.Show();       // Show values of entry 12
  //      Root > t.Show(16);     // Read and show values of entry 16
  //      Root > t.Loop();       // Loop on all entries
  //

  //     This is the loop skeleton where:
  //    jentry is the global entry number in the chain
  //    ientry is the entry number in the current Tree
  //  Note that the argument to GetEntry must be:
  //    jentry for TChain::GetEntry
  //    ientry for TTree::GetEntry and TBranch::GetEntry
  //
  //       To read only selected branches, Insert statements like:
  // METHOD1:
  //    fChain->SetBranchStatus("*",0);  // disable all branches
  //    fChain->SetBranchStatus("branchname",1);  // activate branchname
  // METHOD2: replace line
  //    fChain->GetEntry(jentry);       //read all branches
  //by  b_branchname->GetEntry(ientry); //read only this branch
  if (fChain == 0)
    return;

  Long64_t nentries = fChain->GetEntriesFast();

  Long64_t nbytes = 0, nb = 0;
  for (Long64_t jentry = 0; jentry < nentries; jentry++) {
    Long64_t ientry = LoadTree(jentry);
    if (ientry < 0)
      break;
    nb = fChain->GetEntry(jentry);
    nbytes += nb;
    // if (Cut(ientry) < 0) continue;
  }
}

void hgcHits::bookHistograms() {}

void hgcHits::saveHistos(const char *) {}
