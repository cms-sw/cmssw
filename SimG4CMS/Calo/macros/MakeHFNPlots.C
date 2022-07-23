//////////////////////////////////////////////////////////////////////////////
//
// Usage:
// .L MakeHFNPlots.C+g
//
//   To make plot of layers from a file created using SimHit/Digi/RecHit
//   using the scripts runHFNoseSimHitStudy_cfg.py, runHFNoseDigiStudy_cfg.py,
//   runHFNoseRecHitStudy_cfg.py in Validation/HGCalValidation
//
//     makeLayerPlots(fname, type, todomin, todomax, tag, text, save)
//
//   where
//     fname   std::string   Name of the ROOT file [hfnSimHitD94tt.root]
//     type    int           File type (0:SimHit; 2:Digi; 3:RecHit) [0]
//     todomin int           Range of plots to make    [0]
//     todomax int           (minimum: -1; maximum: 8) [7]
//     tag     std::string   To be added to the name of the canvas [""]
//     text    std::string   To be added to the title of the histogram [""]
//     save    bool          If the canvas is to be saved as jpg file [false]
//
//////////////////////////////////////////////////////////////////////////////

#include <TCanvas.h>
#include <TChain.h>
#include <TFile.h>
#include <TFitResult.h>
#include <TFitResultPtr.h>
#include <TGraphAsymmErrors.h>
#include <TH1D.h>
#include <TH2D.h>
#include <TLegend.h>
#include <TPaveStats.h>
#include <TPaveText.h>
#include <TProfile.h>
#include <TROOT.h>
#include <TStyle.h>

#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

void makeLayerPlots(std::string fname = "hfnSimHitD94tt.root",
                    int type = 0,
                    int todomin = 0,
                    int todomax = 7,
                    std::string tag = "",
                    std::string text = "",
                    bool save = false) {
  std::string dirnm[3] = {"hgcalSimHitStudy", "hfnoseDigiStudy", "hfnoseRecHitStudy"};
  std::string units[3] = {" (mm)", " (cm)", " (cm)"};
  std::string titlx[3] = {"SimHit", "Digi", "RecHit"};

  gStyle->SetCanvasBorderMode(0);
  gStyle->SetCanvasColor(kWhite);
  gStyle->SetPadColor(kWhite);
  gStyle->SetOptStat(0);
  if (type < 0 || type > 2)
    type = 0;
  TFile *file = new TFile(fname.c_str());
  if (file) {
    TDirectory *dir = (TDirectory *)file->FindObjectAny(dirnm[type].c_str());
    char cname[100], name[100], title[100], xtitl[100], ytitl[100];
    for (int i = todomin; i <= todomax; ++i) {
      if (i < 0) {
        sprintf(name, "RZ_HGCalHFNoseSensitive");
        sprintf(title, "%s (%s)", text.c_str(), titlx[type].c_str());
        sprintf(xtitl, "z %s", units[type].c_str());
        sprintf(ytitl, "R %s", units[type].c_str());
      } else {
        sprintf(name, "XY_L%d", i + 1);
        sprintf(title, "%s (Layer %d %s)", text.c_str(), i + 1, titlx[type].c_str());
        sprintf(xtitl, "x %s", units[type].c_str());
        sprintf(ytitl, "y %s", units[type].c_str());
      }
      TH2D *hist = (TH2D *)dir->FindObjectAny(name);
      std::cout << name << " read out at " << hist << std::endl;
      if (hist != nullptr) {
        sprintf(cname, "%s%s", name, tag.c_str());
        TCanvas *pad = new TCanvas(cname, cname, 500, 500);
        pad->SetRightMargin(0.10);
        pad->SetTopMargin(0.10);
        hist->GetYaxis()->SetTitleOffset(1.2);
        hist->GetYaxis()->SetTitle(ytitl);
        hist->GetXaxis()->SetTitle(xtitl);
        hist->SetTitle(title);
        if (i < 0 && type == 0)
          hist->GetXaxis()->SetNdivisions(5);
        hist->Draw("colz");
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
