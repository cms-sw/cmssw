//////////////////////////////////////////////////////////////////////////////
//
// Usage:
// .L MakeGVPlots.C+g
//
//   To make plot from a file created using CaloSimHitAnalysis
//     makeGVPlots(fname, ifG4, todomin, todomax, tag, text, save, dirnm);
//
//   To plot on the same Canvas plots from Geant4 and GeantV done using
//   CaloSimHitAnalysis
//     makeGV2Plots(fnmG4, fnmGV, todomin, todomax, normalize, tag, text, 
//                  save, dirnm)
//
//   To plot from passive hit collection produced using CaloSimHitAnalysis
//      makeGVSPlots(fnmG4, fnmGV, todomin, todomax, tag, text, save, dirnm)
//
//   where
//     fname   std::string   Name of the ROOT file (analG4.root)
//     fnmG4   std::string   Name of the ROOT file created using Geant4
//                           (analG4.root)
//     fnmGV   std::string   Name of the ROOT file created using GeantV
//                           (analGV.root)
//     ifGV    bool          If the file created using Geant4 (true)
//     todomin int           The first one in the series to be created (0)
//     todomax int           The last one in the series to be created
//                           (3:2:5 for GVPlots:GV2Plots:GVSPlots)
//     tag     std::string   To be added to the name of the canvas ("")
//     text    std::string   To be added to the title of the histogram ("")
//     save    bool          If the canvas is to be saved as jpg file (false)
//     dirnm   std::string   Name of the directory ("caloSimHitAnalysis")
//
//   The histogram series have different meanings for each function
//
//   GVPlots (16 in total):
//     "Energy deposit", "Hit time", "Total energy deposit", 
//     "Energy deposit after 15 ns",  "R vs Z", "R vs Z for hits after 15 ns",
//     "#phi vs #eta", "Energy deposit per Hit", 
//     "Energy deposit per Hit after 15 ns", "Total energy deposit in 100 ns", 
//     "Energy deposit for EM particles", "Energy deposit for non-EM particles",
//     "R", "Z", "#eta", "phi"
//
//   GV2Plots (13 in total):
//     "Energy deposit", "Hit time", "Total energy deposit", 
//     "Energy deposit after 15 ns", "Energy deposit per Hit", 
//     "Energy deposit per Hit after 15 ns", "Total energy deposit in 100 ns", 
//     "Energy deposit for EM particles", "Energy deposit for non-EM particles",
//     "R", "Z", "#eta", "phi"
//
//   GVSPlots (6 in total):
//     "Total Hits", "Tracks", "Energy Deposit in all components", 
//     "Time of all hits", "Energy Deposit in tracker subdetectors", 
//     "Time of hits in tracker subdetectors"
//
//   All plots in GVPlots or GV2Plots happen for EB, EE, HB and HE
//   There are 6 subdetectors for tracker:
//      Pixel Barrel, Pixel Forward, TIB, TID, TOB, TEC
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

void makeGVPlots(std::string fname="analG4.root", bool ifG4=true,
		 int todomin=0, int todomax=3, std::string tag="", 
		 std::string text="", bool save=false,
		 std::string dirnm="caloSimHitAnalysis") {

  std::string names[16] = {"Edep", "Time", "Etot", "Edep15", "rz", "rz2",
			   "etaphi", "EdepT", "EdepT15", "EtotG", "EdepEM",
			   "EdepHad", "rr", "zz", "eta", "phi"};
  int         types[16] = {1, 1, 1, 1, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1};
  int         dets[4]   = {0, 1, 2, 3};
  std::string detName[4]= {"EB", "EE", "HB", "HE"};
  std::string xtitle[16] = {"Energy deposit (GeV)", "Hit time (ns)",
			    "Total energy deposit (GeV)", 
			    "Energy deposit (GeV) after 15 ns", 
			    "z (cm)", "z (cm)", "#eta",
			    "Energy deposit (GeV) per Hit", 
			    "Energy deposit (GeV) per Hit after 15 ns", 
			    "Total energy deposit (GeV) in 100 ns", 
			    "Energy deposit (GeV) for EM particles", 
			    "Energy deposit (GeV) for non-EM particles", 
			    "R (cm)", "z (cm)", "#eta", "phi"};
  std::string ytitle[16] = {"Hits", "Hits", "Events", "Hits", "R (cm)",
			    "R (cm)", "#phi", "Hits", "Hits", "Events", 
			    "Hits", "Hits", "Hits", "Hits", "Hits", "Hits"};

  gStyle->SetCanvasBorderMode(0); gStyle->SetCanvasColor(kWhite);
  gStyle->SetPadColor(kWhite);    gStyle->SetFillColor(kWhite);
  gStyle->SetOptStat(111110);
  TFile      *file = new TFile(fname.c_str());
  if (file) {
    TDirectory *dir  = (TDirectory*)file->FindObjectAny(dirnm.c_str());
    char cname[100], name[100], title[100];
    for (int i1 = 0; i1<4; ++i1) {
      for (int i2=todomin; i2<=todomax; ++i2) {
	if (types[i2] == 1) {
	  sprintf (name, "%s%d", names[i2].c_str(), dets[i1]);
	  if (ifG4) 
	    sprintf (title, "%s %s (Geant4 Simulation)", text.c_str(), detName[i1].c_str());
	  else  
	    sprintf (title, "%s %s (GeantV Simulation)", text.c_str(), detName[i1].c_str());
	} else if (i1 == 0) {
	  sprintf (name, "%s", names[i2].c_str());
	  if (ifG4) sprintf (title, "%s (Geant4 Simulation)", text.c_str());
	  else      sprintf (title, "%s (GeantV Simulation)", text.c_str());
	} else {
	  continue;
	}
	TH1D* hist1(nullptr);
	TH2D* hist2(nullptr);
	if (types[i2] == 1) hist1 = (TH1D*)dir->FindObjectAny(name);
	else                hist2 = (TH2D*)dir->FindObjectAny(name);
//      std::cout << name << " read out at " << hist1 << ":" << hist2 << std::endl;
        if ((hist1 != nullptr) || (hist2 != nullptr)) {
	  if (ifG4) sprintf (cname, "%sG4%s", name, tag.c_str());
	  else      sprintf (cname, "%sGV%s", name, tag.c_str());
          TCanvas *pad = new TCanvas(cname,cname,500,500);
          pad->SetRightMargin(0.10); pad->SetTopMargin(0.10);
	  if (types[i2] == 1) {
	    hist1->GetYaxis()->SetTitleOffset(1.2);
	    hist1->GetYaxis()->SetTitle(ytitle[i2].c_str());
	    hist1->GetXaxis()->SetTitle(xtitle[i2].c_str());
	    hist1->SetTitle(title); 
	    pad->SetLogy();
	    hist1->Draw();
	  } else {
	    hist2->GetYaxis()->SetTitleOffset(1.2);
	    hist2->GetYaxis()->SetTitle(ytitle[i2].c_str());
	    hist2->GetXaxis()->SetTitle(xtitle[i2].c_str());
	    hist2->SetTitle(title); 
	    hist2->Draw();
	  }
	  pad->Update();
	  TPaveStats* st1 = ((hist1 != nullptr) ?
			     ((TPaveStats*)hist1->GetListOfFunctions()->FindObject("stats")) :
			     ((TPaveStats*)hist2->GetListOfFunctions()->FindObject("stats")));
	  if (st1 != NULL) {
	    st1->SetY1NDC(0.70); st1->SetY2NDC(0.90);
	    st1->SetX1NDC(0.65); st1->SetX2NDC(0.90);
	  }
	  pad->Modified(); pad->Update();
	  if (save) {
	    sprintf (name, "c_%s.jpg", pad->GetName());
	    pad->Print(name);
	  }
	}
      }
    }
  }
}


void makeGV2Plots(std::string fnmG4="analG4.root", 
		  std::string fnmGV="analGV.root", int todomin=0, 
		  int todomax=2, bool normalize=true, std::string tag="",
		  std::string text="", bool save=false, 
		  std::string dirnm="caloSimHitAnalysis") {

  std::string names[13] = {"Edep", "Time", "Etot", "Edep15", "EdepT", "EdepT15",
			   "EtotG", "EdepEM", "EdepHad", "rr", "zz", "eta", 
			   "phi"};
  int         dets[4]  = {0, 1, 2, 3};
  std::string detName[4] = {"EB", "EE", "HB", "HE"};
  std::string xtitle[13] = {"Energy deposit (GeV)", "Hit time (ns)",
			    "Total energy deposit (GeV)", 
			    "Energy deposit (GeV) after 15 ns",
			    "Energy deposit (GeV) per Hit", 
			    "Energy deposit (GeV) per Hit after 15 ns", 
			    "Total energy deposit (GeV) in 100 ns", 
			    "Energy deposit (GeV) for EM particles", 
			    "Energy deposit (GeV) for non-EM particles", 
			    "R (cm)", "z (cm)", "#eta", "phi"};
  std::string ytitle[13] = {"Hits", "Hits", "Events", "Hits", "Hits", "Hits",
			    "Events", "Hits", "Hits", "Hits", "Hits", "Hits",
			    "Hits"};

  gStyle->SetCanvasBorderMode(0); gStyle->SetCanvasColor(kWhite);
  gStyle->SetPadColor(kWhite);    gStyle->SetFillColor(kWhite);

  if (normalize) gStyle->SetOptStat(0);
  else           gStyle->SetOptStat(111100);
  TFile      *file1 = new TFile(fnmG4.c_str());
  TFile      *file2 = new TFile(fnmGV.c_str());
  if (file1 && file2) {
    TDirectory *dir1 = (TDirectory*)file1->FindObjectAny(dirnm.c_str());
    TDirectory *dir2 = (TDirectory*)file2->FindObjectAny(dirnm.c_str());
    char name[100], cname[100], title[100];
    for (int i1 = 0; i1<4; ++i1) {
      for (int i2=todomin; i2<=todomax; ++i2) {
	sprintf (name, "%s%d", names[i2].c_str(), dets[i1]);
	sprintf (cname, "%s%d%s", names[i2].c_str(), dets[i1], tag.c_str());
	sprintf (title, "%s  %s (Geant4 vs GeantV)", text.c_str(), detName[i1].c_str());
	TH1D *hist[2];
	hist[0] = (TH1D*)dir1->FindObjectAny(name);
	hist[1] = (TH1D*)dir2->FindObjectAny(name);
        if ((hist[0] != nullptr) && (hist[1] != nullptr)) {
	  // Plot superimposed histograms
          TCanvas *pad = new TCanvas(cname,cname,500,500);
	  TLegend *legend = new TLegend(0.44, 0.78, 0.64, 0.89);
          pad->SetRightMargin(0.10); pad->SetTopMargin(0.10); pad->SetLogy();
	  pad->SetFillColor(kWhite); legend->SetFillColor(kWhite);
	  int icol[2] = {1,2};
	  int isty[2] = {1,2};
	  int imty[2] = {20,24};
	  std::string type[2] = {"Geant4", "GeantV"};
	  double ymax(0.90);
	  double total[2] = {0,0};
	  for (int i=0; i<2; ++i) {
	    hist[i]->GetYaxis()->SetTitleOffset(1.2);
	    hist[i]->GetYaxis()->SetTitle(ytitle[i2].c_str());
	    hist[i]->GetXaxis()->SetTitle(xtitle[i2].c_str());
	    hist[i]->SetTitle(title); 
	    hist[i]->SetMarkerStyle(imty[i]);
	    hist[i]->SetMarkerColor(icol[i]);
	    hist[i]->SetLineColor(icol[i]);
	    hist[i]->SetLineStyle(isty[i]);
	    hist[i]->SetNdivisions(505,"X");
	    total[i] = hist[i]->GetEntries();
	    legend->AddEntry(hist[i],type[i].c_str(),"lp");
	    if (i == 0) {
	      if (normalize) hist[i]->DrawNormalized("hist");  
	      else           hist[i]->Draw();
	    } else {
	      if (normalize) hist[i]->DrawNormalized("sames hist");  
              else           hist[i]->Draw("sames");
	    }
	    pad->Update();
	  }
	  legend->Draw("same");
	  pad->Modified(); pad->Update();
	  for (int i=0; i<2; ++i) {
	    TPaveStats* st = (TPaveStats*)hist[i]->GetListOfFunctions()->FindObject("stats");
	    if (st != NULL) {
	      st->SetLineColor(icol[i]); st->SetTextColor(icol[i]); 
	      st->SetY1NDC(ymax-0.20);   st->SetY2NDC(ymax);
	      st->SetX1NDC(0.65);        st->SetX2NDC(0.90);
	      ymax -= 0.20;
	    }
	  }
	  pad->Modified(); pad->Update();
	  if (save) {
	    sprintf (name, "c_%s.jpg", pad->GetName());
	    pad->Print(name);
	  }

	  // Ratio plot
	  if (normalize) {
	    int nbin = hist[0]->GetNbinsX();
	    double xmin = hist[0]->GetBinLowEdge(1);
	    double dx   = hist[0]->GetBinWidth(1);
	    double xmax = xmin + nbin*dx;
	    double fac  = total[1]/total[0];
	    int npt = 0;
	    double sumNum(0), sumDen(0), xpt[200], dxp[200], ypt[200], dyp[200];
	    for (int i=0; i<nbin; ++i) {
	      if (hist[0]->GetBinContent(i+1) > 0 && 
		  hist[1]->GetBinContent(i+1) > 0) {
		ypt[npt] = (fac*hist[0]->GetBinContent(i+1)/
			    hist[1]->GetBinContent(i+1));
		double er1 = hist[0]->GetBinError(i+1)/hist[0]->GetBinContent(i+1);
		double er2 = hist[1]->GetBinError(i+1)/hist[1]->GetBinContent(i+1);
		dyp[npt] = ypt[npt] * sqrt (er1*er1 + er2*er2);
		xpt[npt] = xmin + (i-0.5)*dx;
		dxp[npt] = 0;
		double temp1 = (ypt[npt]>1.0) ? 1.0/ypt[npt] : ypt[npt];
		double temp2 = (ypt[npt]>1.0) ? dyp[npt]/(ypt[npt]*ypt[npt]) : dyp[npt];
		sumNum += (fabs(1-temp1)/(temp2*temp2));
		sumDen += (1.0/(temp2*temp2));
		++npt;
	      }
	    }
	    sumNum  = (sumDen>0)  ? (sumNum/sumDen) : 0;
	    sumDen  = (sumDen>0)  ? 1.0/sqrt(sumDen) : 0;
	    TGraphAsymmErrors *graph = new TGraphAsymmErrors(npt, xpt, ypt, dxp,
							     dxp, dyp, dyp);
	    graph->SetMarkerStyle(24);
	    graph->SetMarkerColor(1);
	    graph->SetMarkerSize(0.8);
	    graph->SetLineColor(1);
	    graph->SetLineWidth(2);
	    sprintf (name, "%sRatio", pad->GetName());
	    TCanvas *canvas = new TCanvas(name, name, 500, 400);
	    gStyle->SetOptStat(0);     gPad->SetTopMargin(0.05);
	    gPad->SetLeftMargin(0.15); gPad->SetRightMargin(0.025);
	    gPad->SetBottomMargin(0.20);
	    TH1F *vFrame = canvas->DrawFrame(xmin, 0.01, xmax, 0.5);
	    vFrame->GetYaxis()->SetRangeUser(0.4,1.5);
	    vFrame->GetXaxis()->SetLabelSize(0.035);
	    vFrame->GetYaxis()->SetLabelSize(0.04);
	    vFrame->GetXaxis()->SetTitleSize(0.045);
	    vFrame->GetYaxis()->SetTitleSize(0.045);
	    vFrame->GetYaxis()->SetTitleOffset(1.2);
	    vFrame->GetXaxis()->SetRangeUser(xmin, xmax);
	    vFrame->GetYaxis()->SetTitle("Geant4/GeantV");  
	    sprintf (name, "%s in %s", xtitle[i2].c_str(), detName[i1].c_str());
	    vFrame->GetXaxis()->SetTitle(name);
	    graph->Draw("P");
	    TLine *line = new TLine(xmin, 1.0, xmax, 1.0);
	    line->SetLineStyle(2); line->SetLineWidth(2); line->SetLineColor(kRed);
	    line->Draw();
	    sprintf (title, "Mean Deviation = %5.3f #pm %5.3f", sumNum, sumDen);
	    TPaveText* text = new TPaveText(0.16, 0.85, 0.60, 0.90, "brNDC");
	    text->AddText(title); text->Draw("same");
	    canvas->Modified(); canvas->Update();
	    if (save) {
	      sprintf (name, "c_%s.jpg", canvas->GetName());
	      canvas->Print(name);
	    }
	  }
	}
      }
    }
  }
}


void makeGVSPlots(std::string fnmG4="analG4.root", 
		  std::string fnmGV="analGV.root", 
		  int todomin=0, int todomax=5,
		  std::string tag="", std::string text="", bool save=false,
		  std::string dirnm="caloSimHitAnalysis") {

  std::string names[6] = {"hitp", "trackp", "edepp", "timep", "edept", "timet"};
  std::string xtitle[6] = {"Hits", "Tracks", "Energy Deposit (MeV)",
			   "Time (ns)", "Energy Deposit (MeV)", "Time (ns)"};
  std::string ytitle[6] = {"Events", "Events", "Hits", "Hits", "Hits", "Hits"};
  std::string detName[6] = {"Barrel Pixel", "Forward Pixel", "TIB", "TID",
			    "TOB", "TEC"};
  int boxp[6] = {0, 1, 0, 0, 0, 0};
  int nhis[6] = {1, 1, 1, 1, 6, 6};

  gStyle->SetCanvasBorderMode(0); gStyle->SetCanvasColor(kWhite);
  gStyle->SetPadColor(kWhite);    gStyle->SetFillColor(kWhite);
  gStyle->SetOptStat(111110);
  TFile      *file1 = new TFile(fnmG4.c_str());
  TFile      *file2 = new TFile(fnmGV.c_str());
  if (file1 && file2) {
    TDirectory *dir1 = (TDirectory*)file1->FindObjectAny(dirnm.c_str());
    TDirectory *dir2 = (TDirectory*)file2->FindObjectAny(dirnm.c_str());
    char name[100], cname[100], title[100];
    for (int i2=todomin; i2<=todomax; ++i2) {
      for (int i1=0; i1<nhis[i2]; ++ i1) {
	if (nhis[i2] == 1) {
	  sprintf (name, "%s", names[i2].c_str());
	  sprintf (cname, "%s%s", names[i2].c_str(), tag.c_str());
	  sprintf (title, "%s (Geant4 vs GeantV)", text.c_str());
	} else {
	  sprintf (name, "%s%d", names[i2].c_str(), i1);
	  sprintf (cname, "%s%d%s", names[i2].c_str(), i1, tag.c_str());
	  sprintf (title, "%s in %s (Geant4 vs GeantV)", text.c_str(),
		   detName[i1].c_str());
	}
	TH1D *hist[2];
	hist[0] = (TH1D*)dir1->FindObjectAny(name);
	hist[1] = (TH1D*)dir2->FindObjectAny(name);
	if ((hist[0] != nullptr) && (hist[1] != nullptr)) {
	  // Plot superimposed histograms
	  TCanvas *pad = new TCanvas(cname,cname,500,500);
	  TLegend *legend = new TLegend(0.44, 0.78, 0.64, 0.89);
	  pad->SetRightMargin(0.10); pad->SetTopMargin(0.10); pad->SetLogy();
	  pad->SetFillColor(kWhite); legend->SetFillColor(kWhite);
	  int icol[2] = {1,2};
	  int isty[2] = {1,2};
	  int imty[2] = {20,24};
	  std::string type[2] = {"Geant4", "GeantV"};
	  double ymax(0.90);
	  double total[2] = {0,0};
	  double ymaxv[2] = {0,0};
	  for (int i=0; i<2; ++i) {
	    hist[i]->GetYaxis()->SetTitleOffset(1.2);
	    hist[i]->GetYaxis()->SetTitle(ytitle[i2].c_str());
	    hist[i]->GetXaxis()->SetTitle(xtitle[i2].c_str());
	    hist[i]->SetTitle(title); 
	    hist[i]->SetMarkerStyle(imty[i]);
	    hist[i]->SetMarkerColor(icol[i]);
	    hist[i]->SetLineColor(icol[i]);
	    hist[i]->SetLineStyle(isty[i]);
	    hist[i]->SetNdivisions(505,"X");
	    total[i] = hist[i]->GetEntries();
	    legend->AddEntry(hist[i],type[i].c_str(),"lp");
	    ymaxv[i] = hist[i]->GetMaximum();
	  }
	  int first =  (ymaxv[0] > ymaxv[1]) ? 0 : 1;
	  int next  = 1 - first;
	  hist[first]->Draw();
	  hist[next]->Draw("sames");
	  pad->Update();
	  legend->Draw("same");
	  pad->Modified(); pad->Update();
	  for (int i=0; i<2; ++i) {
	    TPaveStats* st = (TPaveStats*)hist[i]->GetListOfFunctions()->FindObject("stats");
	    if (st != NULL) {
	      double xl = (boxp[i2] == 0) ? 0.65 : 0.10;
	      st->SetLineColor(icol[i]); st->SetTextColor(icol[i]); 
	      st->SetY1NDC(ymax-0.15);   st->SetY2NDC(ymax);
	      st->SetX1NDC(xl);          st->SetX2NDC(xl+0.25);
	      ymax -= 0.15;
	    }
	  }
	  pad->Modified(); pad->Update();
	  if (save) {
	    sprintf (name, "c_%s.jpg", pad->GetName());
	    pad->Print(name);
	  }
	}
      }
    }
  }
}
