#include "Rtypes.h"
#include "TROOT.h"
#include "TRint.h"
#include "TObject.h"
#include "TFile.h"
#include "TTree.h"
#include "TH1F.h"
#include "TCanvas.h"
#include "TApplication.h"
#include "TRefArray.h"
#include "TStyle.h"
#include "TGraph.h"

#include <string>
#include <iostream>
#include <fstream>
#include <iomanip>

void setTDRStyle();

void plotValid(char name[10]="Hit34", int logy=0, int mode=0, 
	       float xoff=0.55, float yoff=0.15, int save=0, int bin=1) {

  char title[80];
  if      (name == "Hit01") sprintf (title, "Number of Hits in HCal");
  else if (name == "Hit02") sprintf (title, "Hits with wrong Detector ID"); 
  else if (name == "Hit03") sprintf (title, "Hits with wrong Subdet");
  else if (name == "Hit04") sprintf (title, "Hits with wrong ID");
  else if (name == "Hit05") sprintf (title, "Number of Hits in HB");
  else if (name == "Hit06") sprintf (title, "Number of Hits in HE");
  else if (name == "Hit07") sprintf (title, "Number of Hits in HO");
  else if (name == "Hit08") sprintf (title, "Number of Hits in HF");
  else if (name == "Hit09") sprintf (title, "Detector ID");
  else if (name == "Hit10") sprintf (title, "Subdetectors in HCal");
  else if (name == "Hit11") sprintf (title, "Depths in HCal");
  else if (name == "Hit12") sprintf (title, "#eta in HCal");
  else if (name == "Hit13") sprintf (title, "#phi in HCal");
  else if (name == "Hit14") sprintf (title, "Energy in HCal");
  else if (name == "Hit15") sprintf (title, "Time in HCal");
  else if (name == "Hit16") sprintf (title, "Time in HCal (E wtd)");
  else if (name == "Hit17") sprintf (title, "Depths in HB");
  else if (name == "Hit18") sprintf (title, "Depths in HE");
  else if (name == "Hit19") sprintf (title, "Depths in HO");
  else if (name == "Hit20") sprintf (title, "Depths in HF");
  else if (name == "Hit21") sprintf (title, "#eta in HB");
  else if (name == "Hit22") sprintf (title, "#eta in HE");
  else if (name == "Hit23") sprintf (title, "#eta in HO");
  else if (name == "Hit24") sprintf (title, "#eta in HF");
  else if (name == "Hit25") sprintf (title, "#phi in HB");
  else if (name == "Hit26") sprintf (title, "#phi in HE");
  else if (name == "Hit27") sprintf (title, "#phi in HO");
  else if (name == "Hit28") sprintf (title, "#phi in HF");
  else if (name == "Hit29") sprintf (title, "Energy in HB");
  else if (name == "Hit30") sprintf (title, "Energy in HE");
  else if (name == "Hit31") sprintf (title, "Energy in HO");
  else if (name == "Hit32") sprintf (title, "Energy in HF");
  else if (name == "Hit33") sprintf (title, "Time in HB");
  else if (name == "Hit34") sprintf (title, "Time in HE");
  else if (name == "Hit35") sprintf (title, "Time in HO");
  else if (name == "Hit36") sprintf (title, "Time in HF");
  else if (name == "Lay01") sprintf (title, "Layer number of the Hits");
  else if (name == "Lay02") sprintf (title, "#eta of the Hits");
  else if (name == "Lay03") sprintf (title, "#phi of the Hits");
  else if (name == "Lay04") sprintf (title, "Energy of the Hits");
  else if (name == "Lay05") sprintf (title, "Depth  of the Hits");
  else if (name == "Lay06") sprintf (title, "Time of the Hits");
  else if (name == "Lay07") sprintf (title, "Time (wtd) of Hits");
  else if (name == "Lay08") sprintf (title, "#phi vs. #eta of the Hits");
  else if (name == "Lay09") sprintf (title, "Hit in Ecal");
  else if (name == "Lay10") sprintf (title, "Hit in Hcal");
  else if (name == "Lay11") sprintf (title, "Total Hits");
  else if (name == "Lay12") sprintf (title, "Energy per layer");
  else if (name == "Lay13") sprintf (title, "Lonitudinal Shower Profile");
  else if (name == "Lay14") sprintf (title, "Energy per depth");
  else if (name == "Lay15") sprintf (title, "Total Energy");
  else if (name == "Lay16") sprintf (title, "Energy in HO");
  else if (name == "Lay17") sprintf (title, "Energy in HB/HE");
  else if (name == "Lay18") sprintf (title, "Energy in HF (Long)");
  else if (name == "Lay19") sprintf (title, "Energy in HF (Short)");
  else if (name == "Lay20") sprintf (title, "EM   energy in HF");
  else if (name == "Lay21") sprintf (title, "Had. energy in HF");
  else if (name == "Layl0") sprintf (title, "Energy deposit in Layer 1");
  else if (name == "Layl1") sprintf (title, "Energy deposit in Layer 2");
  else if (name == "Layl2") sprintf (title, "Energy deposit in Layer 3");
  else if (name == "Layl3") sprintf (title, "Energy deposit in Layer 4");
  else if (name == "Layl4") sprintf (title, "Energy deposit in Layer 5");
  else if (name == "Layl5") sprintf (title, "Energy deposit in Layer 6");
  else if (name == "Layl6") sprintf (title, "Energy deposit in Layer 7");
  else if (name == "Layl7") sprintf (title, "Energy deposit in Layer 8");
  else if (name == "Layl8") sprintf (title, "Energy deposit in Layer 9");
  else if (name == "Layl9") sprintf (title, "Energy deposit in Layer 10");
  else if (name == "Layl10")sprintf (title, "Energy deposit in Layer 11");
  else if (name == "Layl11")sprintf (title, "Energy deposit in Layer 12");
  else if (name == "Layl12")sprintf (title, "Energy deposit in Layer 13");
  else if (name == "Layl13")sprintf (title, "Energy deposit in Layer 14");
  else if (name == "Layl14")sprintf (title, "Energy deposit in Layer 15");
  else if (name == "Layl15")sprintf (title, "Energy deposit in Layer 16");
  else if (name == "Layl16")sprintf (title, "Energy deposit in Layer 17");
  else if (name == "Layl17")sprintf (title, "Energy deposit in Layer 18");
  else if (name == "Layl18")sprintf (title, "Energy deposit in Layer 19");
  else if (name == "Layl19")sprintf (title, "Energy deposit in Layer 20");
  else if (name == "Layl20")sprintf (title, "Energy deposit in Depth 1");
  else if (name == "Layl21")sprintf (title, "Energy deposit in Depth 2");
  else if (name == "Layl22")sprintf (title, "Energy deposit in Depth 3");
  else if (name == "Layl23")sprintf (title, "Energy deposit in Depth 4");
  else if (name == "Layl24")sprintf (title, "Energy deposit in Depth 5");
  else if (name == "NxN01") sprintf (title, "Energy in ECal (NxN)r");
  else if (name == "NxN02") sprintf (title, "Energy in HCal (NxN)r");
  else if (name == "NxN03") sprintf (title, "Energy in HO (NxN)r");
  else if (name == "NxN04") sprintf (title, "Energy Total (NxN)r");
  else if (name == "NxN05") sprintf (title, "Energy in ECal (NxN)");
  else if (name == "NxN06") sprintf (title, "Energy in HCal (NxN)");
  else if (name == "NxN07") sprintf (title, "Energy in HO (NxN)");
  else if (name == "NxN08") sprintf (title, "Energy Total (NxN)");
  else if (name == "NxN09") sprintf (title, "Energy of Hits in (NxN)");
  else if (name == "NxN10") sprintf (title, "Time   of Hits in (NxN)");
  else if (name == "NxN11") sprintf (title, "Dist.  of Hits in (NxN)");
  else if (name == "Jet01") sprintf (title, "R of Hits");
  else if (name == "Jet02") sprintf (title, "T of Hits");
  else if (name == "Jet03") sprintf (title, "E of Hits");
  else if (name == "Jet04") sprintf (title, "Ecal Energy (First Jet)");
  else if (name == "Jet05") sprintf (title, "Hcal Energy (First Jet)");
  else if (name == "Jet06") sprintf (title, "Ho   Energy (First Jet)");
  else if (name == "Jet07") sprintf (title, "Total Energy(First Jet)");
  else if (name == "Jet08") sprintf (title, "Energy in Hcal vs  Ecal");
  else if (name == "Jet09") sprintf (title, "Delta #eta");
  else if (name == "Jet10") sprintf (title, "Delta #phi");
  else if (name == "Jet11") sprintf (title, "Delta R");
  else if (name == "Jet12") sprintf (title, "Di-jet mass");
  else if (name == "Jet13") sprintf (title, "Jet Energy");
  else if (name == "Jet14") sprintf (title, "Jet #eta");
  else if (name == "Jet15") sprintf (title, "Jet #phi");
  else                      sprintf (title, "Unknown");

  cout << name << " Title " << title << "\n";
  if (title != "Unknown") {
    char file1[50], file2[50], file3[50], file4[50], file5[50], file6[50];
    char lego1[32], lego2[32], lego3[32], lego4[32], lego5[32], lego6[32];
    sprintf (file1, "valid_HB1.root"); sprintf (lego1, "Old default");
    sprintf (file2, "valid_HB3.root"); sprintf (lego2, "t^{n} > 1 #mus");
    sprintf (file3, "valid_HB4.root"); sprintf (lego3, "E_{Kin}^{n} < 100 keV");
    sprintf (file4, "valid_HB5.root"); sprintf (lego4, "E_{Kin}^{n} < 1 MeV");
    sprintf (file5, "valid_HB6.root"); sprintf (lego5, "E_{Kin}^{n} < 10 MeV");
    sprintf (file6, "valid_HB7.root"); sprintf (lego6, "E_{Kin}^{n} < 100 MeV");
    char fileps[50];
    sprintf (fileps, "%s.eps", name);
    cout << fileps << "\n";

    setTDRStyle();
    TCanvas *myc = new TCanvas("myc","",800,600);
    if (logy > 0) gPad->SetLogy(1);

    cout << "Open " << file1 << "\n";
    TFile* File1 = new TFile(file1);
    TDirectory *d1 = (TDirectory*)File1->Get("DQMData/HcalHitValidation");
    TH1F* hist1 = (TH1F*) d1->Get(name);
    hist1->Rebin(bin);
    hist1->SetLineStyle(1);
    hist1->SetLineWidth(3);
    hist1->SetLineColor(2);
    cout << "Open " << file2 << "\n";
    TFile* File2 = new TFile(file2);
    TDirectory *d2 = (TDirectory*)File2->Get("DQMData/HcalHitValidation");
    TH1F* hist2 = (TH1F*) d2->Get(name);
    hist2->Rebin(bin);
    hist2->SetLineStyle(2);
    hist2->SetLineWidth(3);
    hist2->SetLineColor(4);
    cout << "Open " << file3 << "\n";
    TFile* File3 = new TFile(file3);
    TDirectory *d3 = (TDirectory*)File3->Get("DQMData/HcalHitValidation");
    TH1F* hist3 = (TH1F*) d3->Get(name);
    hist3->Rebin(bin);
    hist3->SetLineStyle(3);
    hist3->SetLineWidth(3);
    hist3->SetLineColor(8);
    cout << "Open " << file4 << "\n";
    TFile* File4 = new TFile(file4);
    TDirectory *d4 = (TDirectory*)File4->Get("DQMData/HcalHitValidation");
    TH1F* hist4 = (TH1F*) d4->Get(name);
    hist4->Rebin(bin);
    hist4->SetLineStyle(4);
    hist4->SetLineWidth(3);
    hist4->SetLineColor(1);
    cout << "Open " << file5 << "\n";
    TFile* File5 = new TFile(file5);
    TDirectory *d5 = (TDirectory*)File5->Get("DQMData/HcalHitValidation");
    TH1F* hist5 = (TH1F*) d5->Get(name);
    hist5->Rebin(bin);
    hist5->SetLineStyle(5);
    hist5->SetLineWidth(3);
    hist5->SetLineColor(6);
    cout << "Open " << file6 << "\n";
    TFile* File6 = new TFile(file6);
    TDirectory *d6 = (TDirectory*)File6->Get("DQMData/HcalHitValidation");
    TH1F* hist6 = (TH1F*) d6->Get(name);
    hist6->Rebin(bin);
    hist6->SetLineStyle(6);
    hist6->SetLineWidth(3);
    hist6->SetLineColor(3);

    if (mode == 1) {
      hist1->GetXaxis()->SetTitle(title);
      hist1->Draw();
      hist2->Draw("same");
      hist3->Draw("same");
      hist4->Draw("same");
      hist5->Draw("same");
      hist6->Draw("same");
    } else if (mode == 2) {
      hist2->GetXaxis()->SetTitle(title);
      hist2->Draw();
      hist1->Draw("same");
      hist3->Draw("same");
      hist4->Draw("same");
      hist5->Draw("same");
      hist6->Draw("same");
    } else if (mode == 3) {
      hist3->GetXaxis()->SetTitle(title);
      hist3->Draw();
      hist1->Draw("same");
      hist2->Draw("same");
      hist4->Draw("same");
      hist5->Draw("same");
      hist6->Draw("same");
    } else if (mode == 5) {
      hist5->GetXaxis()->SetTitle(title);
      hist5->Draw();
      hist1->Draw("same");
      hist2->Draw("same");
      hist3->Draw("same");
      hist4->Draw("same");
      hist6->Draw("same");
    } else if (mode == 6) {
      hist6->GetXaxis()->SetTitle(title);
      hist6->Draw();
      hist1->Draw("same");
      hist2->Draw("same");
      hist3->Draw("same");
      hist4->Draw("same");
      hist5->Draw("same");
    } else if (mode < 1) {
      hist4->GetXaxis()->SetTitle(title);
      hist4->Draw();
      hist1->Draw("same");
      hist2->Draw("same");
      hist3->Draw("same");
    } else {
      hist4->GetXaxis()->SetTitle(title);
      hist4->Draw();
      hist1->Draw("same");
      hist2->Draw("same");
      hist3->Draw("same");
      hist5->Draw("same");
      hist6->Draw("same");
    }

    float xmax = xoff+0.3;
    float ymax = yoff+0.2;
    leg1 = new TLegend(xoff,yoff,xmax,ymax);
    leg1->AddEntry(hist1,lego1,"F");
    leg1->AddEntry(hist2,lego2,"F");
    leg1->AddEntry(hist3,lego3,"F");
    leg1->AddEntry(hist4,lego4,"F");
    if (mode > 0) {
      leg1->AddEntry(hist5,lego5,"F");
      leg1->AddEntry(hist6,lego6,"F");
    }
    leg1->Draw();

    if (save > 0) myc->SaveAs(fileps);

  }
}
void plotHFPosition(char file[100]="hf.out", int save=0) {

  setTDRStyle();
  std::ifstream theFile(file);
  static const int NPMAX=20000;
  int    etaIndex[NPMAX], phiIndex[NPMAX], depths[NPMAX], np=0;
  float  radiusH[NPMAX], phiH[NPMAX];
  double deg=3.1415926/180.;
  while (theFile) {
    int   ieta, iphi, idep;
    float rr, phideg;
    theFile >> ieta >> iphi >> idep >> rr >> phideg;
    if (np < NPMAX) {
      etaIndex[np] = ieta;
      phiIndex[np] = iphi;
      depths[np]   = idep;
      radiusH[np]  = rr;
      if (phideg < 0) phideg += 360;
      int iphi = (int)(phideg/20);
      float phi = (phideg - iphi*20.0);
      if (phi > 10) phi -= 20;
      phiH[np] = phi*deg;
      np++;
    }
  }
  std::cout << np << " points found\n";
  //  for (int i=0; i<np; i++) std::cout << std::setw(4) << i << " " << std::setw(3) << etaIndex[i] << " " << std::setw(3) << phiIndex[i] << " " << depths[i] << " " << std::setw(6) << radiusH[i] << " " << std::setw(6) << phiH[i] << "\n";

  int symbol[14] = {20,21,22,23,24,25,26,27,28,29,30,3,5,2};
  int colr[3]    = {2,4,1};
  TGraph *gr[3][14];
  float x1[NPMAX], y1[NPMAX], x2[NPMAX], y2[NPMAX], x3[NPMAX], y3[NPMAX];
  int np1, np2, np3;
  for (int i=0; i<13; i++) {
    np1 = np2 = np3 = 0;
    for (int j=0; j<np; j++) {
      if (etaIndex[j] == (i+29)) {
	int k = 2;
	if (depths[j] == 3) k = 0;
	else if (depths[j] == 4) k = 1;
	if (k == 0) {
	  x1[np1] = radiusH[j]*cos(phiH[j]);
	  y1[np1] = radiusH[j]*sin(phiH[j]);
	  np1++;
	  //	  if (np1 == 1) std::cout << i << " 0 " <<x1[0] << " " <<y1[0] << "\n";
	} else if (k==1) {
	  x2[np2] = radiusH[j]*cos(phiH[j]);
	  y2[np2] = radiusH[j]*sin(phiH[j]);
	  np2++;
	  //	  if (np2 == 1) std::cout << i << " 1 " <<x2[0] << " " <<y2[0] << "\n";
	} else {
	  x3[np3] = radiusH[j]*cos(phiH[j]);
	  y3[np3] = radiusH[j]*sin(phiH[j]);
	  np3++;
	}
      }
    }
    //    std::cout << "i " << i << " " <<np1 << " " <<np2 << " " <<np3 <<"\n";
    if (np1 > 0) {
      gr[0][i] = new TGraph(np1,x1,y1); gr[0][i]->SetTitle(""); 
      gr[0][i]->SetMarkerStyle(symbol[i]);  gr[0][i]->SetMarkerColor(colr[0]);
    } else 
      gr[0][i] = 0;
    if (np2 > 0) {
      gr[1][i] = new TGraph(np2,x2,y2); gr[1][i]->SetTitle(""); 
      gr[1][i]->SetMarkerStyle(symbol[i]);  gr[1][i]->SetMarkerColor(colr[1]);
    } else 
      gr[1][i] = 0;
    if (np3 > 0) {
      gr[2][i] = new TGraph(np3,x3,y3); gr[2][i]->SetTitle(""); 
      gr[2][i]->SetMarkerStyle(symbol[i]);  gr[2][i]->SetMarkerColor(colr[2]);
    } else 
      gr[2][i] = 0;
  }
  np1 = np2 = np3 = 0;
  for (int j=0; j<np; j++) {
    if (etaIndex[j] < 29 || etaIndex[j] > 41) {
      int k = 2;
      if (depths[j] == 3) k = 0;
      else if (depths[j] == 4) k = 1;
      if (k == 0) {
	x1[np1] = radiusH[j]*cos(phiH[j]);
	y1[np1] = radiusH[j]*sin(phiH[j]);
	np1++;
	if (np1 == 1) std::cout << i << " 0 " <<x1[0] << " " <<y1[0] << "\n";
      } else if (k==1) {
	x2[np2] = radiusH[j]*cos(phiH[j]);
	y2[np2] = radiusH[j]*sin(phiH[j]);
	np2++;
	if (np2 == 1) std::cout << i << " 1 " <<x2[0] << " " <<y2[0] << "\n";
      } else {
	x3[np3] = radiusH[j]*cos(phiH[j]);
	y3[np3] = radiusH[j]*sin(phiH[j]);
	np3++;
      }
    }
  }
  //    std::cout << "i " << i << " " <<np1 << " " <<np2 << " " <<np3 <<"\n";
  if (np1 > 0) {
    gr[0][13] = new TGraph(np1,x1,y1); gr[0][13]->SetTitle(""); 
    gr[0][13]->SetMarkerStyle(symbol[13]);  gr[0][13]->SetMarkerColor(colr[0]);
  } else 
    gr[0][13] = 0;
  if (np2 > 0) {
    gr[1][13] = new TGraph(np2,x2,y2); gr[1][13]->SetTitle(""); 
    gr[1][13]->SetMarkerStyle(symbol[13]);  gr[1][13]->SetMarkerColor(colr[1]);
  } else 
    gr[1][13] = 0;
  if (np3 > 0) {
    gr[2][13] = new TGraph(np3,x3,y3); gr[2][13]->SetTitle(""); 
    gr[2][13]->SetMarkerStyle(symbol[13]);  gr[2][13]->SetMarkerColor(colr[2]);
  } else 
    gr[2][13] = 0;

  TCanvas *c0  = new TCanvas("c0","PMT Hits",800,600); 
  TH1F *vFrame = c0->DrawFrame(1000.0,-250.0,1500.0,250.0);
  vFrame->SetXTitle("x (mm)");
  vFrame->SetYTitle("y (mm)");
  for (int i=0; i<=13; i++) {
    for (int j=0; j<3; j++) {
      if (gr[j][i] != 0) {
	gr[j][i]->Draw("p");
	gr[j][i]->SetLineColor(colr[j]); gr[j][i]->SetLineWidth(2);
	//	std::cout << "Next " << i << " " << j << "\n";
      }
    }
  }
  TLegend *leg1 = new TLegend(0.75,0.55,0.90,0.90);
  char list[40];
  for (i=0; i<= 13; i++) {
    if (i < 13) sprintf (list, "#eta = %d", i+29);
    else        sprintf (list, "Unknown #eta");
    if      (gr[0][i] != 0) leg1->AddEntry(gr[0][i], list, "P");
    else if (gr[1][i] != 0) leg1->AddEntry(gr[1][i], list, "P");
  }
  for (i=0; i<2; i++) {
    if (i == 0) sprintf (list, "Long Fibre");
    else        sprintf (list, "Short Fibre");
    if      (gr[i][0] != 0) leg1->AddEntry(gr[i][0], list, "L");
    else if (gr[i][1] != 0) leg1->AddEntry(gr[i][1], list, "L");
    else if (gr[i][2] != 0) leg1->AddEntry(gr[i][2], list, "L");
  }
  leg1->SetFillColor(0); leg1->SetTextSize(0.03); 
  leg1->SetBorderSize(1); leg1->Draw();

  if (save != 0) {
    if (save > 0) c0->SaveAs("PMTHits.eps");
    else          c0->SaveAs("PMTHits.gif");
  }
}

void setTDRStyle() {
  TStyle *tdrStyle = new TStyle("tdrStyle","Style for P-TDR");

// For the canvas:
  tdrStyle->SetCanvasBorderMode(0);
  tdrStyle->SetCanvasColor(kWhite);
  tdrStyle->SetCanvasDefH(600); //Height of canvas
  tdrStyle->SetCanvasDefW(600); //Width of canvas
  tdrStyle->SetCanvasDefX(0);   //POsition on screen
  tdrStyle->SetCanvasDefY(0);

// For the Pad:
  tdrStyle->SetPadBorderMode(0);
  // tdrStyle->SetPadBorderSize(Width_t size = 1);
  tdrStyle->SetPadColor(kWhite);
  tdrStyle->SetPadGridX(false);
  tdrStyle->SetPadGridY(false);
  tdrStyle->SetGridColor(0);
  tdrStyle->SetGridStyle(3);
  tdrStyle->SetGridWidth(1);

// For the frame:
  tdrStyle->SetFrameBorderMode(0);
  tdrStyle->SetFrameBorderSize(1);
  tdrStyle->SetFrameFillColor(0);
  tdrStyle->SetFrameFillStyle(0);
  tdrStyle->SetFrameLineColor(1);
  tdrStyle->SetFrameLineStyle(1);
  tdrStyle->SetFrameLineWidth(1);


//For the date:
  tdrStyle->SetOptDate(0);
  // tdrStyle->SetDateX(Float_t x = 0.01);
  // tdrStyle->SetDateY(Float_t y = 0.01);

// For the statistics box:
  tdrStyle->SetOptFile(0);
  tdrStyle->SetOptStat(0); // To display the mean and RMS:   SetOptStat("mr");

// For the Global title:

  tdrStyle->SetOptTitle(0);
  tdrStyle->SetTitleFont(42);
  tdrStyle->SetTitleColor(1);
  tdrStyle->SetTitleTextColor(1);
  tdrStyle->SetTitleFillColor(10);
  tdrStyle->SetTitleFontSize(0.05);

// For the axis titles:

  tdrStyle->SetTitleColor(1, "XYZ");
  tdrStyle->SetTitleFont(42, "XYZ");
  tdrStyle->SetTitleSize(0.06, "XYZ");
  // tdrStyle->SetTitleXSize(Float_t size = 0.02); // Another way to set the size?
  // tdrStyle->SetTitleYSize(Float_t size = 0.02);
  tdrStyle->SetTitleXOffset(0.7);
  tdrStyle->SetTitleYOffset(0.7);
  // tdrStyle->SetTitleOffset(1.1, "Y"); // Another way to set the Offset

// For the axis labels:

  tdrStyle->SetLabelColor(1, "XYZ");
  tdrStyle->SetLabelFont(42, "XYZ");
  tdrStyle->SetLabelOffset(0.007, "XYZ");
  tdrStyle->SetLabelSize(0.03, "XYZ");

// For the axis:

  tdrStyle->SetAxisColor(1, "XYZ");
  tdrStyle->SetStripDecimals(kTRUE);
  tdrStyle->SetTickLength(0.03, "XYZ");
  tdrStyle->SetNdivisions(510, "XYZ");
  tdrStyle->SetPadTickX(1);  // To get tick marks on the opposite side of the frame
  tdrStyle->SetPadTickY(1);

  tdrStyle->cd();

}
