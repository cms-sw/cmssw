#include <iostream>
#include <fstream>
#include <iomanip>

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

void setTDRStyle();

void drawHFTime(char file[100]="/uscms_data/d1/sunanda/CMSSW_1_3_3/src/SimG4CMS/Calo/test/vcal5x5.root", int mode=0, float xmax=1000) {

  setTDRStyle();
  TFile *File1 = new TFile(file);
  TTree *em = File1->Get("h3");
  TTree *had = File1->Get("h8");
  TH1F *h1 = new TH1F("EM","Time (EM)",500,0,xmax);
  TH1F *h2 = new TH1F("Had","Time (Had)",500,0,xmax);
  if (mode <=0) {
    em->Project("EM","It");
    had->Project("Had","It");
  } else {
    em->Project("EM","T");
    had->Project("Had","T");
  }
  h1->GetXaxis()->SetTitle("Time (EM)");
  h2->GetXaxis()->SetTitle("Time (Had)");

  TCanvas *em1 = new TCanvas("em1","",800,500);
  gPad->SetLogy(1);
  h1->Draw();

  TCanvas *had1 = new TCanvas("had1","",800,500);
  gPad->SetLogy(1);
  h2->Draw();
}

void drawHFPosition(char file[100]="hf.out", int save=0) {

  setTDRStyle();
  std::ifstream theFile(file);
  static const int NPMAX=20000;
  int    etaIndex[NPMAX], phiIndex[NPMAX], depths[NPMAX], np=0;
  float  radiusT[NPMAX], radiusH[NPMAX], phiH[NPMAX];
  double deg=3.1415926/180.;
  while (theFile) {
    int   ieta, iphi, idep;
    float etaR, rr, phideg;
    theFile >> ieta >> iphi >> idep >> etaR >> rr >> phideg;
    if (np < NPMAX) {
      etaIndex[np] = ieta;
      phiIndex[np] = iphi;
      depths[np]   = idep;
      radiusT[np]  = etaR;
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
  //  for (int i=0; i<np; i++) std::cout << std::setw(4) << i << " " << std::setw(3) << etaIndex[i] << " " << std::setw(3) << phiIndex[i] << " " << depths[i] << " " << std::setw(6) << radiusT[i] << " " << std::setw(6) << radiusH[i] << " " << std::setw(6) << phiH[i] << "\n";

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

void plotBERT(char name[10]="Hit0", char filx[10]="minbias", bool logy=true, 
	      int bin=1, int mode=0) {

  setTDRStyle();
  TCanvas *myc = new TCanvas("myc","",800,600);
  if (logy) gPad->SetLogy(1);
  char file1[50], file2[50], lego1[32], lego2[32];
  sprintf (file1, "%s_QGSP_EMV.root", filx); sprintf (lego1, "QGSP_EMV");
  sprintf (file2, "%s_QGSP_BERT_EMV.root", filx); sprintf (lego2, "QGSP_BERT_EMV");

  std::cout << "Open " << file1 << "\n";
  TFile* File1 = new TFile(file1);
  TDirectory *d1 = (TDirectory*)File1->Get("caloSimHitStudy");
  TH1F* hist1 = (TH1F*) d1->Get(name);
  hist1->Rebin(bin);
  hist1->SetLineStyle(1);
  hist1->SetLineWidth(3);
  hist1->SetLineColor(2);
  std::cout << "Open " << file2 << "\n";
  TFile* File2 = new TFile(file2);
  TDirectory *d2 = (TDirectory*)File2->Get("caloSimHitStudy");
  TH1F* hist2 = (TH1F*) d2->Get(name);
  hist2->Rebin(bin);
  hist2->SetLineStyle(2);
  hist2->SetLineWidth(3);
  hist2->SetLineColor(4);

  if (mode <= 0) {
    hist1->Draw();
    hist2->Draw("sames");
  } else {
    hist2->Draw();
    hist1->Draw("sames");
  }
  
  gPad->Update();
  TPaveStats *st1 = (TPaveStats*)hist1->GetFunction("stats");
  st1->SetTextColor(2);
  st1->SetLineColor(2);
  TPaveStats *st2 = (TPaveStats*)hist2->GetFunction("stats");
  st2->SetTextColor(4);
  st2->SetLineColor(4);
  double x1 = st1->GetX1NDC();
  double y1 = st1->GetY1NDC();
  double x2 = st1->GetX2NDC();
  double y2 = st1->GetY2NDC();
  double xx = x2-x1;
  double yy = y2-y1;
  st2->SetX1NDC(0.95-2*xx);
  st2->SetY1NDC(y1);
  st2->SetX2NDC(x1);
  st2->SetY2NDC(y2);
  gPad->Modified();

  TLegend *leg1 = new TLegend(0.65,0.52,0.90,0.67);
  char head[40];
  if      (filx == "ttbar") sprintf (head, "t#bar{t}");
  else if (filx == "zee")   sprintf (head, "Z#rightarrowe^{+}e^{-}");
  else                      sprintf (head, "Minimum Bias");
  leg1->SetHeader(head); leg1->SetFillColor(0); leg1->SetTextSize(0.03);
  leg1->AddEntry(hist1,lego1,"F");
  leg1->AddEntry(hist2,lego2,"F");
  leg1->Draw();
}

void plotValid(char name[10]="Hit0", char filx[10]="minbias", bool logy=true, 
	       int bin=1, double xmax=-1., int mode=0) {

  setTDRStyle();
  TCanvas *myc = new TCanvas("myc","",800,600);
  if (logy) gPad->SetLogy(1);
  char lego1[32], lego2[32], lego3[32], lego4[32];
  
  sprintf (lego1,"Original 3_1_0"); 
  sprintf (lego2,"3_1_0 + New PhysicsLists");
  sprintf (lego3,"Modified Code (Default)");  
  sprintf (lego4,"Change in Response");

  TFile *File[4];
  TH1F  *hist[4];
  double xmin=0;
  for (int i=0; i<4; i++) {
    char file[50];
    sprintf (file, "%s_QGSP_BERT_EML%d.root", filx, i); 
    std::cout << "Open " << file << "\n";
    File[i] = new TFile(file);
    TDirectory *dir = (TDirectory*)File[i]->Get("caloSimHitStudy");
    hist[i] = (TH1F*) dir->Get(name);
    hist[i]->Rebin(bin);
    hist[i]->SetLineStyle(i+1);
    hist[i]->SetLineWidth(3);
    hist[i]->SetLineColor(i+1);
    xmin = hist[i]->GetXaxis()->GetXmin();
    if (xmax > 0) hist[i]->GetXaxis()->SetRangeUser(xmin,xmax);
  }

  if (mode < 0 || mode > 3) mode = 0;
  hist[mode]->Draw();
  for (int i=0; i<4; i++) {
    if (i != mode) hist[i]->Draw("sames");
  }
  
  gPad->Update();
  TPaveStats *st[4];
  for (int i=0; i<4; i++) {
    st[i] = (TPaveStats*)hist[i]->GetFunction("stats");
    if (st[i]) {
      st[i]->SetTextColor(i+1);
      st[i]->SetLineColor(i+1);
    }
  }
  double x1 = st[0]->GetX1NDC();
  double y1 = st[0]->GetY1NDC();
  double x2 = st[0]->GetX2NDC();
  double y2 = st[0]->GetY2NDC();
  double xx = x2-x1;
  double yy = y2-y1;
  if (st[1]) {
    st[1]->SetX1NDC(0.95-2*xx);
    st[1]->SetY1NDC(y1);
    st[1]->SetX2NDC(x1);
    st[1]->SetY2NDC(y2);
  }
  if (st[2]) {
    st[2]->SetX1NDC(x1);
    st[2]->SetY1NDC(0.95-2*yy);
    st[2]->SetX2NDC(x2);
    st[2]->SetY2NDC(y1);
  }
  if (st[3]) {
    st[3]->SetX1NDC(0.95-2*xx);
    st[3]->SetY1NDC(0.95-2*yy);
    st[3]->SetX2NDC(x1);
    st[3]->SetY2NDC(y1);
  }
  gPad->Modified();

  TLegend *leg1 = new TLegend(0.10,0.75,0.45,0.90);
  char head[40];
  if      (filx == "ttbar") sprintf (head, "t#bar{t}");
  else if (filx == "zee")   sprintf (head, "Z#rightarrowe^{+}e^{-}");
  else                      sprintf (head, "Minimum Bias");
  leg1->SetHeader(head); leg1->SetFillColor(0); leg1->SetTextSize(0.03);
  leg1->AddEntry(hist[0],lego1,"F");
  leg1->AddEntry(hist[1],lego2,"F");
  leg1->AddEntry(hist[2],lego3,"F");
  leg1->AddEntry(hist[3],lego4,"F");
  leg1->Draw();
}

void plotHF(char name[10]="Dist", char beam[10]="elec", char ener[10]="100",
	    char tag[10]="SL", bool logy=true, int bin=10) {

  setTDRStyle();
  TCanvas *myc = new TCanvas("myc","",800,600);
  tdrStyle->SetOptStat(0);
  if (logy) gPad->SetLogy(1);
  char file[50], head[60], lego1[32], lego2[32], name1[12], name2[12];
  sprintf (file, "%s%s_%s.root", beam, ener, tag);
  if (beam == "elec")   sprintf (lego1, "electron");
  else                  sprintf (lego1, "pion");
  if      (tag == "SL") sprintf (head, "%s GeV %s with Shower Library", ener, lego1);
  else if (tag == "PSL")sprintf (head, "%s GeV %s with parametrization using SL", ener, lego1);
  else                  sprintf (head, "%s GeV %s with parametrization using SL (new)", ener, lego1);
  sprintf (name1, "HCalSD%s5", name); sprintf (lego1, "Long  Fibre");
  sprintf (name2, "HCalSD%s6", name); sprintf (lego2, "Short Fibre");

  std::cout << "Open " << file << "\n";
  TFile* File1 = new TFile(file);
  TDirectory *d1 = (TDirectory*)File1->Get("g4SimHits");
  TH1F* hist1 = (TH1F*) d1->Get(name1);
  TH1F* hist2 = (TH1F*) d1->Get(name2);
  hist1->Rebin(bin); hist1->SetLineStyle(1); hist1->SetLineWidth(3); hist1->SetLineColor(2);
  hist2->Rebin(bin); hist2->SetLineStyle(2); hist2->SetLineWidth(3); hist2->SetLineColor(4);

  if (logy) hist1->SetMinimum(1);
  else      hist1->SetMinimum(0);
  hist1->Draw(); hist2->Draw("sames");

  TLegend *leg1 = new TLegend(0.35,0.75,0.90,0.90);
  leg1->SetHeader(head); leg1->SetFillColor(0); leg1->SetTextSize(0.03);
  leg1->SetBorderSize(1);
  leg1->AddEntry(hist1,lego1,"F");
  leg1->AddEntry(hist2,lego2,"F");
  leg1->Draw();
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
  tdrStyle->SetOptFile(11);
  tdrStyle->SetOptStat(11111111);

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
