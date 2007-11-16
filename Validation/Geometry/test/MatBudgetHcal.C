// include files
#include <iostream>
#include <iomanip>
#include <fstream>
#include <cmath>
#include <vector>

const int nlaymax = 25;
const int nbinmax = 41;
double mean[nlaymax][nbinmax],  diff[nlaymax][nbinmax]; 

double towLow[41]    = { 0.000,  0.087,  0.174,  0.261,  0.348, 
			 0.435,  0.522,  0.609,  0.696,  0.783,
			 0.870,  0.957,  1.044,  1.131,  1.218,
			 1.305,  1.392,  1.479,  1.566,  1.653,
			 1.740,  1.830,  1.930,  2.043,  2.172, 
			 2.322,  2.500,  2.650,  2.853,  2.964, 
			 3.139,  3.314,  3.489,  3.664,  3.839, 
			 4.013,  4.191,  4.363,  4.538,  4.716, 
			 4.889};
double towHigh[41]   = { 0.087,  0.174,  0.261,  0.348,  0.435,
			 0.522,  0.609,  0.696,  0.783,  0.870,
			 0.957,  1.044,  1.131,  1.218,  1.305,
			 1.392,  1.479,  1.566,  1.653,  1.740,
			 1.830,  1.930,  2.043,  2.172,  2.322,
			 2.500,  2.650,  3.000,  2.964,  3.139,
			 3.314,  3.489,  3.664,  3.839,  4.013,
			 4.191,  4.363,  4.538,  4.716,  4.889,
			 5.191};

int colorLayer[25] = {152, 107,   9,  30,  34,  38,  14,  40,  41,  42,
		       45,  46,  48,  49,  37,  28,   4, 154, 104,  50,
		        3,   5,   6, 156, 159};

void standardPlot (TString fileName="matbdg_HCAL.root", 
		   TString outputFileName="hcal.txt") {

  etaPhiPlot (fileName, "IntLen", 0, 19, 1, true,  4.8);
  etaPhiPlot (fileName, "IntLen", 0, 17, 1, false, -1.);
  etaPhiPlot (fileName, "RadLen", 0, 19, 1, true,  4.8);
  etaPhiPlot (fileName, "RadLen", 0, 17, 1, false, -1);
  etaPhiPlot (fileName, "StepLen",0, 19, 1, true,  -1);
  etaPhiPlot (fileName, "StepLen",0, 18, 1, false, -1);
  plotDiff   (fileName, "IntLen");
  plotDiff   (fileName, "RadLen");
  printTable (fileName, outputFileName);
  etaPhi2DPlot(fileName, "IntLen", 0, 19, 1)
  etaPhi2DPlot(fileName, "RadLen", 0, 19, 1)
}

void etaPhiPlot(TString fileName="matbdg_HCAL.root", TString plot="IntLen", 
		int ifirst=0, int ilast=19, int drawLeg=1, bool ifEta=true,
		double maxEta=-1) {

  TFile* hcalFile = new TFile(fileName);
  setStyle();

  TString xtit = TString("#eta");
  TString ytit = "none";
  int ymin = 0, ymax = 20, istart = 200;
  double xl = 0.74;
  if (plot.CompareTo("RadLen") == 0) {
    ytit = TString("HCal Material Budget X_{0}");
    ymin = 0;  ymax = 200; istart = 100;
  } else if (plot.CompareTo("StepLen") == 0) {
    ytit = TString("HCal Material Budget (Step Length)");
    ymin = 0;  ymax = 15000; istart = 300; xl = 0.61;
  } else {
    ytit = TString("HCal Material Budget (#lambda)");
    ymin = 0;  ymax = 20; istart = 200;
  }
  if (!ifEta) {
    istart += 400;
    xtit    = TString("#phi"); 
  }
  
  TLegend *leg = new TLegend(xl, 0.60, xl+0.09, 0.90);
  leg->SetBorderSize(1); leg->SetFillColor(10); leg->SetMargin(0.6);
  
  int nplots=0;
  TProfile *prof[nlaymax];
  for (int ii=ilast; ii>=ifirst; ii--) {
    char hname[10], title[50];
    sprintf(hname, "%i", istart+ii);
    prof[nplots] = (TProfile*)hcalFile->Get(hname);
    prof[nplots]->GetXaxis()->SetTitle(xtit);
    prof[nplots]->GetYaxis()->SetTitle(ytit);
    prof[nplots]->GetYaxis()->SetRangeUser(ymin, ymax);
    prof[nplots]->SetLineColor(colorLayer[ii]);
    prof[nplots]->SetFillColor(colorLayer[ii]);
    if (ifEta && maxEta > 0) 
      prof[nplots]->GetXaxis()->SetRangeUser(-maxEta,maxEta);
    sprintf(title, "Layer %d", ii+1);
    leg->AddEntry(prof[nplots], title, "lf");
    nplots++;
  }

  TString cname = "c_" + plot + xtit;
  TCanvas *cc1 = new TCanvas(cname, cname, 700, 400);

  prof[0]->Draw("h");
  for(int i=1; i<nplots; i++)
    prof[i]->Draw("h sames");
  if (drawLeg > 0) leg->Draw("sames");
}

void etaPhi2DPlot(TString fileName="matbdg_HCAL.root", TString plot="IntLen", 
		  int ifirst=0, int ilast=19, int drawLeg=1) {

  TFile* hcalFile = new TFile(fileName);
  setStyle();

  TString xtit = TString("#eta");
  TString ytit = TString("#phi");
  TString ztit = TString("HCal Material Budget (#lambda)");
  int ymin = 0, ymax = 20, istart = 1000;
  double xl = 0.81;
  if (plot.CompareTo("RadLen") == 0) {
    ztit = TString("HCal Material Budget X_{0}");
    ymin = 0;  ymax = 200; istart = 900;
  } else if (plot.CompareTo("StepLen") == 0) {
    ytit = TString("HCal Material Budget (Step Length)");
    ymin = 0;  ymax = 15000; istart = 1100; 
  }
  
  TLegend *leg = new TLegend(xl, 0.60, xl+0.09, 0.90);
  leg->SetBorderSize(1); leg->SetFillColor(10); leg->SetMargin(0.6);
  
  int nplots=0;
  TProfile2D *prof[nlaymax];
  for (int ii=ilast; ii>=ifirst; ii--) {
    char hname[10], title[50];
    sprintf(hname, "%i", istart+ii);
    prof[nplots] = (TProfile2D*)hcalFile->Get(hname);
    prof[nplots]->GetXaxis()->SetTitle(xtit);
    prof[nplots]->GetYaxis()->SetTitle(ytit);
    prof[nplots]->GetZaxis()->SetTitle(ztit);
    prof[nplots]->GetZaxis()->SetRangeUser(ymin, ymax);
    prof[nplots]->SetLineColor(colorLayer[ii]);
    prof[nplots]->SetFillColor(colorLayer[ii]);
    sprintf(title, "Layer %d", ii+1);
    leg->AddEntry(prof[nplots], title, "lf");
    nplots++;
  }

  TString cname = "c_" + plot + xtit + ytit;
  TCanvas *cc1 = new TCanvas(cname, cname, 700, 400);

  prof[0]->Draw("lego fb bb");
  for(int i=1; i<nplots; i++)
    prof[i]->Draw("lego fb bb sames");
  if (drawLeg > 0) leg->Draw("sames");
}

void printTable (TString fileName="matbdg_HCAL.root", 
		 TString outputFileName="hcal.txt",
		 TString inputFileName="None") {

  double radl[nlaymax][nbinmax],  intl[nlaymax][nbinmax]; 
  bool compare = false;
  if (inputFileName != "None") {
    ifstream inp(inputFileName, ios::in);
    cout << "Opens " << inputFileName << "\n";
    if (inp) {
      TString line;
      int     tower;
      double  eta;
      for (int i = 0; i < 23; i++) 
	inp >> line;
      for (int itow=0; itow<nbinmax; itow++) {
	inp >> tower >> eta;
	int laymax=18;
	if (itow > 27)     laymax = 2;
	else if (itow > 3) laymax = 17;
	for (int ilay=0; ilay<laymax; ilay++)
	  inp >> intl[ilay][tower];
      }
      for (int i = 0; i < 23; i++) 
	inp >> line;
      for (int itow=0; itow<nbinmax; itow++) {
	inp >> tower >> eta;
	int laymax=18;
	if (itow > 27)     laymax = 2;
	else if (itow > 3) laymax = 17;
	for (int ilay=0; ilay<laymax; ilay++)
	  inp >> radl[ilay][tower];
      }
      compare = true;
      inp.close();
    }
  }
  std::ofstream os;
  os.open(outputFileName);

  int nbadI=0;
  getDiff (fileName, "IntLen");
  os << "Interaction Length\n" << "==================\n"
     << "Eta Tower/Layer   0      1       2       3       4       5     "
     << "  6       7       8       9     10      11      12      13     "
     << " 14      15      16      17\n";
  for (int itow=0; itow<nbinmax; itow++) {
    os << setw(3)<< itow << setw(7) << setprecision(3) 
       << 0.5*(towLow[itow]+towHigh[itow]);
    int laymax=18;
    if (itow > 27)     laymax = 2;
    else if (itow > 3) laymax = 17;
    for (int ilay=0; ilay<laymax; ilay++) {
      os << setw(8) << setprecision(4) <<  diff[ilay][itow];
      if (compare) {
	double num = (diff[ilay][itow] - intl[ilay][itow]);
	double den = (diff[ilay][itow] + intl[ilay][itow]);
	double dd  = (den == 0.? 0. : 2.0*num/den);
	if (dd > 0.01) {
	  nbadI++;
	  cout << "Lambda::Tower " << setw(3) << itow << " Layer " << setw(3) 
	       << ilay << " Old" << setw(8) << setprecision(4) 
	       << intl[ilay][itow] << " New" << setw(8) << setprecision(4) 
	       << diff[ilay][itow] << " Diff"<< setw(8) << setprecision(4) 
	       << dd << "\n";
	}
      }
    }
    os << "\n";
  }

  int nbadR = 0;
  getDiff (fileName, "RadLen");
  os << "\n\nRadiation Length\n" << "================\n"
     << "Eta Tower/Layer   0      1       2       3       4       5     "
     << "  6       7       8       9     10      11      12      13     "
     << " 14      15      16      17\n";
  for (int itow=0; itow<nbinmax; itow++) {
    os << setw(3)<< itow << setw(7) << setprecision(3) 
       << 0.5*(towLow[itow]+towHigh[itow]);
    int laymax=18;
    if (itow > 27)     laymax = 2;
    else if (itow > 3) laymax = 17;
    for (int ilay=0; ilay<laymax; ilay++) {
      os << setw(8) << setprecision(4) <<  diff[ilay][itow];
      if (compare) {
	double num = (diff[ilay][itow] - radl[ilay][itow]);
	double den = (diff[ilay][itow] + radl[ilay][itow]);
	double dd  = (den == 0.? 0. : 2.0*num/den);
	if (dd > 0.01) {
	  nbadR++;
	  cout << "X0::Tower " << setw(3) << itow << " Layer " << setw(3) 
	       << ilay << " Old" << setw(8) << setprecision(4) 
	       << radl[ilay][itow] << " New" << setw(8) << setprecision(4) 
	       << diff[ilay][itow] << " Diff"<< setw(8) << setprecision(4)
	       << dd << "\n";
	}
      }
    }
    os << "\n";
  }
  os.close();

  cout << "Comparison Results " << nbadI << " discrepancies for Lambda and "
       << nbadR << " discrepancies for X0\n";
}

void plotDiff (TString fileName="matbdg_HCAL.root", TString plot="IntLen") {

  setStyle();
  getDiff (fileName, plot);
  TString xtit = TString("Layer Number");
  TString ytit = TString("HCal Material Budget (#lambda)");
  if (plot.CompareTo("RadLen") == 0) 
    ytit = TString("HCal Material Budget X_{0}");

  TMultiGraph *mg = new TMultiGraph();
  TLegend *leg_mg = new TLegend(.5,.5,.67,.73);
  leg_mg->SetFillColor(10);
  leg_mg->SetBorderSize(1);

  double diff_lay[18],  idx[18];
  for (int ilay=1; ilay<19; ilay++) {
     diff_lay[ilay-1] = diff[ilay][0];
     idx[ilay-1] = ilay;
  }
  TGraph *gr_eta1 = new TGraph(18, idx, diff_lay);
  gr_eta1->SetMarkerStyle(20);
  gr_eta1->SetMarkerColor(2);
  gr_eta1->SetLineColor(2);
  mg->Add(gr_eta1,  "pc");
  leg_mg->AddEntry(gr_eta1, "HB #eta = 1");

  for (int ilay=1; ilay<19; ilay++) 
    diff_lay[ilay-1] = diff[ilay][6];
  TGraph *gr_eta7 = new TGraph(17, idx, diff_lay);
  gr_eta7->SetMarkerStyle(22);
  gr_eta7->SetMarkerColor(4);
  gr_eta7->SetLineColor(4);
  mg->Add(gr_eta7,  "pc");
  leg_mg->AddEntry(gr_eta7, "HB #eta = 7");

  for (int ilay=1; ilay<19; ilay++) 
    diff_lay[ilay-1] = diff[ilay][12];
  TGraph *gr_eta13 = new TGraph(17, idx, diff_lay);
  gr_eta13->SetMarkerStyle(29);
  gr_eta13->SetMarkerColor(kGreen+100);
  gr_eta13->SetLineColor(kGreen+100);
  mg->Add(gr_eta13, "pc");
  leg_mg->AddEntry(gr_eta13,"HB #eta = 13");

  for (int ilay=1; ilay<19; ilay++)
    diff_lay[ilay-1] = diff[ilay][19];
  TGraph *gr_eta19 = new TGraph(17, idx, diff_lay);
  gr_eta19->SetMarkerStyle(24);
  gr_eta19->SetMarkerColor(kCyan+100);
  gr_eta19->SetLineColor(kCyan+100);
  mg->Add(gr_eta19, "pc");
  leg_mg->AddEntry(gr_eta19,"HE #eta = 19");

  for(int ilay=1; ilay<19; ilay++) 
    diff_lay[ilay-1] = diff[ilay][25];
  TGraph *gr_eta25 = new TGraph(17, idx, diff_lay);
  gr_eta25->SetMarkerStyle(26);
  gr_eta25->SetMarkerColor(kCyan+100);
  gr_eta25->SetLineColor(kCyan+100);
  mg->Add(gr_eta25, "pc");
  leg_mg->AddEntry(gr_eta25,"HE #eta = 25");

  TString cname = "c_diff_" + plot;
  TCanvas *cc2  = new TCanvas(cname, cname, 700, 400);
  mg->Draw("a");
  mg->GetXaxis()->SetTitle(xtit);
  mg->GetYaxis()->SetTitle(ytit);
  leg_mg->Draw("same");
}

void getDiff (TString fileName="matbdg_HCAL.root", TString plot="IntLen") {

  TFile* hcalFile = new TFile(fileName);

  int    istart = 200;
  if (plot.CompareTo("RadLen") == 0) {
    istart = 100;
  } else if (plot.CompareTo("StepLen") == 0) {
    istart = 300; 
  }

  for (int ilay=0; ilay<19; ilay++) {
    char hname[10];
    sprintf(hname, "%i", istart+ilay);
    TProfile *prof = (TProfile*)hcalFile->Get(hname);
    int      nbins = prof->GetNbinsX();
    for (int itow=0; itow<nbinmax; itow++) {
      double ent = 0, value = 0;
      for (int ii=0; ii<nbins; ii++) {
	double xl = prof->GetBinLowEdge(ii+1);
	double xu = prof->GetBinWidth(ii+1);
	if (xl >= 0) { xu += xl;}
	else         { double tmp = xu; xu =-xl; xl = xu-tmp;}
	double cont = (prof->GetBinContent(ii+1));
	double dx   = 1;
	if (cont > 0) {
	  if (xl >= towLow[itow] && xu <= towHigh[itow]) {
	    ent += dx; value += cont;
	  } else if (xl < towLow[itow] && xu > towLow[itow]) {
	    dx   = (xu-towLow[itow])/(xu-xl);
	    ent += dx; value += dx*cont;
	  } else if (xu > towHigh[itow] && xl < towHigh[itow]) {
	    dx   = (towHigh[itow]-xl)/(xu-xl);
	    ent += dx; value += dx*cont;
	  }
	}
      }
      if (ent > 0) mean[ilay][itow] = value/ent;
      else         mean[ilay][itow] = 0.;
    }
  }
  for (int itow=0; itow<nbinmax; itow++) {
    if (itow > 4) mean[18][itow] = 0;
    diff[0][itow] = mean[0][itow];
  }
  for (int ilay=1; ilay<19; ilay++) {
    for (int itow=0; itow<nbinmax; itow++) {
      diff[ilay][itow] = mean[ilay][itow]-mean[ilay-1][itow];
      if (diff[ilay][itow] < 0) diff[ilay][itow] = 0;
    }
  }
  /*
  for (int ilay=17; ilay<19; ilay++) {
    for (int itow=0; itow<nbinmax; itow++) {
      cout << ilay << " " << itow << " " << mean[ilay][itow] << " " << diff[ilay][itow] << "\n";
    }
  }
  */
}

void setStyle () {

  gStyle->SetCanvasBorderMode(0); gStyle->SetCanvasColor(kWhite);
  gStyle->SetPadColor(kWhite);    gStyle->SetFrameBorderMode(0);
  gStyle->SetFrameBorderSize(1);  gStyle->SetFrameFillColor(0);
  gStyle->SetFrameFillStyle(0);   gStyle->SetFrameLineColor(1);
  gStyle->SetFrameLineStyle(1);   gStyle->SetFrameLineWidth(1);
  gStyle->SetOptStat(0);          gStyle->SetLegendBorderSize(1);
  gStyle->SetOptTitle(0);         gStyle->SetTitleOffset(2.5,"Y");

}

