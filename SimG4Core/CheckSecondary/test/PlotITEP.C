#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>
#include <list>

#include <math.h>
#include <vector>

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

const int models=4, nEnergy0=8, nEnergy1=7, nEnergy2=10, nEnergy3=8;
//const int models=5, nEnergy0=8, nEnergy1=7, nEnergy2=10, nEnergy3=8;
std::string Models[4]      = {"LEP", "RPG", "FTF", "Bertini"};
//std::string Models[5]      = {"LEP", "Binary", "FTFP", "QGSC", "Bertini"};
//std::string Models[5]      = {"LEP", "QGSP", "FTFP", "QGSC", "Bertini"};
int         colModel[5]    = {1, 2, 6, 3, 7};
int         symbModel[5]   = {24, 29, 25, 27, 26};
double      keproton[4]    = {0.09, 0.15, 0.19, 0.23};
double      keneutron[4]   = {0.07, 0.11, 0.15, 0.17};
double      energyScan0[8] = {5.7, 6.2, 6.5, 7.0, 7.5, 8.2, 8.5, 9.0};
double      energyScan1[7] = {6.0, 6.5, 7.0, 7.5, 8.2, 8.5, 9.0};
double      energyScan2[10]= {1.0, 2.0, 3.0,  5.0, 6.0, 6.5,
			      7.0, 7.5, 8.25, 9.0};
bool        debug=false;

void plotData(char element[2], char ene[6], char angle[6], 
	      char beam[8]="proton", char particle[8]="neutron", int save=0) {

  char file1[50], file2[50];
  sprintf (file1, "itep/%s/%s/%s%sGeV%sdeg.dat",  beam, particle, element, ene, angle);
  sprintf (file2, "itep/%s/%s/%s%sGeV%sdeg.dat2", beam, particle, element, ene, angle);
  cout << file1 << "  " << file2 << "\n";
  ifstream infile1, infile2;
  infile1.open(file1);
  infile2.open(file2);
  
  Int_t   q1, i=0;
  Float_t m1, r1, x1[30], y1[30], stater1[30], syser1[30];
  infile1 >> m1 >> r1 >> q1;
  for (i=0; i<q1; i++) infile1 >> x1[i] >> y1[i] >> stater1[i] >> syser1[i];

  Int_t   q2, n=0;
  Float_t m2, r2, x2[30], y2[30], stater2[30], syser2[30];
  infile2 >> m2 >> r2 >> q2;
  for (i=0; i<q2; i++) infile2 >> x2[i] >> y2[i] >> stater2[i] >> syser2[i];

  Float_t x[30], chi[30], dif[30], edif[30], chi2=0.0, diff=0., dify=0. ;
  for (i=0; i<q1; i++) {
    for (int j=0; j<q2; j++) {
      double dx = ((x1[i]-x2[j]) >= 0 ? (x1[i]-x2[j]): -(x1[i]-x2[j]));
      if (dx < 0.0001) {
	double d1 = stater1[i]/y1[i];
	double d2 = stater2[j]/y2[j];
	double dd = sqrt(stater1[i]*stater1[i] + stater2[j]*stater2[j]);
	x[n]    = x1[i];
	dif[n]  = 200.*(y1[i]-y2[j])/(y1[i]+y2[j]);
	double da = (dif[n] > 0 ? dif[n]: -dif[n]);
	edif[n] = da*sqrt(d1*d1+d2*d2);
	chi[n]  = pow(((y1[i]-y2[j])/dd),2);
	diff   += da;
	chi2   += chi[n];
	dify   += dif[n];
	n++;
	break;
      }
    }
  }

  for (i=0; i<n; i++)
    std::cout << "Data " << i << " E " << x[i] << " Difference " << dif[i] << " +- " << edif[i] << " Chi2 " << chi[i] << "\n";

  dify /= n;
  std::cout << "Chi-Square = " << chi2 << "/" << n << " Mean Difference = " << diff/n << "\n";

  setStyle(); gStyle->SetOptStat(1111);

  char name[30], title[60];
  sprintf (name, "%s%sGeV%sdeg", element, ene, angle);
  sprintf (title, "%s from p+%s at %s GeV (#theta = %s^{o})", particle, element, ene, angle);
  TCanvas* c1  = new TCanvas("c1",name,400,300); c1->SetLeftMargin(0.15);
  TGraph*  gr1 = new TGraphErrors(q1,x1,y1,0,stater1);
  gr1->SetTitle(""); gr1->SetMarkerColor(4);  // blue
  gr1->SetMarkerStyle(22);  gr1->SetMarkerSize(1.4); gr1->Draw("ALP");
  gr1->GetXaxis()->SetTitle("Energy (GeV)"); 
  gr1->GetYaxis()->SetTitle("E#frac{d^{3}#sigma}{dp^{3}} (mb/GeV^{2})"); 

  TGraph* gr2 = new TGraphErrors(q2,x2,y2,0,stater2);
  gr2->SetMarkerColor(2);  // red
  gr2->SetMarkerStyle(23);  gr2->SetMarkerSize(1.4); gr2->Draw("LP");

  TLegend *leg1 = new TLegend(0.55,0.80,0.90,0.90);
  leg1->SetHeader(title); leg1->SetFillColor(0);
  leg1->SetTextSize(0.04);
  leg1->Draw();

  sprintf (name, "Chi%s%sGeV%sdeg", element, ene, angle);
  TCanvas *c2  = new TCanvas("c2",name,400,300); c2->SetLeftMargin(0.15);
  TGraph  *gr3 = new TGraph(n,x,chi);
  gr3->SetTitle("");  gr3->SetMarkerStyle(22);  gr3->SetMarkerColor(4);
  gr3->GetXaxis()->SetTitle("Energy (GeV)"); gr3->GetYaxis()->SetTitle("#chi^{2}");  
  gr3->Draw("ALP"); gr3->SetMarkerSize(1.4);
  leg1->Draw();

  TGraph*  gr4 = new TGraphErrors(n,x,dif,0,edif);
  gr4->SetTitle("");  gr4->SetMarkerStyle(20); gr4->SetMarkerSize(1.25);
  gr4->SetMarkerColor(6);
  gr4->GetYaxis()->SetRangeUser(-25.,25.);
  gr4->GetXaxis()->SetTitle("Energy (GeV)"); 
  gr4->GetYaxis()->SetTitle("Difference (%)");
  double xmin = gr4->GetXaxis()->GetXmin();
  double xmax = gr4->GetXaxis()->GetXmax();
  if (debug) std::cout << " Xmin " << xmin << " " << xmax << "\n";
  sprintf (name, "Diff%s%sGeV%sdeg", element, ene, angle);
  TCanvas *c3  = new TCanvas("c3",name,400,300); c3->SetLeftMargin(0.15);
  TLine   *line = new TLine(xmin,dify,xmax,dify);
  line->SetLineStyle(2); line->SetLineColor(4);
  gr4->Draw("AP"); line->Draw();
  leg1->Draw();

  if (save > 0) {
    char fname[60];
    sprintf (fname, "%s%sGeV%sdeg_1.eps",  element, ene, angle);
    c1->SaveAs(fname);
    sprintf (fname, "%s%sGeV%sdeg_2.eps",  element, ene, angle);
    c2->SaveAs(fname);
    sprintf (fname, "%s%sGeV%sdeg_3.eps",  element, ene, angle);
    c3->SaveAs(fname);
  }
}

void plotKEx(char ene[6], char angle[6], int first=0, int logy=0, int save=0, 
	     char beam[8]="proton", char particle[8]="proton") {

  setStyle();
  TCanvas *myc = new TCanvas("myc","",800,600); myc->Divide(2,2);

  myc->cd(1); if (logy != 0) gPad->SetLogy(1); gPad->SetLeftMargin(0.15);
  plotKE("C", ene, angle, first, logy, beam, particle);
  myc->cd(2); if (logy != 0) gPad->SetLogy(1); gPad->SetLeftMargin(0.15);
  plotKE("Cu", ene, angle, first, logy, beam, particle);
  myc->cd(3); if (logy != 0) gPad->SetLogy(1); gPad->SetLeftMargin(0.15);
  plotKE("Pb", ene, angle, first, logy, beam, particle);
  myc->cd(4); if (logy != 0) gPad->SetLogy(1); gPad->SetLeftMargin(0.15);
  plotKE("U", ene, angle, first, logy, beam, particle);

  char anglx[6], fname[60];
  int nx = 0;
  for (int i=0; i<6; i++) {
    if (angle[i] != ' ') { anglx[nx] = angle[i]; nx++;}
  }
  if (save != 0) {
    if (save > 0) sprintf (fname, "%sCCuPbUto%sat%sGeV%sdeg.eps", beam, particle, ene, anglx);
    else          sprintf (fname, "%sCCuPbUto%sat%sGeV%sdeg.gif", beam, particle, ene, anglx);
    myc->SaveAs(fname);
  }
}

void plotKE4(char element[2], char ene[6], int first=0, int logy=0, int save=0,
	     char beam[8]="proton", char particle[8]="proton", 
	     char dir[8]="root") {

  setStyle();  
  TCanvas *myc = new TCanvas("myc","",800,600); myc->Divide(2,2);

  myc->cd(1); if (logy != 0) gPad->SetLogy(1); gPad->SetLeftMargin(0.15);
  plotKE(element, ene, " 59.1", first, logy, beam, particle, dir);
  myc->cd(2); if (logy != 0) gPad->SetLogy(1); gPad->SetLeftMargin(0.15);
  plotKE(element, ene, " 89.0", first, logy, beam, particle, dir);
  myc->cd(3); if (logy != 0) gPad->SetLogy(1); gPad->SetLeftMargin(0.15);
  plotKE(element, ene, "119.0", first, logy, beam, particle, dir);
  myc->cd(4); if (logy != 0) gPad->SetLogy(1); gPad->SetLeftMargin(0.15);
  plotKE(element, ene, "159.6", first, logy, beam, particle, dir);

  char fname[40];
  if (save != 0) {
    if (save > 0) sprintf (fname, "%s%sto%sat%sGeV_1.eps", beam, element, particle, ene);
    else          sprintf (fname, "%s%sto%sat%sGeV_1.gif", beam, element, particle, ene);
    myc->SaveAs(fname);
  }

}

void plotKE1(char element[2], char ene[6], char angle[6], int first=0, 
	     int logy=0, int save=0 , char beam[8]="proton", 
	     char particle[8]="proton", char dir[8]="root") {

  setStyle();
  TCanvas *myc = new TCanvas("myc","",800,600); myc->SetLeftMargin(0.15);
  if (logy != 0) gPad->SetLogy(1);
  plotKE(element, ene, angle, first, logy, beam, particle, dir);

  char anglx[6], fname[100];
  int nx = 0;
  for (int i=0; i<6; i++) {
    if (angle[i] != ' ') { anglx[nx] = angle[i]; nx++;}
  }
  if (save != 0) {
    if (save > 0) sprintf (fname, "%s%sto%sat%sGeV%sdeg.eps", beam, element, particle, ene, anglx);
    else          sprintf (fname, "%s%sto%sat%sGeV%sdeg.gif", beam, element, particle, ene, anglx);
    myc->SaveAs(fname);
  }
}

void plotKE(char element[2], char ene[6], char angle[6], int first=0, 
	    int logy=0, char beam[8]="proton", char particle[8]="proton", 
	    char dir[8]="root") {

  char fname[60], list[10], hname[40], titlx[50];
  TH1F *hi[5];
  int i=0, icol=1;
  sprintf (titlx, "Kinetic Energy of %s (GeV)", particle);
  double  ymx0=1, ymi0=100., xlow=0.06, xhigh=0.26;
  if (particle == "neutron") {xlow= 0.0; xhigh=0.20;}
  for (i=0; i<models; i++) {
    sprintf (list, "%s", Models[i].c_str()); icol = colModel[i];
    sprintf (fname, "%s/%s/%s/%s%s%sGeV_1.root", dir, beam, particle, element, list, ene);
    sprintf (hname, "KE0%s%s%sGeV%s", element, list, ene, angle);
    TFile *file = new TFile(fname);
    hi[i] = (TH1F*) file->Get(hname);
    std::cout << "Get " << hname << " from " << fname <<" as " << hi[i] <<"\n";
    int nx = hi[i]->GetNbinsX();
    for (int k=1; k <= nx; k++) {
      double xx = hi[i]->GetBinCenter(k);
      double yy = hi[i]->GetBinContent(k);
      if (xx > xlow && xx < xhigh) {
	if (yy > ymx0) ymx0 = yy;
	if (yy < ymi0 && yy > 0) ymi0 = yy;
      }
    }
    hi[i]->GetXaxis()->SetRangeUser(xlow, xhigh); hi[i]->SetTitle("");
    hi[i]->GetXaxis()->SetTitle(titlx);
    hi[i]->SetLineStyle(1);  hi[i]->SetLineWidth(2); hi[i]->SetLineColor(icol);
    //    file->Close();
  }

  char anglx[6];
  int nx = 0;
  for (i=0; i<6; i++) {
    if (angle[i] != ' ') { anglx[nx] = angle[i]; nx++;}
  }
  sprintf (fname, "itep/%s/%s/%s%sGeV%sdeg.dat", beam, particle, element, ene, anglx);
  std::cout << "Reads data from file " << fname << "\n";
  ifstream infile;
  infile.open(fname);
  
  int     q1;
  float   m1, r1, x1[30], y1[30], stater1[30], syser1[30];
  infile >> m1 >> r1 >> q1;
  for (i=0; i<q1; i++) {
    infile >> x1[i] >> y1[i] >> stater1[i] >> syser1[i];
    syser1[i] *= y1[i];
    double err = sqrt(syser1[i]*syser1[i]+stater1[i]*stater1[i]);
    stater1[i] = err;
    if (y1[i]+stater1[i] > ymx0) ymx0 = y1[i]+stater1[i];    
    if (y1[i]-stater1[i] < ymi0 && y1[i]-stater1[i] > 0) ymi0=y1[i]-stater1[i];
    if (debug) std::cout << i << " " << x1[i] << " " << y1[i] << " " << stater1[i] << "\n";
  }
  TGraph*  gr1 = new TGraphErrors(q1,x1,y1,0,stater1);
  gr1->SetMarkerColor(4);  gr1->SetMarkerStyle(22);
  gr1->SetMarkerSize(1.6);

  if (logy == 0) {ymx0 *= 1.5; ymi0 *= 0.8;}
  else           {ymx0 *=10.0; ymi0 *= 0.2; }
  for (i = 0; i<models; i++) {
    if (debug) std::cout << "Model " << i << " " << hi[i] << " " << ymi0 << " " << ymx0 << "\n";
    hi[i]->GetYaxis()->SetRangeUser(ymi0,ymx0);
  }

  hi[first]->GetYaxis()->SetTitleOffset(1.6);
  hi[first]->Draw();
  for (i=0; i<models; i++) {
    if (i != first) hi[i]->Draw("same");
  }
  gr1->Draw("p");

  TLegend *leg1 = new TLegend(0.42,0.70,0.90,0.90);
  for (i=0; i<models; i++) {
    sprintf (list, "%s", Models[i].c_str()); 
    leg1->AddEntry(hi[i],list,"F");
  }
  char header[120], beamx[8], partx[2];
  if      (beam == "piplus")  sprintf (beamx, "#pi^{+}");
  else if (beam == "piminus") sprintf (beamx, "#pi^{-}");
  else                        sprintf (beamx, "p");
  if      (particle == "neutron") sprintf (partx, "n");
  else                            sprintf (partx, "p");
  sprintf (header,"%s+%s #rightarrow %s+X at %s GeV (#theta = %s^{o})", beamx, element, partx, ene, angle);
  leg1->SetHeader(header); leg1->SetFillColor(0);
  leg1->SetTextSize(0.04);
  leg1->Draw("same");
  if (debug) std::cout << "End\n";
}

void plotCT4(char element[2], char ene[6], int first=0, int scan=1, int logy=0,
	     int save=0, char beam[8]="proton", char particle[8]="proton", 
	     char dir[8]="root") {

  setStyle();
  TCanvas *myc = new TCanvas("myc","",800,600); myc->Divide(2,2);

  for (int i=0; i<4; i++) {
    myc->cd(i+1); if (logy != 0) gPad->SetLogy(1); gPad->SetLeftMargin(0.15);
    double ke = keproton[i];
    if (particle == "neutron") ke = keneutron[i];
    plotCT(element, ene, ke, first, scan, logy, beam, particle, dir); 
  }

  char fname[40];
  if (save != 0) {
    if (save > 0) sprintf (fname, "%s%sto%sat%sGeV_2.eps", beam, element, particle, ene);
    else          sprintf (fname, "%s%sto%sat%sGeV_2.gif", beam, element, particle, ene);
    myc->SaveAs(fname);
  }
}

void plotCT1(char element[2], char ene[6], double ke, int first=0, int scan=1,
	     int logy=0, int save=0, char beam[8]="proton", 
	     char particle[8]="proton", char dir[8]="root") {

  setStyle();
  TCanvas *myc = new TCanvas("myc","",800,600); myc->SetLeftMargin(0.15);
  if (logy != 0) gPad->SetLogy(1);
  plotCT(element, ene, ke, first, scan, logy, beam, particle, dir);

  char fname[40];
  if (save != 0) {
    if (save > 0) sprintf (fname, "%s%sto%sat%sGeV%4.2fGeV.eps", beam, element, particle, ene, ke);
    else          sprintf (fname, "%s%sto%sat%sGeV%4.2fGeV.gif", beam, element, particle, ene, ke);
    myc->SaveAs(fname);
  }
}

void plotCT(char element[2], char ene[6], double ke, int first=0, int scan=1,
	    int logy=0, char beam[8]="proton", char particle[8]="proton", 
	    char dir[8]="root") {

  static double pi  = 3.1415926;
  static double deg = pi/180.; 
  if (debug) std::cout << "Scan " << scan;
  std::vector<double> angles = angleScan(scan);
  int    nn = (int)(angles.size());
  if (debug) std::cout << " gives " << nn << " angles\n";

  char fname[40], list[10], hname[40];
  TH1F *hi[5];
  int i=0, icol=1;
  double  ymx0=1, ymi0=100., xlow=-1.0, xhigh=1.0;
  for (i=0; i<models; i++) {
    sprintf (list, "%s", Models[i].c_str()); icol = colModel[i];
    sprintf (fname, "%s/%s/%s/%s%s%sGeV_1.root", dir, beam, particle, element, list, ene);
    sprintf (hname, "CT0%s%s%sGeV%4.2f", element, list, ene, ke);
    TFile *file = new TFile(fname);
    hi[i] = (TH1F*) file->Get(hname);
    if (debug) std::cout << "Get " << hname << " from " << fname <<" as " << hi[i] <<"\n";
    int nx = hi[i]->GetNbinsX();
    for (int k=1; k <= nx; k++) {
      double xx = hi[i]->GetBinCenter(k);
      double yy = hi[i]->GetBinContent(k);
      if (xx > xlow && xx < xhigh) {
	if (yy > ymx0)           ymx0 = yy;
	if (yy < ymi0 && yy > 0) ymi0 = yy;
      }
    }
    if (debug) std::cout << "Y limit " << ymi0 << " " << ymx0 << " after " << i;
    hi[i]->GetXaxis()->SetRangeUser(xlow, xhigh); hi[i]->SetTitle("");
    hi[i]->SetLineStyle(1);  hi[i]->SetLineWidth(2); hi[i]->SetLineColor(icol);
    //    file->Close();
  }

  int     q1, kk0=0;
  float   m1, r1, x1[30], y1[30], stater1[30], syser1[30];

  for (int kk=0; kk<nn; kk++) {
    char angle[6], anglx[6];
    sprintf (angle, "%5.1f", angles[kk]);
    int nx = 0;
    for (i=0; i<6; i++) {
      if (angle[i] != ' ') { anglx[nx] = angle[i]; nx++;}
    }
    sprintf (fname, "itep/%s/%s/%s%sGeV%sdeg.dat", beam, particle, element, ene, anglx);
    ifstream infile;
    infile.open(fname);
  
    infile >> m1 >> r1 >> q1;
    for (i=0; i<q1; i++) {
      float xx1, yy1, stater, syser;
      infile >> xx1 >> yy1 >> stater >> syser;
      if (xx1 > ke-0.001 && xx1 < ke+0.001) {
	x1[kk0] = cos(deg*angles[kk]);
	y1[kk0] = yy1; stater1[kk0] = stater; syser1[kk0] = syser;
	syser *= yy1;
	double err = sqrt(syser*syser+stater*stater);
	stater1[kk0] = err; 
	if (y1[kk0]+stater1[kk0] > ymx0) ymx0 = y1[kk0]+stater1[kk0];
	if (y1[kk0]-stater1[kk0] < ymi0 && y1[kk0]-stater1[kk0] > 0) ymi0 = y1[kk0]-stater1[kk0];
	kk0++;
      }
    }
    infile.close();
    if (debug) std::cout << kk << " File " << fname << " X " << x1[kk] << " Y " << y1[kk] << " DY " << stater1[kk] << "\n";
  }

  TGraph*  gr1 = new TGraphErrors(kk0,x1,y1,0,stater1);
  gr1->SetMarkerColor(4);  gr1->SetMarkerStyle(22);
  gr1->SetMarkerSize(1.6);

  if (logy == 0) {ymx0 *= 1.5; ymi0 *= 0.8;}
  else           {ymx0 *=10.0; ymi0 *= 0.2; }
  for (i = 0; i<models; i++)
    hi[i]->GetYaxis()->SetRangeUser(ymi0,ymx0);
  
  hi[first]->GetYaxis()->SetTitleOffset(1.6);
  hi[first]->Draw();
  for (i=0; i<models; i++) {
    if (i != first)  hi[i]->Draw("same");
  }
  gr1->Draw("p");

  TLegend *leg1 = new TLegend(0.15,0.70,0.62,0.90);
  for (i=0; i<models; i++) {
    sprintf (list, "%s", Models[i].c_str());
    leg1->AddEntry(hi[i],list,"F");
  }
  char header[80], beamx[8], partx[2];
  if      (beam == "piplus")  sprintf (beamx, "#pi^{+}");
  else if (beam == "piminus") sprintf (beamx, "#pi^{-}");
  else                        sprintf (beamx, "p");
  if      (particle == "neutron") sprintf (partx, "n");
  else                            sprintf (partx, "p");
  sprintf (header, "%s+%s #rightarrow %s+X at %s GeV (%4.2f GeV)", beamx, element, partx, ene, ke);
  leg1->SetHeader(header); leg1->SetFillColor(0);
  leg1->SetTextSize(0.04);
  leg1->Draw();

}

void plotBE4(char element[2], int logy=0, int scan=1, int save=0, 
	     char beam[8]="proton", char particle[8]="proton", 
	     char dir[8]="root") {

  setStyle();
  TCanvas *myc = new TCanvas("myc","",800,600); myc->Divide(2,2);

  myc->cd(1); if (logy != 0) gPad->SetLogy(1); gPad->SetLeftMargin(0.15);
  plotBE(element, " 59.1", 0.11, logy, scan, beam, particle, dir);
  myc->cd(2); if (logy != 0) gPad->SetLogy(1); gPad->SetLeftMargin(0.15);
  plotBE(element, " 59.1", 0.21, logy, scan, beam, particle, dir);
  myc->cd(3); if (logy != 0) gPad->SetLogy(1); gPad->SetLeftMargin(0.15);
  plotBE(element, "119.0", 0.11, logy, scan, beam, particle, dir);
  myc->cd(4); if (logy != 0) gPad->SetLogy(1); gPad->SetLeftMargin(0.15);
  plotBE(element, "119.0", 0.21, logy, scan, beam, particle, dir);

  char fname[40];
  if (save != 0) {
    if (save > 0) sprintf (fname, "%s%sto%s_1.eps", beam, element, particle);
    else          sprintf (fname, "%s%sto%s_1.gif", beam, element, particle);
    myc->SaveAs(fname);
  }
}

void plotBE1(char element[2], char angle[6], double ke, int logy=0, int scan=1,
	     int save=0, char beam[8]="proton", char particle[8]="proton", 
	     char dir[8]="root") {

  setStyle();
  TCanvas *myc = new TCanvas("myc","",800,600); myc->SetLeftMargin(0.15);
  if (logy != 0) gPad->SetLogy(1);
  plotBE(element, angle, ke, logy, scan, beam, particle, dir);

  char anglx[6], fname[40];
  int i=0, nx=0;
  for (i=0; i<6; i++) {
    if (angle[i] != ' ') { anglx[nx] = angle[i]; nx++;}
  }
  if (save != 0) {
    if (save>0) sprintf (fname, "%s%sto%sat%sdeg%4.2fGeV.eps", beam, element, particle, anglx, ke);
    else        sprintf (fname, "%s%sto%sat%sdeg%4.2fGeV.gif", beam, element, particle, anglx, ke);
    myc->SaveAs(fname);
  }
}

void plotBE(char element[2], char angle[6], double ke, int logy=0, int scan=1,
	    char beam[8]="proton", char particle[8]="proton", 
	    char dir[8]="root") {

  double ene[15];
  int    nene=0;
  if (scan == 0) {
    nene = nEnergy0;
    for (int i=0; i<nene; i++) ene[i] = energyScan0[i];
  } else if (scan <= 1) {
    nene = nEnergy1;
    for (int i=0; i<nene; i++) ene[i] = energyScan1[i];
  } else if (scan == 2) {
    nene = nEnergy2;
    for (int i=0; i<nene; i++) ene[i] = energyScan2[i];
  } else {
    nene = nEnergy3;
    for (int i=0; i<nene; i++) ene[i] = energyScan2[i];
  }
 
  char anglx[6];
  int i=0, nx=0;
  for (i=0; i<6; i++) {
    if (angle[i] != ' ') { anglx[nx] = angle[i]; nx++;}
  }

  TGraph *gr[4];
  char fname[40], list[10], hname[40];
  int j=0, icol=1, ityp=20;
  double  ymx0=1, ymi0=10000., xmi=5.0, xmx=10.0;
  if (scan > 1) { 
    xmi = 0.5;
    xmx = 9.5;
  }
  for (i=0; i<models; i++) {
    sprintf (list, "%s", Models[i].c_str()); 
    icol = colModel[i]; ityp = symbModel[i];
    double yt[15];
    for (j=0; j<nene; j++) {
      sprintf (fname, "%s/%s/%s/%s%s%3.1fGeV_1.root", dir, beam, particle, element, list, ene[j]);
      sprintf (hname, "KE0%s%s%3.1fGeV%s", element, list, ene[j], angle);
      TFile *file = new TFile(fname);
      TH1F *hi = (TH1F*) file->Get(hname);
      if (hi == 0) {
	sprintf (hname, "KE0%s%s%4.2fGeV%s", element, list, ene[j], angle);
	hi = (TH1F*) file->Get(hname);
      }
      if (debug) std::cout << "Get " << hname << " from " << fname <<" as " << hi <<"\n";
      int    nk=0, nx = hi->GetNbinsX();
      double yy0=0;
      for (int k=1; k <= nx; k++) {
	double xx0 = hi->GetBinCenter(k);
	if (xx0 > ke-0.01 && xx0 < ke+0.01) {
	  yy0 += hi->GetBinContent(k);
	  nk++;
	}
      }
      if (nk > 0 )                 yy0 /= nk;
      if (yy0 > ymx0)              ymx0 = yy0;
      if (yy0 < ymi0 && yy0 > 0.1) ymi0 = yy0;
      if (debug) std::cout << hname << " # " << nk << " Y " << yy0 << " min " << ymi0 << " max " << ymx0 << "\n";
      yt[j] = yy0;
      file->Close();
    }
    gr[i] = new TGraph(nene, ene, yt); gr[i]->SetMarkerSize(1.2);
    gr[i]->SetTitle(list); gr[i]->SetLineColor(icol); 
    gr[i]->SetLineStyle(i+1); gr[i]->SetLineWidth(2);
    gr[i]->SetMarkerColor(icol);  gr[i]->SetMarkerStyle(ityp); 
    gr[i]->GetXaxis()->SetTitle("Beam Energy (GeV)");
    if (debug) {
      std::cout << "Graph " << i << " with " << nene << " points\n";
      for (j=0; j<nene; j++) std::cout << j << " x " << ene[j] << " y " << yt[j] << "\n";
    }
  }

  double ye[15], dy[15];
  for (j=0; j<nene; j++) {
    sprintf (fname, "itep/%s/%s/%s%3.1fGeV%sdeg.dat", beam, particle, element, ene[j], anglx);
    if (debug) std::cout << "Reads data from file " << fname << "\n";
    ifstream infile;
    infile.open(fname);
  
    int     q1;
    float   m1, r1, xx, yy, stater, syser;
    infile >> m1 >> r1 >> q1;
    for (i=0; i<q1; i++) {
      infile >> xx >> yy >> stater >> syser;
      if (xx > ke-0.01 && xx < ke+0.01) {
	ye[j] = yy;
	syser *= yy;
	double err = sqrt(syser*syser+stater*stater);
	dy[j] = err;
      }
    }
    infile.close();
    if (ye[j]+dy[j] > ymx0) ymx0 = ye[j]+dy[j];
    if (ye[j]-dy[j] < ymi0 && ye[j]-dy[j] > 0) ymi0 = ye[j]-dy[j];
  }
  if (debug) {
    std::cout << "Graph Data with " << nene << " points\n";
    for (j=0; j<nene; j++) std::cout << j << " x " << ene[j] << " y " << ye[j] << " +- " << dy[j] << "\n";
  }
  TGraph*  gr1 = new TGraphErrors(nene,ene,ye,0,dy);
  gr1->SetMarkerColor(1);  gr1->SetMarkerStyle(22);
  gr1->SetMarkerSize(1.6);

  if (logy == 0) {
    ymx0 *= 1.8; ymi0 *= 0.8;
  } else {
    ymx0 *= 50.0; ymi0 *= 0.2;
    if (scan > 1) ymx0 *= 4;
  }
  for (i = 0; i<models; i++) {
    gr[i]->GetYaxis()->SetRangeUser(ymi0,ymx0);
    gr[i]->GetXaxis()->SetRangeUser(xmi,xmx);
  }
  gr1->GetXaxis()->SetRangeUser(xmi,xmx);
  gr1->GetYaxis()->SetRangeUser(ymi0,ymx0);
  gr1->GetXaxis()->SetTitle("Energy (GeV)"); 
  gr1->GetYaxis()->SetTitle("E#frac{d^{3}#sigma}{dp^{3}} (mb/GeV^{2})"); 

  gr1->GetYaxis()->SetTitleOffset(1.6); gr1->SetTitle("");
  gr1->Draw("ap");
  for (i=0; i<models; i++)
    gr[i]->Draw("lp");
  
  TLegend *leg1 = new TLegend(0.35,0.60,0.90,0.90);
  for (i=0; i<models; i++) {
    sprintf (list, "%s", Models[i].c_str());
    leg1->AddEntry(gr[i],list,"LP");
  }
  char header[80], beamx[8], partx[2];
  if      (beam == "piplus")  sprintf (beamx, "#pi^{+}");
  else if (beam == "piminus") sprintf (beamx, "#pi^{-}");
  else                        sprintf (beamx, "p");
  if      (particle == "neutron") sprintf (partx, "n");
  else                            sprintf (partx, "p");
  sprintf (header, "%s+%s #rightarrow %s+X at (KE = %3.1f GeV, #theta = %s^{o})", beamx, element, partx, ke, angle);
  leg1->SetHeader(header); leg1->SetFillColor(0);
  leg1->SetTextSize(0.04);
  leg1->Draw();

}

void setStyle() {

  gStyle->SetCanvasBorderMode(0); gStyle->SetCanvasColor(kWhite);
  gStyle->SetPadColor(kWhite);    gStyle->SetFrameBorderMode(0);
  gStyle->SetFrameBorderSize(1);  gStyle->SetFrameFillColor(0);
  gStyle->SetFrameFillStyle(0);   gStyle->SetFrameLineColor(1);
  gStyle->SetFrameLineStyle(1);   gStyle->SetFrameLineWidth(1);
  gStyle->SetTitleOffset(1.6,"Y");  gStyle->SetOptStat(0);
  gStyle->SetLegendBorderSize(1);

}

std::vector<double> angleScan(int scan) {

  std::vector<double> tmp;
  if (scan <= 1) {
    tmp.push_back(59.1);
    tmp.push_back(89.0);
    tmp.push_back(119.0);
    tmp.push_back(159.6);
  } else {
    tmp.push_back(10.1);
    tmp.push_back(15.0);
    tmp.push_back(19.8);
    tmp.push_back(24.8);
    tmp.push_back(29.5);
    tmp.push_back(34.6);
    tmp.push_back(39.6);
    tmp.push_back(44.3);
    tmp.push_back(49.3);
    tmp.push_back(54.2);
    tmp.push_back(59.1);
    tmp.push_back(64.1);
    tmp.push_back(69.1);
    tmp.push_back(74.1);
    tmp.push_back(79.1);
    tmp.push_back(84.1);
    tmp.push_back(89.0);
    tmp.push_back(98.9);
    tmp.push_back(108.9);
    tmp.push_back(119.0);
    tmp.push_back(129.1);
    tmp.push_back(139.1);
    tmp.push_back(149.3);
    tmp.push_back(159.6);
    tmp.push_back(161.4);
    tmp.push_back(165.5);
    tmp.push_back(169.5);
    tmp.push_back(173.5);
    tmp.push_back(177.0);
  }
  if (debug) {
    std::cout << "Scan " << tmp.size() << " angular regions:\n";
    for (unsigned int i=0; i<tmp.size(); i++) {
      std::cout << tmp[i];
      if (i == tmp.size()-1) std::cout << " degrees\n";
      else                   std::cout << ", ";
    }
  }
  return tmp;
}
