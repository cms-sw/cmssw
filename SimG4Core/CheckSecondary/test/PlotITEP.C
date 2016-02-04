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

//const int modelsITEP=5, modelsBNL=5, nEnergy0=8, nEnergy1=7, nEnergy2=10, nEnergy3=8;
//const int modelsITEP=2, modelsBNL=2, nEnergy0=8, nEnergy1=7, nEnergy2=10, nEnergy3=8;
const int modelsITEP=6, modelsBNL=5, nEnergy0=8, nEnergy1=7, nEnergy2=10, nEnergy3=8;
//const int modelsITEP=4, modelsBNL=4, nEnergy0=8, nEnergy1=7, nEnergy2=10, nEnergy3=8;
//const int modelsITEP=5, modelsBNL=5, nEnergy0=8, nEnergy1=7, nEnergy2=10, nEnergy3=8;
std::string ModelsI[6]      = {"LEP", "FTF", "Bertini", "Binary", "QGSC", "FTFP"};
std::string ModelFilesI[6]  = {"LEP", "FTF", "Bertini", "Binary", "QGSC", "FTFP"};
std::string ModelNamesI[6]  = {"LEP", "FTF-Binary", "Bertini", "Binary", "QGS-CHIPS", "FTF-Preco"};
std::string ModelsB[6]      = {"LEP", "FTF", "Bertini", "QGSP", "QGSC", "FTFP"};
std::string ModelFilesB[6]  = {"LEP", "FTF", "Bertini", "QGSP", "QGSC", "FTFP"};
std::string ModelNamesB[6]  = {"LEP", "FTF-Binary", "Bertini", "QGS-Preco", "QGS-CHIPS", "FTF-Preco"};
//std::string ModelsB[6]      = {"LEP", "FTF", "Bertini", "QGSP", "QGSC", "RPG"};
//std::string ModelFilesB[6]  = {"LEP", "FTF", "Bertini", "QGSP", "QGSC", "RPG"};
//std::string ModelNamesB[6]  = {"LEP", "FTF", "Bertini", "QGSP", "QGSC", "RPG"};
//std::string ModelsI[5]      = {"LEP", "FTF", "Bertini", "Binary", "QGSC"};
//std::string ModelFilesI[5]  = {"LEP", "FTF", "Bertini", "Binary", "QGSC"};
//std::string ModelNamesI[5]  = {"LEP", "FTF", "Bertini", "Binary", "QGSC"};
//std::string ModelsI[6]      = {"Bertini1", "Bertini2", "Bertini3", "Bertini4", "Bertini5", "Bertini6"};
//std::string ModelFilesI[6]  = {"Bertini1", "Bertini2", "Bertini3", "Bertini4", "Bertini5", "Bertini6"};
//std::string ModelNamesI[6]  = {"Bertini (9.2.b01)", "Bertini (9.1.ref08)", "Bertini (+ Coulomb)", "Bertini (9.2)", "Bertini (9.2.ref02)", "Bertini (9.2.p01)"};
//std::string ModelsI[2]      = {"Bertini", "Bertini"};
//std::string ModelFilesI[2]  = {"OldBertini", "NewBertini"};
//std::string ModelNamesI[2]  = {"Bertini (Old)", "Bertini (New)"};
//std::string ModelsI[6]      = {"LEP", "FTF", "Bertini", "QGSP", "QGSC", "RPG"};
//std::string ModelFilesI[6]  = {"LEP", "FTF", "Bertini", "QGSP", "QGSC", "RPG"};
//std::string ModelNamesI[6]  = {"LEP", "FTF", "Bertini", "QGSP", "QGSC", "RPG"};
//std::string ModelsI[4]      = {"LEP", "RPG", "FTF", "Bertini"};
//std::string ModelFilesI[4]  = {"LEP", "RPG", "FTF", "Bertini"};
//std::string ModelNamesI[4]  = {"LEP", "RPG", "FTF", "Bertini"};
//std::string ModelsI[5]      = {"LEP", "Binary", "FTFP", "QGSC", "Bertini"};
//std::string ModelFilesI[5]  = {"LEP", "Binary", "FTFP", "QGSC", "Bertini"};
//std::string ModelNamesI[5]  = {"LEP", "Binary", "FTFP", "QGSC", "Bertini"};
//std::string ModelsI[5]      = {"LEP", "QGSP", "FTFP", "QGSC", "Bertini"};
//std::string ModelFilesI[5]  = {"LEP", "QGSP", "FTFP", "QGSC", "Bertini"};
//std::string ModelNamesI[5]  = {"LEP", "QGSP", "FTFP", "QGSC", "Bertini"};
int         colModel[6]    = {1, 2, 6, 4, 7, 11};
int         symbModel[6]   = {24, 29, 25, 27, 26, 23};
int         stylModel[6]   = {1, 2, 3, 4, 5, 6};
double      keproton[4]    = {0.09, 0.15, 0.19, 0.23};
double      keneutron[4]   = {0.07, 0.11, 0.15, 0.17};
double      energyScan0[8] = {5.7, 6.2, 6.5, 7.0, 7.5, 8.2, 8.5, 9.0};
double      energyScan1[7] = {6.0, 6.5, 7.0, 7.5, 8.2, 8.5, 9.0};
double      energyScan2[10]= {1.0, 2.0, 3.0,  5.0, 6.0, 6.5,
			      7.0, 7.5, 8.25, 9.0};
bool        debug=true;

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
 	     double ymin=-1., char particle[8]="proton", char beam[8]="proton",
	     bool ratio='false', int leg1=1, int leg2=1, char dir[20]=".", 
	     char dird[40]=".", char mark=' ') {

  setStyle();
  TCanvas *myc = new TCanvas("myc","",800,600); myc->Divide(2,2);

  int leg=leg1; if (leg2 == 0) leg=leg2;
  char markf[4]=" ";
  myc->cd(1); if (logy != 0) gPad->SetLogy(1); gPad->SetLeftMargin(0.15);
  if (mark == 'y') sprintf(markf, "(a)");
  if (ratio)
    plotKERatio("C", ene,angle,first,logy,ymin,particle,beam,leg1,dir,dird,markf);
  else
    plotKE("C", ene,angle,first,logy,ymin,particle,beam, leg1, dir,dird,markf);
  myc->cd(2); if (logy != 0) gPad->SetLogy(1); gPad->SetLeftMargin(0.15);
  if (mark == 'y') sprintf(markf, "(b)");
  if (ratio)
    plotKERatio("Cu",ene,angle,first,logy,ymin,particle,beam,leg2,dir,dird,markf);
  else
    plotKE("Cu",ene,angle,first,logy,ymin,particle,beam,leg2,dir,dird,markf);
  myc->cd(3); if (logy != 0) gPad->SetLogy(1); gPad->SetLeftMargin(0.15);
  if (mark == 'y') sprintf(markf, "(c)");
  if (ratio)
    plotKERatio("Pb",ene,angle,first,logy,ymin,particle,beam,leg, dir,dird,markf);
  else
    plotKE("Pb",ene,angle,first,logy,ymin,particle,beam,leg, dir,dird,markf);
  myc->cd(4); if (logy != 0) gPad->SetLogy(1); gPad->SetLeftMargin(0.15);
  if (mark == 'y') sprintf(markf, "(d)");
  if (ratio)
    plotKERatio("U", ene,angle,first,logy,ymin,particle,beam,leg, dir,dird,markf);
  else
    plotKE("U", ene,angle,first,logy,ymin,particle,beam,leg, dir,dird,markf);

  char anglx[6], fname[60];
  int nx = 0;
  for (int i=0; i<6; i++) {
    if (angle[i] != ' ') { anglx[nx] = angle[i]; nx++;}
  }
  if (save != 0) {
    std::string tag=".gif";
    if (ratio) {
      if (save > 0) tag = "R.eps";
      else          tag = "R.gif";
    } else {
      if (save > 0) tag = ".eps";
    }
    sprintf (fname, "%sCCuPbUto%sat%sGeV%sdeg%s", beam, particle, ene, anglx, tag.c_str());
    myc->SaveAs(fname);
  }
}


void plotKEn(char ene[6], int first=0, int logy=0, int save=0, double ymin=-1.,
	     char beam[8]="proton", bool ratio=false, int leg1=1, int leg2=1,
	     char dir[20]=".", char dird[40]=".", char mark=' ') {

  setStyle();  
  TCanvas *myc = new TCanvas("myc","",800,600); myc->Divide(2,2);

  int leg=leg1; if (leg2 == 0) leg=leg2;
  char markf[4]=" ";
  myc->cd(1); if (logy != 0) gPad->SetLogy(1); gPad->SetLeftMargin(0.15);
  if (mark == 'y') sprintf(markf, "(a)");
  if (ratio)
    plotKERatio("C","1.4","119.0",first,logy,ymin,"neutron",beam,leg1,dir,dird,markf);
  else
    plotKE("C","1.4","119.0",first,logy,ymin,"neutron",beam,leg1,dir,dird,markf);
  myc->cd(2); if (logy != 0) gPad->SetLogy(1); gPad->SetLeftMargin(0.15);
  if (mark == 'y') sprintf(markf, "(b)");
  if (ratio)
    plotKERatio("C",ene,   "119.0",first,logy,ymin,"neutron",beam,leg2,dir,dird,markf);
  else
    plotKE("C",ene,   "119.0",first,logy,ymin,"neutron",beam,leg2,dir,dird,markf);
  myc->cd(3); if (logy != 0) gPad->SetLogy(1); gPad->SetLeftMargin(0.15);
  if (mark == 'y') sprintf(markf, "(c)");
  if (ratio)
    plotKERatio("U","1.4","119.0",first,logy,ymin,"neutron",beam,leg,dir,dird,markf);
  else
    plotKE("U","1.4","119.0",first,logy,ymin,"neutron",beam,leg,dir,dird,markf);
  myc->cd(4); if (logy != 0) gPad->SetLogy(1); gPad->SetLeftMargin(0.15);
  if (mark == 'y') sprintf(markf, "(d)");
  if (ratio)
    plotKERatio("U",ene,   "119.0",first,logy,ymin,"neutron",beam,leg,dir,dird,markf);
  else
    plotKE("U",ene,   "119.0",first,logy,ymin,"neutron",beam,leg,dir,dird,markf);
  
  char fname[60];
  if (save != 0) {
    std::string tag=".gif";
    if (ratio) {
      if (save > 0) tag = "R.eps";
      else          tag = "R.gif";
    } else {
      if (save > 0) tag = ".eps";
    }
    sprintf (fname, "%sCUtoneutron_1%s", beam, tag.c_str());
    myc->SaveAs(fname);
  }

}

void plotKEp(char element[2], char ene[6], int first=0, int logy=0, int save=0,
	     double ymin=-1., char beam[8]="proton", bool ratio=false, 
	     int leg1=1, int leg2=1, char dir[20]=".", char dird[40]=".", 
	     char mark=' ') {

  setStyle();  
  TCanvas *myc = new TCanvas("myc","",800,600); myc->Divide(2,2);

  int leg=leg1; if (leg2 == 0) leg=leg2;
  char markf[4]=" ";
  myc->cd(1); if (logy != 0) gPad->SetLogy(1); gPad->SetLeftMargin(0.15);
  if (mark == 'y') sprintf(markf, "(a)");
  if (ratio)
    plotKERatio(element,"1.4"," 59.1",first,logy,ymin,"proton",beam,leg1,dir,dird,markf);
  else
    plotKE(element,"1.4"," 59.1",first,logy,ymin,"proton",beam,leg1,dir,dird,markf);
  myc->cd(2); if (logy != 0) gPad->SetLogy(1); gPad->SetLeftMargin(0.15);
  if (mark == 'y') sprintf(markf, "(b)");
  if (ratio)
    plotKERatio(element,ene,   " 59.1",first,logy,ymin,"proton",beam,leg2,dir,dird,markf);
  else
    plotKE(element,ene,   " 59.1",first,logy,ymin,"proton",beam,leg2,dir,dird,markf);
  myc->cd(3); if (logy != 0) gPad->SetLogy(1); gPad->SetLeftMargin(0.15);
  if (mark == 'y') sprintf(markf, "(c)");
  if (ratio)
    plotKERatio(element,"1.4","119.0",first,logy,ymin,"proton",beam,leg,dir,dird,markf);
  else
    plotKE(element,"1.4","119.0",first,logy,ymin,"proton",beam,leg,dir,dird,markf);
  myc->cd(4); if (logy != 0) gPad->SetLogy(1); gPad->SetLeftMargin(0.15);
  if (mark == 'y') sprintf(markf, "(d)");
  if (ratio)
    plotKERatio(element,ene,   "119.0",first,logy,ymin,"proton",beam,leg,dir,dird,markf);
  else
    plotKE(element,ene,   "119.0",first,logy,ymin,"proton",beam,leg,dir,dird,markf);

  char fname[60];
  if (save != 0) {
    std::string tag=".gif";
    if (ratio) {
      if (save > 0) tag = "R.eps";
      else          tag = "R.gif";
    } else {
      if (save > 0) tag = ".eps";
    }
    sprintf (fname, "%s%stoproton_1%s", beam, element, tag.c_str());
    myc->SaveAs(fname);
  }

}

void plotKE4(char element[2], char ene[6], int first=0, int logy=0, int save=0,
	     double ymin=-1., char particle[8]="proton", char beam[8]="proton",
	     bool ratio=false, int leg1=1, int leg2=1, char dir[20]=".", 
	     char dird[40]=".", char mark=' ') {

  setStyle();  
  TCanvas *myc = new TCanvas("myc","",800,600); myc->Divide(2,2);

  int leg=leg1; if (leg2 == 0) leg=leg2;
  char markf[4]=" ";
  myc->cd(1); if (logy != 0) gPad->SetLogy(1); gPad->SetLeftMargin(0.15);
  if (mark == 'y') sprintf(markf, "(a)");
  if (ratio)
    plotKERatio(element,ene," 59.1",first,logy,ymin,particle,beam,leg1,dir,dird,markf);
  else
    plotKE(element,ene," 59.1",first,logy,ymin,particle,beam,leg1,dir,dird,markf);
  myc->cd(2); if (logy != 0) gPad->SetLogy(1); gPad->SetLeftMargin(0.15);
  if (mark == 'y') sprintf(markf, "(b)");
  if (ratio)
    plotKERatio(element,ene," 89.0",first,logy,ymin,particle,beam,leg2,dir,dird,markf);
  else
    plotKE(element,ene," 89.0",first,logy,ymin,particle,beam,leg2,dir,dird,markf);
  myc->cd(3); if (logy != 0) gPad->SetLogy(1); gPad->SetLeftMargin(0.15);
  if (mark == 'y') sprintf(markf, "(c)");
  if (ratio)
    plotKERatio(element,ene,"119.0",first,logy,ymin,particle,beam,leg,dir,dird,markf);
  else
    plotKE(element,ene,"119.0",first,logy,ymin,particle,beam,leg,dir,dird,markf);
  myc->cd(4); if (logy != 0) gPad->SetLogy(1); gPad->SetLeftMargin(0.15);
  if (mark == 'y') sprintf(markf, "(d)");
  if (ratio)
    plotKERatio(element,ene,"159.6",first,logy,ymin,particle,beam,leg,dir,dird,markf);
  else
    plotKE(element,ene,"159.6",first,logy,ymin,particle,beam,leg,dir,dird,markf);

  char fname[60];
  if (save != 0) {
    std::string tag=".gif";
    if (ratio) {
      if (save > 0) tag = "R.eps";
      else          tag = "R.gif";
    } else {
      if (save > 0) tag = ".eps";
    }
    sprintf (fname, "%s%sto%sat%sGeV_1%s", beam, element, particle, ene, tag.c_str());
    myc->SaveAs(fname);
  }

}

void plotKE1(char element[2], char ene[6], char angle[6], int first=0, 
	     int logy=0, int save=0, double ymin=-1, char particle[8]="proton",
	     char beam[8]="proton", bool ratio=false, int legend=1, 
	     char dir[20]=".", char dird[40]=".", char markf[4]=" ") {

  setStyle();
  TCanvas *myc = new TCanvas("myc","",800,600); myc->SetLeftMargin(0.15);
  if (logy != 0) gPad->SetLogy(1);
  if (ratio) 
    plotKERatio(element,ene,angle,first,logy,ymin,particle,beam,legend,dir,dird,markf);
  else
    plotKE(element,ene,angle,first,logy,ymin,particle,beam,legend,dir,dird,markf);

  char anglx[6], fname[120];
  int nx = 0;
  for (int i=0; i<6; i++) {
    if (angle[i] != ' ') { anglx[nx] = angle[i]; nx++;}
  }
  if (save != 0) {
    std::string tag=".gif";
    if (ratio) {
      if (save > 0) tag = "R.eps";
      else          tag = "R.gif";
    } else {
      if (save > 0) tag = ".eps";
    }
    sprintf (fname, "%s%sto%sat%sGeV%sdeg%s", beam, element, particle, ene, anglx, tag.c_str());
    myc->SaveAs(fname);
  }
}

void plotKE(char element[2], char ene[6], char angle[6], int first=0, 
	    int logy=0, double ymin=-1, char particle[8]="proton",
            char beam[8]="proton", int legend=1, char dir[20]=".", 
	    char dird[40]=".", char markf[4]=" ") {

  char fname[120], list[40], hname[60], titlx[50];
  TH1F *hi[6];
  int i=0, icol=1;
  sprintf (titlx, "Kinetic Energy of %s (GeV)", particle);
  double  ymx0=1, ymi0=100., xlow=0.06, xhigh=0.26;
  if (particle == "neutron") {xlow= 0.0; xhigh=0.20;}
  for (i=0; i<modelsITEP; i++) {
    sprintf (list, "%s", ModelFilesI[i].c_str()); 
    sprintf (fname, "%s/%s/%s/%s%s%sGeV_1.root", dir, beam, particle, element, list, ene);
    sprintf (list, "%s", ModelsI[i].c_str()); icol = colModel[i];
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
  sprintf (fname, "%s/itep/%s/%s/%s%sGeV%sdeg.dat", dird, beam, particle, element, ene, anglx);
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
  if (ymin > 0) ymi0 = ymin;
  for (i = 0; i<modelsITEP; i++) {
    if (debug) std::cout << "Model " << i << " " << hi[i] << " " << ymi0 << " " << ymx0 << "\n";
    hi[i]->GetYaxis()->SetRangeUser(ymi0,ymx0);
  }

  hi[first]->GetYaxis()->SetTitleOffset(1.6);
  hi[first]->Draw();
  for (i=0; i<modelsITEP; i++) {
    if (i != first) hi[i]->Draw("same");
  }
  gr1->Draw("p");

  TLegend *leg1;
  if (legend < 0) {
    leg1 = new TLegend(0.60,0.55,0.90,0.90);
  } else {
    if (markf == " ") leg1 = new TLegend(0.42,0.55,0.90,0.90);
    else              leg1 = new TLegend(0.38,0.70,0.90,0.90);
  }
  for (i=0; i<modelsITEP; i++) {
    sprintf (list, "%s", ModelNamesI[i].c_str()); 
    leg1->AddEntry(hi[i],list,"F");
  }
  char header[120], beamx[8], partx[2];
  if      (beam == "piplus")  sprintf (beamx, "#pi^{+}");
  else if (beam == "piminus") sprintf (beamx, "#pi^{-}");
  else                        sprintf (beamx, "p");
  if      (particle == "neutron") sprintf (partx, "n");
  else                            sprintf (partx, "p");
  if (legend < 0) {
    sprintf (header,"%s+A #rightarrow %s+X", beamx, partx);
  } else {
    if (markf == " ") 
      sprintf (header,"%s+%s #rightarrow %s+X at %s GeV (#theta = %s^{o})", beamx, element, partx, ene, angle);
    else 
      sprintf (header,"%s %s+%s #rightarrow %s+X at %s GeV (#theta = %s^{o})", markf, beamx, element, partx, ene, angle);
  }
  leg1->SetHeader(header); leg1->SetFillColor(0);
  leg1->SetTextSize(0.04);
  if (legend != 0) leg1->Draw("same");
  if (debug) std::cout << "End\n";
}

void plotKERatio(char element[2], char ene[6], char angle[6], int first=0, 
		 int logy=0, double ymin=-1, char particle[8]="proton",
		 char beam[8]="proton", int legend=1, char dir[20]=".", 
		 char dird[40]=".", char markf[4]=" ") {

  // First open the data file
  char anglx[6], fname[120];
  int nx = 0;
  for (int i=0; i<6; i++) {
    if (angle[i] != ' ') { anglx[nx] = angle[i]; nx++;}
  }
  sprintf (fname, "%s/itep/%s/%s/%s%sGeV%sdeg.dat", dird, beam, particle, element, ene, anglx);
  if (debug) std::cout << "Reads data from file " << fname << "\n";
  ifstream infile;
  infile.open(fname);

  // Read contents of the data file
  int     q1;
  float   m1, r1, x1[30], y1[30], er1[30], staterr, syserr;
  infile >> m1 >> r1 >> q1;
  for (i=0; i<q1; i++) {
    infile >> x1[i] >> y1[i] >> staterr >> syserr;
    syserr *= y1[i];
    er1[i]  = sqrt(syserr*syserr+staterr*staterr);
    if (debug) std::cout << i << " " << x1[i] << " " << y1[i] << " " << er1[i] << "\n";
  }

  char list[40], hname[60], titlx[50];
  TGraphErrors *gr[6];
  int icol=1, ityp=20;
  sprintf (titlx, "Kinetic Energy of %s (GeV)", particle);
  double  ymx0=0.1, ymi0=100., xlow=0.06, xhigh=0.26;
  if (particle == "neutron") {xlow= 0.0; xhigh=0.20;}
  for (int i=0; i<modelsITEP; i++) {
    icol = colModel[i]; ityp = symbModel[i];
    sprintf (list, "%s", ModelFilesI[i].c_str()); 
    sprintf (fname, "%s/%s/%s/%s%s%sGeV_1.root", dir, beam, particle, element, list, ene);
    sprintf (list, "%s", ModelsI[i].c_str()); 
    sprintf (hname, "KE0%s%s%sGeV%s", element, list, ene, angle);

    TFile *file = new TFile(fname);
    TH1F *hi = (TH1F*) file->Get(hname);
    if (debug) std::cout << "Get " << hname << " from " << fname <<" as " << hi <<"\n";
            
    if (hi != 0 && q1 > 0) {
      float xx[30], dx[30], rat[30], drt[30];
      int   nx = hi->GetNbinsX();
      int   np = 0;
      if (debug) std::cout << "Start with " << nx << " bins\n";
      for (int k=1; k <= nx; k++) {
	double xx1 = hi->GetBinLowEdge(k);
	double xx2 = hi->GetBinWidth(k);
	for (int j=0; j<q1; j++) {
	  if (xx1 < x1[j] && xx1+xx2 > x1[j]) {
	    double yy = hi->GetBinContent(k);
	    xx[np]    = x1[j];
	    dx[np]    = 0;
	    rat[np]   = yy/y1[j];
	    drt[np]   = er1[j]*rat[j]/y1[j];
	    if (xx[np] > xlow && xx[np] < xhigh) {
	      if (rat[np]+drt[np] > ymx0) ymx0 = rat[np]+drt[np];
	      if (rat[np]-drt[np] < ymi0) ymi0 = rat[np]-drt[np];
	    }
	    if (debug) std::cout << np << "/" << j << "/" << k << " x " << xx[np] << " (" << xx1 << ":" << xx1+xx2 << ")" << " y " << yy << "/" << y1[j] << " = " << rat[np] << " +- " << drt[np] << "\n";
	    np++;
	    break;
	  }
	}
      }
      gr[i] = new TGraphErrors(np, xx, rat, dx, drt);
      gr[i]->GetXaxis()->SetRangeUser(xlow, xhigh); gr[i]->SetTitle("");
      gr[i]->GetXaxis()->SetTitle(titlx);
      gr[i]->GetYaxis()->SetTitle("MC/Data");
      gr[i]->SetLineStyle(stylModel[i]); gr[i]->SetLineWidth(2); 
      gr[i]->SetLineColor(icol);         gr[i]->SetMarkerColor(icol); 
      gr[i]->SetMarkerStyle(ityp);       gr[i]->SetMarkerSize(1.0); 
    } else {
      gr[i] = 0;
    }
    file->Close();
  }

  if (logy == 0) {ymx0 *= 1.5; ymi0 *= 0.8;}
  else           {ymx0 *=10.0; ymi0 *= 0.2; }
  if (ymin > 0)   ymi0 = ymin;
  for (i = 0; i<modelsITEP; i++) {
    if (debug) std::cout << "Model " << i << " " << gr[i] << " " << ymi0 << " " << ymx0 << "\n";
    if (gr[i] != 0) gr[i]->GetYaxis()->SetRangeUser(ymi0,ymx0);
  }

  gr[first]->GetYaxis()->SetTitleOffset(1.6);
  gr[first]->Draw("APl");
  for (i=0; i<modelsITEP; i++) {
    if (i != first && gr[i] != 0) gr[i]->Draw("Pl");
  }

  TLegend *leg1;
  if (legend < 0) {
    leg1 = new TLegend(0.60,0.55,0.90,0.90);
  } else {
    if (markf == " ") leg1 = new TLegend(0.42,0.55,0.90,0.90);
    else              leg1 = new TLegend(0.38,0.70,0.90,0.90);
  }
  for (i=0; i<modelsITEP; i++) {
    if (gr[i] != 0) {
      sprintf (list, "%s", ModelNamesI[i].c_str()); 
      leg1->AddEntry(gr[i],list,"lP");
    }
  }
  char header[120], beamx[8], partx[2];
  if      (beam == "piplus")  sprintf (beamx, "#pi^{+}");
  else if (beam == "piminus") sprintf (beamx, "#pi^{-}");
  else                        sprintf (beamx, "p");
  if      (particle == "neutron") sprintf (partx, "n");
  else                            sprintf (partx, "p");
  if (legend < 0) {
    sprintf (header,"%s+A #rightarrow %s+X", beamx, partx);
  } else {
    if (markf == " ") 
      sprintf (header,"%s+%s #rightarrow %s+X at %s GeV (#theta = %s^{o})", beamx, element, partx, ene, angle);
    else 
      sprintf (header,"%s %s+%s #rightarrow %s+X at %s GeV (#theta = %s^{o})", markf, beamx, element, partx, ene, angle);
  }
  leg1->SetHeader(header); leg1->SetFillColor(0);
  leg1->SetTextSize(0.04);
  if (legend != 0) leg1->Draw("same");

  xx[0]=xlow; xx[1]=xhigh; rat[0]=rat[1]=1.0;
  TGraph *gr0 = new TGraph(2, xx, rat);
  gr0->GetXaxis()->SetRangeUser(xlow, xhigh); gr0->SetTitle("");
  gr0->SetLineStyle(1);   gr0->SetLineWidth(1.4); 
  gr0->SetLineColor(1);   gr0->SetMarkerColor(1); 
  gr0->SetMarkerStyle(20);gr0->SetMarkerSize(1.6);
  gr0->Draw("l");
}

void plotCT4(char element[2], char ene[6], int first=0, int scan=1, int logy=0,
	     int save=0, char particle[8]="proton", char beam[8]="proton", 
	     int leg1=1, int leg2=1, char dir[20]=".", char dird[40]=".") {

  setStyle();
  TCanvas *myc = new TCanvas("myc","",800,600); myc->Divide(2,2);

  for (int i=0; i<4; i++) {
    myc->cd(i+1); if (logy != 0) gPad->SetLogy(1); gPad->SetLeftMargin(0.15);
    double ke = keproton[i];
    if (particle == "neutron") ke = keneutron[i];
    if (i == 0) 
      plotCT(element, ene,ke, first, scan, logy, particle,beam,leg1,dir,dird); 
    else
      plotCT(element, ene,ke, first, scan, logy, particle,beam,leg2,dir,dird); 
  }

  char fname[60];
  if (save != 0) {
    if (save > 0) sprintf (fname, "%s%sto%sat%sGeV_2.eps", beam, element, particle, ene);
    else          sprintf (fname, "%s%sto%sat%sGeV_2.gif", beam, element, particle, ene);
    myc->SaveAs(fname);
  }
}
 
void plotCT1(char element[2], char ene[6], double ke, int first=0, int scan=1,
	     int logy=0, int save=0, char particle[8]="proton", 
	     char beam[8]="proton", int legend=1, char dir[20]=".",
	     char dird[40]=".") {

  setStyle();
  TCanvas *myc = new TCanvas("myc","",800,600); myc->SetLeftMargin(0.15);
  if (logy != 0) gPad->SetLogy(1);
  plotCT(element, ene,ke, first, scan, logy, particle,beam, legend, dir,dird);

  char fname[60];
  if (save != 0) {
    if (save > 0) sprintf (fname, "%s%sto%sat%sGeV%4.2fGeV.eps", beam, element, particle, ene, ke);
    else          sprintf (fname, "%s%sto%sat%sGeV%4.2fGeV.gif", beam, element, particle, ene, ke);
    myc->SaveAs(fname);
  }
}

void plotCT(char element[2], char ene[6], double ke, int first=0, int scan=1,
	    int logy=0, char particle[8]="proton", char beam[8]="proton", 
	    int legend=1, char dir[20]=".", char dird[40]=".") {

  static double pi  = 3.1415926;
  static double deg = pi/180.; 
  if (debug) std::cout << "Scan " << scan;
  std::vector<double> angles = angleScan(scan);
  int    nn = (int)(angles.size());
  if (debug) std::cout << " gives " << nn << " angles\n";

  char fname[120], list[40], hname[60];
  TH1F *hi[6];
  int i=0, icol=1;
  double  ymx0=1, ymi0=100., xlow=-1.0, xhigh=1.0;
  for (i=0; i<modelsITEP; i++) {
    sprintf (list, "%s", ModelFilesI[i].c_str()); icol = colModel[i];
    sprintf (fname, "%s/%s/%s/%s%s%sGeV_1.root", dir, beam, particle, element, list, ene);
    sprintf (list, "%s", ModelsI[i].c_str()); icol = colModel[i];
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
    sprintf (fname, "%s/itep/%s/%s/%s%sGeV%sdeg.dat", dird, beam, particle, element, ene, anglx);
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
  for (i = 0; i<modelsITEP; i++)
    hi[i]->GetYaxis()->SetRangeUser(ymi0,ymx0);
  
  hi[first]->GetYaxis()->SetTitleOffset(1.6);
  hi[first]->Draw();
  for (i=0; i<modelsITEP; i++) {
    if (i != first)  hi[i]->Draw("same");
  }
  gr1->Draw("p");

  TLegend *leg1;
  if (legend == 1) leg1 = new TLegend(0.15,0.70,0.62,0.90);
  else             leg1 = new TLegend(0.15,0.55,0.62,0.90);
  for (i=0; i<modelsITEP; i++) {
    sprintf (list, "%s", ModelNamesI[i].c_str());
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
  if (legend != 0) leg1->Draw();

}

void plotBE4(char element[2], int logy=0, int scan=1, int save=0, 
	     char particle[8]="proton", char beam[8]="proton", 
	     char dir[20]=".", char dird[40]=".") {

  setStyle();
  TCanvas *myc = new TCanvas("myc","",800,600); myc->Divide(2,2);

  myc->cd(1); if (logy != 0) gPad->SetLogy(1); gPad->SetLeftMargin(0.15);
  plotBE(element, " 59.1", 0.11, logy, scan, particle, beam, dir, dird);
  myc->cd(2); if (logy != 0) gPad->SetLogy(1); gPad->SetLeftMargin(0.15);
  plotBE(element, " 59.1", 0.21, logy, scan, particle, beam, dir, dird);
  myc->cd(3); if (logy != 0) gPad->SetLogy(1); gPad->SetLeftMargin(0.15);
  plotBE(element, "119.0", 0.11, logy, scan, particle, beam, dir, dird);
  myc->cd(4); if (logy != 0) gPad->SetLogy(1); gPad->SetLeftMargin(0.15);
  plotBE(element, "119.0", 0.21, logy, scan, particle, beam, dir, dird);

  char fname[60];
  if (save != 0) {
    if (save > 0) sprintf (fname, "%s%sto%s_1.eps", beam, element, particle);
    else          sprintf (fname, "%s%sto%s_1.gif", beam, element, particle);
    myc->SaveAs(fname);
  }
}

void plotBE1(char element[2], char angle[6], double ke, int logy=0, int scan=1,
	     int save=0, char particle[8]="proton", char beam[8]="proton", 
	     char dir[20]=".", char dird[40]=".") {

  setStyle();
  TCanvas *myc = new TCanvas("myc","",800,600); myc->SetLeftMargin(0.15);
  if (logy != 0) gPad->SetLogy(1);
  plotBE(element, angle, ke, logy, scan, particle, beam, dir, dird);

  char anglx[6], fname[60];
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
	    char particle[8]="proton", char beam[8]="proton", 
	    char dir[20]=".", char dird[40]=".") {

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
  char fname[120], list[40], hname[60];
  int j=0, icol=1, ityp=20;
  double  ymx0=1, ymi0=10000., xmi=5.0, xmx=10.0;
  if (scan > 1) { 
    xmi = 0.5;
    xmx = 9.5;
  }
  for (i=0; i<modelsITEP; i++) {
    icol = colModel[i]; ityp = symbModel[i];
    double yt[15];
    for (j=0; j<nene; j++) {
      sprintf (list, "%s", ModelFilesI[i].c_str()); 
      sprintf (fname, "%s/%s/%s/%s%s%3.1fGeV_1.root", dir, beam, particle, element, list, ene[j]);
      sprintf (list, "%s", ModelsI[i].c_str()); 
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
    sprintf (fname, "%s/itep/%s/%s/%s%3.1fGeV%sdeg.dat", dird, beam, particle, element, ene[j], anglx);
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
  for (i = 0; i<modelsITEP; i++) {
    gr[i]->GetYaxis()->SetRangeUser(ymi0,ymx0);
    gr[i]->GetXaxis()->SetRangeUser(xmi,xmx);
  }
  gr1->GetXaxis()->SetRangeUser(xmi,xmx);
  gr1->GetYaxis()->SetRangeUser(ymi0,ymx0);
  gr1->GetXaxis()->SetTitle("Energy (GeV)"); 
  gr1->GetYaxis()->SetTitle("E#frac{d^{3}#sigma}{dp^{3}} (mb/GeV^{2})"); 

  gr1->GetYaxis()->SetTitleOffset(1.6); gr1->SetTitle("");
  gr1->Draw("ap");
  for (i=0; i<modelsITEP; i++)
    gr[i]->Draw("lp");
  
  TLegend *leg1 = new TLegend(0.35,0.60,0.90,0.90);
  for (i=0; i<modelsITEP; i++) {
    sprintf (list, "%s", ModelNamesI[i].c_str());
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
 
void plotMT4(char ene[6], int first=0, int logy=0, int save=0, double ymin=-1,
	     char particle[8]="piplus", char beam[8]="proton",bool ratio=false,
	     int leg1=1, int leg2=1, char dir[20]=".", char dird[40]=".", 
	     char mark=' ') {

  setStyle();
  TCanvas *myc = new TCanvas("myc","",800,600); myc->Divide(2,2);

  char markf[4]=" ";
  int leg=leg1; if (leg2 == 0) leg=leg2;
  myc->cd(1); if (logy != 0) gPad->SetLogy(1); gPad->SetLeftMargin(0.15);
  if (mark == 'y') sprintf(markf, "(a)");
  if (ratio)
    plotMTRatio("Be",ene,"1.10", first,logy,ymin,particle,beam,leg1,dir,dird,markf);
  else
    plotMT("Be",ene,"1.10", first,logy,ymin,particle,beam,leg1,dir,dird,markf);
  myc->cd(2); if (logy != 0) gPad->SetLogy(1); gPad->SetLeftMargin(0.15);
  if (mark == 'y') sprintf(markf, "(b)");
  if (ratio)
    plotMTRatio("Be",ene,"2.30", first,logy,ymin,particle,beam,leg2,dir,dird,markf);
  else
    plotMT("Be",ene,"2.30", first,logy,ymin,particle,beam,leg2,dir,dird,markf);
  myc->cd(3); if (logy != 0) gPad->SetLogy(1); gPad->SetLeftMargin(0.15);
  if (mark == 'y') sprintf(markf, "(c)");
  if (ratio)
    plotMTRatio("Au",ene,"1.10", first,logy,ymin,particle,beam,leg,dir,dird,markf);
  else
    plotMT("Au",ene,"1.10", first,logy,ymin,particle,beam,leg,dir,dird,markf);
  myc->cd(4); if (logy != 0) gPad->SetLogy(1); gPad->SetLeftMargin(0.15);
  if (mark == 'y') sprintf(markf, "(d)");
  if (ratio)
    plotMTRatio("Au",ene,"2.30", first,logy,ymin,particle,beam,leg,dir,dird,markf);
  else
    plotMT("Au",ene,"2.30", first,logy,ymin,particle,beam,leg,dir,dird,markf);

  char fname[60];
  if (save != 0) {
    std::string tag=".gif";
    if (ratio) {
      if (save > 0) tag = "R.eps";
      else          tag = "R.gif";
    } else {
      if (save > 0) tag = ".eps";
    }
    sprintf (fname, "%sBeAuto%sat%sGeV%s", beam, particle, ene, tag.c_str());
    myc->SaveAs(fname);
  }
}
 
void plotMT4(char element[2], char ene[6], int first=0, int logy=0, int save=0,
	     double ymin=-1, char particle[8]="piplus", char beam[8]="proton", 
	     bool ratio=false, int leg1=1, int leg2=1, char dir[20]=".", 
	     char dird[40]=".", char mark=' ') {

  setStyle();
  TCanvas *myc = new TCanvas("myc","",800,600); myc->Divide(2,2);

  char markf[4]=" ";
  int leg=leg1; if (leg2 == 0) leg=leg2;
  myc->cd(1); if (logy != 0) gPad->SetLogy(1); gPad->SetLeftMargin(0.15);
  if (mark == 'y') sprintf(markf, "(a)");
  if (ratio)
    plotMTRatio(element,ene,"1.10",first,logy,ymin,particle,beam,leg1,dir,dird,markf);
  else
    plotMT(element,ene,"1.10",first,logy,ymin,particle,beam,leg1,dir,dird,markf);
  myc->cd(2); if (logy != 0) gPad->SetLogy(1); gPad->SetLeftMargin(0.15);
  if (mark == 'y') sprintf(markf, "(b)");
  if (ratio)
    plotMTRatio(element,ene,"1.50",first,logy,ymin,particle,beam,leg2,dir,dird,markf);
  else
    plotMT(element,ene,"1.50",first,logy,ymin,particle,beam,leg2,dir,dird,markf);
  myc->cd(3); if (logy != 0) gPad->SetLogy(1); gPad->SetLeftMargin(0.15);
  if (mark == 'y') sprintf(markf, "(c)");
  if (ratio)
    plotMTRatio(element,ene,"1.90",first,logy,ymin,particle,beam,leg2,dir,dird,markf);
  else
    plotMT(element,ene,"1.90",first,logy,ymin,particle,beam,leg2,dir,dird,markf);
  myc->cd(4); if (logy != 0) gPad->SetLogy(1); gPad->SetLeftMargin(0.15);
  if (mark == 'y') sprintf(markf, "(d)");
  if (ratio)
    plotMTRatio(element,ene,"2.30",first,logy,ymin,particle,beam,leg2,dir,dird,markf);
  else
    plotMT(element,ene,"2.30",first,logy,ymin,particle,beam,leg2,dir,dird,markf);

  char fname[60];
  if (save != 0) {
    std::string tag=".gif";
    if (ratio) {
      if (save > 0) tag = "R.eps";
      else          tag = "R.gif";
    } else {
      if (save > 0) tag = ".eps";
    }
    sprintf (fname, "%s%sto%sat%sGeV%s", beam, element, particle, ene, tag.c_str());
    myc->SaveAs(fname);
  }
}
 
void plotMT1(char element[2], char ene[6], char rapid[6], int first=0, 
	     int logy=0, int save=0, double ymin=-1, char particle[8]="piplus",
	     char beam[8]="proton", bool ratio=false, int legend=1, 
	     char dir[20]=".", char dird[40]=".", char markf[4]=" ") {

  setStyle();
  TCanvas *myc = new TCanvas("myc","",800,600); myc->SetLeftMargin(0.15);
  if (logy != 0) gPad->SetLogy(1);
  if (ratio)
    plotMTRatio(element,ene,rapid,first,logy,ymin,particle,beam,legend,dir,dird,markf);
  else
    plotMT(element,ene,rapid,first,logy,ymin,particle,beam,legend,dir,dird,markf);

  char fname[60];
  if (save != 0) {
    std::string tag=".gif";
    if (ratio) {
      if (save > 0) tag = "R.eps";
      else          tag = "R.gif";
    } else {
      if (save > 0) tag = ".eps";
    }
    sprintf (fname, "%s%sto%sat%sGeVY%s%s", beam, element, particle, ene, rapid, tag.c_str());
    myc->SaveAs(fname);
  }
}

void plotMT(char element[2], char ene[6], char rapid[6], int first=0, 
	    int logy=0, double ymin=-1, char particle[8]="piplus", 
	    char beam[8]="proton", int legend=0, char dir[20]=".",
	    char dird[40]=".", char markf[4]=" ") {

  char fname[120], list[40], hname[60], titlx[50], sym[6];
  TH1F *hi[6];
  int i=0, icol=1;
  if      (particle=="piminus") sprintf(sym, "#pi^{-}");
  else if (particle=="piplus")  sprintf(sym, "#pi^{+}");
  else if (particle=="kminus")  sprintf(sym, "K^{-}");
  else if (particle=="kplus")   sprintf(sym, "K^{+}");
  else                          sprintf(sym, "p");
  sprintf (titlx, "Reduced m_{T} (GeV)");
  double  ymx0=1, ymi0=100., xlow=0.1, xhigh=1.6;
  for (i=0; i<modelsBNL; i++) {
    sprintf (list, "%s", ModelFilesB[i].c_str()); 
    sprintf (fname, "%s/%s/%s/%s%s%sGeV_1.root", dir, beam, particle, element, list, ene);
    sprintf (list, "%s", ModelsB[i].c_str()); icol = colModel[i];
    sprintf (hname, "KE0%s%s%sGeVy%s", element, list, ene, rapid);
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
    if (debug) std::cout << "ylimit " << ymi0 << ":" << ymx0 << "\n";
    hi[i]->GetXaxis()->SetRangeUser(xlow, xhigh); hi[i]->SetTitle("");
    hi[i]->GetXaxis()->SetTitle(titlx);
    hi[i]->SetLineStyle(1);  hi[i]->SetLineWidth(2); hi[i]->SetLineColor(icol);
    //    file->Close();
  }

  sprintf (fname, "%s/bnl802/%s/%s/%s%sGeVRap%s.dat", dird, beam, particle, element, ene, rapid);
  std::cout << "Reads data from file " << fname << "\n";
  ifstream infile;
  infile.open(fname);
  int     q1;
  float   ym1, ym2, sys, x1[50], y1[50], stater1[50], syser1[50];
  infile >> q1 >> ym1 >> ym2 >> sys;
  for (i=0; i<q1; i++) {
    infile >> x1[i] >> y1[i] >> stater1[i];
    syser1[i] = sys*y1[i];
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
  if (ymin > 0) ymi0 = ymin;
  for (i = 0; i<modelsBNL; i++) {
    if (debug) std::cout << "Model " << i << " " << hi[i] << " " << ymi0 << " " << ymx0 << "\n";
    hi[i]->GetYaxis()->SetRangeUser(ymi0,ymx0);
  }

  hi[first]->GetYaxis()->SetTitleOffset(1.1);
  hi[first]->Draw();
  for (i=0; i<modelsBNL; i++) {
    if (i != first) hi[i]->Draw("same");
  }
  gr1->Draw("p");

  TLegend *leg1;
  if (legend < 0) {
    leg1 = new TLegend(0.50,0.55,0.90,0.90);
  } else {
    if (markf == " " ) leg1 = new TLegend(0.42,0.70,0.90,0.90);
    else               leg1 = new TLegend(0.38,0.70,0.90,0.90);
  }
  for (i=0; i<modelsBNL; i++) {
    sprintf (list, "%s", ModelNamesB[i].c_str()); 
    leg1->AddEntry(hi[i],list,"F");
  }
  char header[120], beamx[8], partx[2];
  if      (beam == "piplus")  sprintf (beamx, "#pi^{+}");
  else if (beam == "piminus") sprintf (beamx, "#pi^{-}");
  else                        sprintf (beamx, "p");
  if (legend < 0) {
    sprintf (header,"%s+%s #rightarrow %s+X at %s GeV", beamx, element, sym, ene);
  } else {
    if (markf == " ")
      sprintf (header,"%s+%s #rightarrow %s+X at %s GeV (y = %s)", beamx, element, sym, ene, rapid);
    else
      sprintf (header,"%s %s+%s #rightarrow %s+X at %s GeV (y = %s)", markf, beamx, element, sym, ene, rapid);
  }
  leg1->SetHeader(header); leg1->SetFillColor(0);
  leg1->SetTextSize(0.04);
  if (legend != 0) leg1->Draw("same");

  if (debug) std::cout << "End\n";
}

void plotMTRatio(char element[2], char ene[6], char rapid[6], int first=0, 
		 int logy=0, double ymin=-1, char particle[8]="piplus", 
		 char beam[8]="proton", int legend=0, char dir[20]=".",
		 char dird[40]=".", char markf[4]=" ") {

  char titlx[50], sym[6];
  int  i=0, icol=1, ityp=20;
  if      (particle=="piminus") sprintf(sym, "#pi^{-}");
  else if (particle=="piplus")  sprintf(sym, "#pi^{+}");
  else if (particle=="kminus")  sprintf(sym, "K^{-}");
  else if (particle=="kplus")   sprintf(sym, "K^{+}");
  else                          sprintf(sym, "p");
  sprintf (titlx, "Reduced m_{T} (GeV)");

  //Read in the data files
  char fname[120];
  sprintf (fname, "%s/bnl802/%s/%s/%s%sGeVRap%s.dat", dird, beam, particle, element, ene, rapid);
  if (debug) std::cout << "Reads data from file " << fname << "\n";
  ifstream infile;
  infile.open(fname);
  int     q1;
  float   ym1, ym2, sys, x1[50], y1[50], er1[50], staterr, syserr;
  infile >> q1 >> ym1 >> ym2 >> sys;
  for (i=0; i<q1; i++) {
    infile >> x1[i] >> y1[i] >> staterr;
    syserr = sys*y1[i];
    er1[i] = sqrt(syserr*syserr+staterr*staterr);
    if (debug) std::cout << i << " " << x1[i] << " " << y1[i] << " " << er1[i] << "\n";
  }

  char          list[40], hname[60];
  TGraphErrors *gr[6];
  double        ymx0=1, ymi0=100., xlow=0.1, xhigh=1.6;
  for (i=0; i<modelsBNL; i++) {
    icol = colModel[i]; ityp = symbModel[i];
    sprintf (list, "%s", ModelFilesB[i].c_str()); 
    sprintf (fname, "%s/%s/%s/%s%s%sGeV_1.root", dir, beam, particle, element, list, ene);
    sprintf (list, "%s", ModelsB[i].c_str());
    sprintf (hname, "KE0%s%s%sGeVy%s", element, list, ene, rapid);

    TFile *file = new TFile(fname);
    TH1F  *hi   = (TH1F*) file->Get(hname);
    if (debug) std::cout << "Get " << hname << " from " << fname <<" as " << hi <<"\n";

    if (hi != 0 && q1 > 0) {
      float xx[50], dx[50], rat[50], drt[50];
      int   nx = hi->GetNbinsX();
      int   np = 0;
      if (debug) std::cout << "Start with " << nx << " bins\n";
      for (int k=1; k <= nx; k++) {
	double xx1 = hi->GetBinLowEdge(k);
	double xx2 = hi->GetBinWidth(k);
	for (int j=0; j<q1; j++) {
	  if (xx1 < x1[j] && xx1+xx2 > x1[j]) {
	    double yy = hi->GetBinContent(k);
	    xx[np]    = x1[j];
	    dx[np]    = 0;
	    rat[np]   = yy/y1[j];
	    drt[np]   = er1[j]*rat[j]/y1[j];
	    if (xx[np] > xlow && xx[np] < xhigh) {
	      if (rat[np]+drt[np] > ymx0) ymx0 = rat[np]+drt[np];
	      if (rat[np]-drt[np] < ymi0) ymi0 = rat[np]-drt[np];
	    }
	    if (debug) std::cout << np << "/" << j << "/" << k << " x " << xx[np] << " (" << xx1 << ":" << xx1+xx2 << ")" << " y " << yy << "/" << y1[j] << " = " << rat[np] << " +- " << drt[np] << "\n";
	    np++;
	    break;
	  }
	}
      }
      gr[i] = new TGraphErrors(np, xx, rat, dx, drt);
      gr[i]->GetXaxis()->SetRangeUser(xlow, xhigh); gr[i]->SetTitle("");
      gr[i]->GetXaxis()->SetTitle(titlx);
      gr[i]->GetYaxis()->SetTitle("MC/Data");
      gr[i]->SetLineStyle(stylModel[i]); gr[i]->SetLineWidth(2); 
      gr[i]->SetLineColor(icol);         gr[i]->SetMarkerColor(icol); 
      gr[i]->SetMarkerStyle(ityp);       gr[i]->SetMarkerSize(1.0); 
    } else {
      gr[i] = 0;
    }
    file->Close();
  }

  if (logy == 0) {ymx0 *= 1.5; ymi0 *= 0.8;}
  else           {ymx0 *=10.0; ymi0 *= 0.2; }
  if (ymin > 0) ymi0 = ymin;
  for (i = 0; i<modelsBNL; i++) {
    if (gr[i] != 0) {
      if (debug) std::cout << "Model " << i << " " << gr[i] << " " << ymi0 << " " << ymx0 << "\n";
      gr[i]->GetYaxis()->SetRangeUser(ymi0,ymx0);
    }
  }

  if (gr[first] > 0) {
    gr[first]->GetYaxis()->SetTitleOffset(1.1);
    gr[first]->Draw("APl");
    for (i=0; i<modelsBNL; i++) {
      if (i != first && gr[i] != 0) gr[i]->Draw("Pl");
    }

    TLegend *leg1;
    if (legend < 0) {
      leg1 = new TLegend(0.50,0.55,0.90,0.90);
    } else {
      if (markf == " " ) leg1 = new TLegend(0.42,0.70,0.90,0.90);
      else               leg1 = new TLegend(0.38,0.70,0.90,0.90);
    }
    for (i=0; i<modelsBNL; i++) {
      if (gr[i] != 0) {
	sprintf (list, "%s", ModelNamesB[i].c_str()); 
	leg1->AddEntry(gr[i],list,"lP");
      }
    }
    char header[120], beamx[8], partx[2];
    if      (beam == "piplus")  sprintf (beamx, "#pi^{+}");
    else if (beam == "piminus") sprintf (beamx, "#pi^{-}");
    else                        sprintf (beamx, "p");
    if (legend < 0) {
      sprintf (header,"%s+%s #rightarrow %s+X at %s GeV", beamx, element, sym, ene);
    } else {
      if (markf == " ")
	sprintf (header,"%s+%s #rightarrow %s+X at %s GeV (y = %s)", beamx, element, sym, ene, rapid);
      else
	sprintf (header,"%s %s+%s #rightarrow %s+X at %s GeV (y = %s)", markf, beamx, element, sym, ene, rapid);
    }
    leg1->SetHeader(header); leg1->SetFillColor(0);
    leg1->SetTextSize(0.04);
    if (legend != 0) leg1->Draw("same");

    xx[0]=xlow; xx[1]=xhigh; rat[0]=rat[1]=1.0;
    TGraph *gr0 = new TGraph(2, xx, rat);
    gr0->GetXaxis()->SetRangeUser(xlow, xhigh); gr0->SetTitle("");
    gr0->SetLineStyle(1);   gr0->SetLineWidth(1.4); 
    gr0->SetLineColor(1);   gr0->SetMarkerColor(1); 
    gr0->SetMarkerStyle(20);gr0->SetMarkerSize(1.6);
    gr0->Draw("l");
  }
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
