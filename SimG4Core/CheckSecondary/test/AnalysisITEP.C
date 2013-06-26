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

void AnalyseITEP(char element[2], char list[20], char ene[6], char part[2]="p", int scan=1, char plot='Y') {

  static double massP = 938.272;
  static double massN = 939.565;
  static double pi  = 3.1415926;
  static double deg = pi/180.;
  static double dcth= 0.05, de=0.02;
  double massp = massP;
  std::string fpart("Proton");
  if (part == "n" || part == "N") { 
    massp = massN;
    fpart = "Neutron";
  }
  char fname[40];
  sprintf (fname, "%s%s%sGeV.root", element, list, ene);

  double rhol = rhoL(element);
  double atwt = atomicWt(element);
  cout << fname << " rhoL " << rhol << " atomic weight " << atwt << "\n";
  std::vector<double> angles = angleScan(scan);
  std::vector<double> energies = energyScan(part);

  char name[60], title[160];
  sprintf (title, "All %s", fpart.c_str());
  TH1F *hiK0 = new TH1F ("hiK0", title,  800, 0.,8.);
  TH1F *hiC0 = new TH1F ("hiC0", title,  100,-1.,1.);
  sprintf (title, "Elastc Scattered %s", fpart.c_str());
  TH1F *hiK1 = new TH1F ("hiK1", title,  800, 0.,8.);
  TH1F *hiC1 = new TH1F ("hiC1", title,  100,-1.,1.);
  sprintf (title, "Inelastc Scattered %s", fpart.c_str());
  TH1F *hiK2 = new TH1F ("hiK2", title,  800, 0.,8.);
  TH1F *hiC2 = new TH1F ("hiC2", title,  100,-1.,1.);
  std::vector<double> cthmin, cthmax;
  TH1F *hiKE1[30], *hiKE2[30];
  unsigned int ii=0;
  for (ii=0; ii<angles.size(); ii++) {
    double cth = cos(angles[ii]);
    cthmin.push_back(cth-0.5*dcth);
    cthmax.push_back(cth+0.5*dcth);
    sprintf (name, "KE1%s%s%sGeV%5.1f", element, list, ene, angles[ii]/deg);
    sprintf (title, "p+%s at %s GeV (%s) (#theta = %8.2f)", element, ene, list, angles[ii]/deg);
    hiKE1[ii] = new TH1F (name, title, 800, 0., 8.);
    //std::cout << "hiKE1[" << ii << "] = " << hiKE1[ii] << " " <<  name << "   " << title << "\n";
    sprintf (name, "KE2%s%s%sGeV%5.1f", element, list, ene, angles[ii]/deg);
    sprintf (title, "p+%s at %s GeV (%s) (#theta = %8.2f)", element, ene, list, angles[ii]/deg);
    hiKE2[ii] = new TH1F (name, title, 800, 0., 8.);
    //std::cout << "hiKE2[" << ii << "] = " << hiKE2[ii] << " " <<  name << "   " << title << "\n";
  }

  std::vector<double> emin, emax;
  TH1F *hiCT1[30], *hiCT2[30];
  for (ii=0; ii<energies.size(); ii++) {
    double en = energies[ii];
    emin.push_back(en-0.5*de);
    emax.push_back(en+0.5*de);
    sprintf (name, "CT1%s%s%sGeV%4.2f", element, list, ene, energies[ii]);
    sprintf (title, "p+%s at %s GeV (%s) (KE = %6.2f GeV)", element, ene, list, energies[ii]);
    hiCT1[ii] = new TH1F (name, title, 80, -1., 1.);
    //std::cout << "hiCT1[" << ii << "] = " << hiCT1[ii] << " " <<  name << "   " << title << "\n";
    sprintf (name, "CT2%s%s%sGeV%4.2f", element, list, ene, energies[ii]);
    sprintf (title, "p+%s at %s GeV (%s) (KE = %6.2f GeV)", element, ene, list, energies[ii]);
    hiCT2[ii] = new TH1F (name, title, 80, -1., 1.);
    //std::cout << "hiCT2[" << ii << "] = " << hiCT2[ii] << " " <<  name << "   " << title << "\n";
  }

  TFile *file = new TFile(fname);
  TTree *tree = (TTree *) file->Get("T1");
  int interval = 100000;
  if (plot == 'N' || plot == 'n') interval = 100000;

  if (!tree) {
    std::cout << "Cannot find Tree T1 in file " << fname << "\n";
  } else {
    std::cout << "Tree T1 found with " << tree->GetEntries() << " entries\n";
    int nentry = tree->GetEntries();
    int ninter=0, elastic=0, inelastic=0;
    for (int i=0; i<nentry; i++) {
      if (i%interval == 0) std:cout << "Started with event # " << i << "\n";
      std::vector<int>                     *nsec, *procids;
      std::vector<double>                  *px, *py, *pz, *mass;
      std::vector<std::string>             *procs;
      tree->SetBranchAddress("NumberSecondaries", &nsec);
      tree->SetBranchAddress("ProcessID",         &procids);
      //      tree->SetBranchAddress("ProcessNames",      &procs);
      tree->SetBranchAddress("SecondaryPx",       &px);
      tree->SetBranchAddress("SecondaryPy",       &py);
      tree->SetBranchAddress("SecondaryPz",       &pz);
      tree->SetBranchAddress("SecondaryMass",     &mass);
      tree->GetEntry(i);
      if ((*nsec).size() > 0) {
	ninter++;
	bool isItElastic = false;
	if ((*procids)[0] == 17) {elastic++; isItElastic = true;}
	else                     inelastic++;

	if (ninter <3) {
	  std::cout << "Interaction " << ninter << "/" << i+1 << " Type "
		    << (*procids)[0]  << " with " << (*nsec)[0] << " secondaries\n";
	  for (int k=0; k<(*nsec)[0]; k++)
	    std::cout << " Secondary " << k << " Px " << (*px)[k] << " Py " << (*py)[k] << " Pz " << (*pz)[k] << " Mass " << (*mass)[k] << "\n";
	}

	for (int k=0; k<(*nsec)[0]; k++) {
	  if (abs((*mass)[k]-massp) < 0.01) { // This is the required particle
	    double pl = (*py)[k];
	    double pt = ((*px)[k])*((*px)[k])+((*pz)[k])*((*pz)[k]);
	    double pp = (pt+pl*pl);
	    double ke = (sqrt (pp + massp*massp) - massp)/1000.;
	    pp        = sqrt (pp);
	    double cth= (pp == 0. ? -2. : (pl/pp));
	    double wt = (pp == 0. ?  0. : (1000./pp));
	    // std::cout << "Entry " << i << " Secondary " << k << " Cth " << cth << " KE " << ke << " WT " << wt << "\n";
	    hiK0->Fill(ke);
	    hiC0->Fill(cth);
	    if (isItElastic) {
	      hiK1->Fill(ke);
	      hiC1->Fill(cth);
	    } else {
	      hiK2->Fill(ke);
	      hiC2->Fill(cth);
	      for (ii=0; ii<angles.size(); ii++) {
		if (cth > cthmin[ii] && cth <= cthmax[ii]) {
		  // std::cout << " Loop " << ii << " Limit " << cthmin[ii] << " " << cthmax[ii] << " " << hiKE1[ii] << " " << hiKE2[ii] << "\n";
		  hiKE1[ii]->Fill(ke);
		  hiKE2[ii]->Fill(ke,wt);
		}
	      }
	      for (ii=0; ii<energies.size(); ii++) {
		if (ke > emin[ii] && ke <= emax[ii]) {
		  // std::cout << " Loop " << ii << " Limit " << emin[ii] << " " << emax[ii] << " " << hiCT1[ii] << " " << hiCT2[ii] << "\n";
		  hiCT1[ii]->Fill(cth);
		  hiCT2[ii]->Fill(cth,wt);
		}
	      }
	    }
	  }
	}
      }
    }

    std::cout << ninter << " interactions seen in " << nentry << " trials\n";
    double sigma = atwt*10000.*log((double)(nentry)/(double)(nentry-ninter))/(rhol*6.023);
    double dsigma    = sigma/sqrt(double(max(ninter,1)));
    double sigmaEl   = sigma*((double)(elastic))/((double)(max(ninter,1)));
    double dsigmaEl  = sigmaEl/sqrt(double(max(elastic,1)));
    double sigmaInel = sigma*((double)(inelastic))/((double)(max(ninter,1)));
    double dsigmaInel= sigmaInel/sqrt(double(max(inelastic,1)));
    std::cout << "Total     " << sigma << " +- " << dsigma 
	      << " mb (" << ninter << " events)\n"
	      << "Elastic   " << sigmaEl<< " +- " << dsigmaEl
	      << " mb (" << ninter << " events)\n"
	      << "Inelastic " << sigmaInel << " +- " << dsigmaInel
	      << " mb (" << ninter << " events)\n";
  }

  gStyle->SetCanvasBorderMode(0); gStyle->SetCanvasColor(kWhite);
  gStyle->SetPadColor(kWhite);    gStyle->SetFrameBorderMode(0);
  gStyle->SetFrameBorderSize(1);  gStyle->SetFrameFillColor(0);
  gStyle->SetFrameFillStyle(0);   gStyle->SetFrameLineColor(1);
  gStyle->SetFrameLineStyle(1);   gStyle->SetFrameLineWidth(1);
  gStyle->SetOptLogy(1);          gStyle->SetTitleOffset(1.2,"Y");
    
  sprintf (title, "Kinetic Energy of %s (GeV)", fpart.c_str());
  hiK0->GetXaxis()->SetTitle(title);
  hiK1->GetXaxis()->SetTitle(title);
  hiK2->GetXaxis()->SetTitle(title);
  if (plot != 'N' && plot != 'n') {
    TCanvas *c1 = new TCanvas("c1","K.E.",800,600); c1->Divide(2,2);
    c1->cd(1); hiK1->Draw(); c1->cd(2); hiK2->Draw(); c1->cd(3); hiK0->Draw();
  }
  sprintf (title, "cos (#theta) of scattered %s", fpart.c_str());
  hiC0->GetXaxis()->SetTitle(title);
  hiC1->GetXaxis()->SetTitle(title);
  hiC2->GetXaxis()->SetTitle(title);
  if (plot != 'N' && plot != 'n') {
    TCanvas *c2 = new TCanvas("c2","cos#theta",800,600); c2->Divide(2,2);
    c2->cd(1); hiC1->Draw(); c2->cd(2); hiC2->Draw(); c2->cd(3); hiC0->Draw();
  }
  TCanvas *cc[30];
  TH1F    *hiKE0[30];
  for (ii=0; ii<angles.size(); ii++) {
    double xbin = hiKE1[ii]->GetBinWidth(1);
    sprintf (title, "Kinetic Energy of %s (GeV)", fpart.c_str());
    hiKE1[ii]->GetXaxis()->SetTitle(title);
    sprintf (title, "Events/%6.3f GeV", xbin);
    hiKE1[ii]->GetYaxis()->SetTitle(title);
    double xbin  = hiKE2[ii]->GetBinWidth(1);
    double scale = sigmaInel/(((double)(max(inelastic,1)))*xbin*2.*pi*dcth);
    std::cout << "Bin " << ii << " Angle " << angles[ii]/deg << " Bin " << xbin << " Scale " << scale << " " << title << "\n";
    sprintf (title, "Kinetic Energy of %s (GeV)", fpart.c_str());
    hiKE2[ii]->GetXaxis()->SetTitle(title);
    sprintf (title, "Events (scaled by #frac{1}{p})/%6.3f GeV", xbin);
    hiKE2[ii]->GetYaxis()->SetTitle(title);
    sprintf (name, "KE0%s%s%sGeV%5.1f", element, list, ene, angles[ii]/deg);
    hiKE0[ii] = (TH1F*)hiKE2[ii]->Clone();
    hiKE0[ii]->SetName(name); hiKE0[ii]->Scale(scale);
    hiKE0[ii]->GetYaxis()->SetTitle("E#frac{d^{3}#sigma}{dp^{3}} (mb/GeV^{2})");
      
    if (plot != 'N' && plot != 'n') {
      sprintf(name, "Canvas%i", ii);
      sprintf (title, "p+%s at %s GeV (%s) (#theta = %8.2f)", element, ene, list, angles[ii]/deg);
      cc[ii] = new TCanvas(name,title,800,600); cc[ii]->Divide(2,2);
    
      std::cout << "hiKE1: " << hiKE1[ii]->GetName() << " " << hiKE1[ii]->GetEntries() << " " << hiKE1[ii] << "\n";
      cc[ii]->cd(1); hiKE1[ii]->Draw();
      std::cout << "hiKE0: " << hiKE0[ii]->GetName() << " " << hiKE0[ii]->GetEntries() << " " << hiKE0[ii] << "\n";
      cc[ii]->cd(2); hiKE0[ii]->Draw();
      std::cout << "hiKE2: " << hiKE2[ii]->GetName() << " " << hiKE2[ii]->GetEntries() << " " << hiKE2[ii] << "\n";
      cc[ii]->cd(3); hiKE2[ii]->Draw(); 
    }
  }
    
  TCanvas *ct[30];
  TH1F    *hiCT0[30];
  for (ii=0; ii<energies.size(); ii++) {
    double xbin = hiCT1[ii]->GetBinWidth(1);
    sprintf (title, "Events/%6.3f", xbin);
    hiCT1[ii]->GetXaxis()->SetTitle("cos (#theta)");
    hiCT1[ii]->GetYaxis()->SetTitle(title);
    double xbin  = hiCT2[ii]->GetBinWidth(1);
    double scale = sigmaInel/(((double)(max(inelastic,1)))*xbin*2.*pi*de);
    std::cout << "Bin " << ii << " KE " << energies[ii] << " GeV Bin " << xbin << " Scale " << scale << " " << title << "\n";
    sprintf (title, "Events (scaled by #frac{1}{p})/%6.3f", xbin);
    hiCT2[ii]->GetXaxis()->SetTitle("cos (#theta)");
    hiCT2[ii]->GetYaxis()->SetTitle(title);
    sprintf (name, "CT0%s%s%sGeV%4.2f", element, list, ene, energies[ii]);
    hiCT0[ii] = (TH1F*)hiCT2[ii]->Clone();
    hiCT0[ii]->SetName(name); hiCT0[ii]->Scale(scale);
    hiCT0[ii]->GetYaxis()->SetTitle("E#frac{d^{3}#sigma}{dp^{3}} (mb/GeV^{2})");
      
    if (plot != 'N' && plot != 'n') {
      sprintf(name, "Canvas0%i", ii);
      sprintf (title, "p+%s at %s GeV (%s) (KE = %6.2f GeV)", element, ene, list, energies[ii]);
      ct[ii] = new TCanvas(name,title,800,600); ct[ii]->Divide(2,2);
      std::cout << "hiCT1: " << hiCT1[ii]->GetName() << " " << hiCT1[ii]->GetEntries() << " " << hiCT1[ii] << "\n";
      ct[ii]->cd(1); hiCT1[ii]->Draw();
      std::cout << "hiCT0: " << hiCT0[ii]->GetName() << " " << hiCT0[ii]->GetEntries() << " " << hiCT0[ii] << "\n";
      ct[ii]->cd(2); hiCT0[ii]->Draw();
      std::cout << "hiCT2: " << hiCT2[ii]->GetName() << " " << hiCT2[ii]->GetEntries() << " " << hiCT2[ii] << "\n";
      ct[ii]->cd(3); hiCT2[ii]->Draw(); 
    }
  }

  char ofile[40];
  sprintf (ofile, "%s%s%sGeV_%i.root", element, list, ene, scan);
  TFile f(ofile, "recreate");
  hiK0->Write(); hiK1->Write(); hiK2->Write();
  hiC0->Write(); hiC1->Write(); hiC2->Write();
  for (ii=0; ii<angles.size(); ii++) {
    hiKE1[ii]->Write(); hiKE0[ii]->Write(); hiKE2[ii]->Write();
  }
  for (ii=0; ii<energies.size(); ii++) {
    hiCT1[ii]->Write(); hiCT0[ii]->Write(); hiCT2[ii]->Write();
  }
  f.Close();
  std::cout << "o/p saved in file " << ofile << "\n";

  file->Close();
}

void AnalyseBNL(char element[2], char list[10], char ene[6], char part[4]="p", char plot='Y') {
  
  static double pi   = 3.1415926;
  static double deg  = pi/180.;
  int           indx = 13;
  std::string fpart("Proton");
  if      (part == "n"    || part == "N")    {fpart = "Neutron";    indx = 14;}
  else if (part == "pbar" || part == "PBAR") {fpart = "\bar{p}";    indx = 12;}
  else if (part == "kz"   || part == "KZ")   {fpart = "K^{0}";      indx = 11;}
  else if (part == "kp"   || part == "KP")   {fpart = "K^{+}";      indx = 10;}
  else if (part == "km"   || part == "KM")   {fpart = "K^{-}";      indx = 9;}
  else if (part == "pip"  || part == "PIP")  {fpart = "#pi^{+}";    indx = 8;}
  else if (part == "pim"  || part == "PIM")  {fpart = "#pi^{-}";    indx = 7;}
  char fname[40];
  sprintf (fname, "%s%s%sGeV.root", element, list, ene);
  
  double rhol = rhoL(element);
  double atwt = atomicWt(element);
  cout << fname << " rhoL " << rhol << " atomic weight " << atwt << "\n";
  std::vector<double> rapidities = rapidityScan();
  
  char name[60], title[160];
  sprintf (title, "All %s", fpart.c_str());
  TH1F *hiK0 = new TH1F ("hiK0", title,  500, 0.,5.);
  TH1F *hiC0 = new TH1F ("hiC0", title,  100,-1.,1.);
  sprintf (title, "Elastc Scattered %s", fpart.c_str());
  TH1F *hiK1 = new TH1F ("hiK1", title,  500, 0.,5.);
  TH1F *hiC1 = new TH1F ("hiC1", title,  100,-1.,1.);
  sprintf (title, "Inelastc Scattered %s", fpart.c_str());
  TH1F *hiK2 = new TH1F ("hiK2", title,  500, 0.,5.);
  TH1F *hiC2 = new TH1F ("hiC2", title,  100,-1.,1.);
  std::vector<double> ymin, ymax;
  TH1F *hiKE1[30], *hiKE2[30];
  unsigned int ii=0;
  for (ii=0; ii<rapidities.size()-1; ii++) {
    ymin.push_back(rapidities[ii]);
    ymax.push_back(rapidities[ii+1]);
    double yv = 0.5*(rapidities[ii]+rapidities[ii+1]);
    sprintf (name, "KE1%s%s%sGeVy%4.2f", element, list, ene, yv);
    sprintf (title, "p+%s at %s GeV (%s) (y = %8.2f)", element, ene, list, yv);
    hiKE1[ii] = new TH1F (name, title, 800, 0., 8.);
    std::cout << "hiKE1[" << ii << "] = " << hiKE1[ii] << " " <<  name << "   " << title << "\n";
    sprintf (name, "KE2%s%s%sGeVy%4.2f", element, list, ene, yv);
    sprintf (title, "p+%s at %s GeV (%s) (y = %8.2f)", element, ene, list, yv);
    hiKE2[ii] = new TH1F (name, title, 800, 0., 8.);
    std::cout << "hiKE2[" << ii << "] = " << hiKE2[ii] << " " <<  name << "   " << title << "\n";
  }
  
  TFile *file = new TFile(fname);
  TTree *tree = (TTree *) file->Get("T1");
  int interval = 100000;
  if (plot == 'N' || plot == 'n') interval = 100000;
  
  if (!tree) {
    std::cout << "Cannot find Tree T1 in file " << fname << "\n";
  } else {
    std::cout << "Tree T1 found with " << tree->GetEntries() << " entries\n";
    int nentry = tree->GetEntries();
    int ninter=0, elastic=0, inelastic=0;
    for (int i=0; i<nentry; i++) {
      if (i%interval == 0) std:cout << "Started with event # " << i << "\n";
      std::vector<int>                     *nsec, *procids;
      std::vector<double>                  *px, *py, *pz, *mass;
      std::vector<std::string>             *procs;
      tree->SetBranchAddress("NumberSecondaries", &nsec);
      tree->SetBranchAddress("ProcessID",         &procids);
      //      tree->SetBranchAddress("ProcessNames",      &procs);
      tree->SetBranchAddress("SecondaryPx",       &px);
      tree->SetBranchAddress("SecondaryPy",       &py);
      tree->SetBranchAddress("SecondaryPz",       &pz);
      tree->SetBranchAddress("SecondaryMass",     &mass);
      tree->GetEntry(i);
      if ((*nsec).size() > 0) {
	ninter++;
	bool isItElastic = false;
	if ((*procids)[0] == 17) {elastic++; isItElastic = true;}
	else                     inelastic++;
	
	if (ninter <10) {
	  std::cout << "Interaction " << ninter << "/" << i+1 << " Type "
		    << (*procids)[0]  << " with " << (*nsec)[0] << " secondaries\n";
	  for (int k=0; k<(*nsec)[0]; k++)
	    std::cout << " Secondary " << k << " Px " << (*px)[k] << " Py " << (*py)[k] << " Pz " << (*pz)[k] << " Mass " << (*mass)[k] << "\n";
	}
	
	for (int k=0; k<(*nsec)[0]; k++) {
	  double massp = (*mass)[k];
	  int    type  = type(massp);
	  if (type == indx) { // This is the required particle
	    double pl = (*py)[k];
	    double pt = ((*px)[k])*((*px)[k])+((*pz)[k])*((*pz)[k]);
	    double pp = (pt+pl*pl);
	    double mt = sqrt (pt + massp*massp);
	    double mtp= (mt - std::abs(massp))/1000.;
	    double ee = sqrt(pp+massp*massp);
	    double yv = 0.5*log((ee+pl)/(ee-pl));
	    pp        = sqrt (pp);
	    double cth= (pp == 0. ? -2. : (pl/pp));
	    double wt = (mt == 0. ?  0. : (1000./mt));
	    if (ninter <10) std::cout << "Entry " << i << " Secondary " << k << " yv " << yv << " mtp " << mtp << " WT " << wt << "\n";
	    hiK0->Fill(mtp);
	    hiC0->Fill(cth);
	    if (isItElastic) {
	      hiK1->Fill(mtp);
	      hiC1->Fill(cth);
	    } else {
	      hiK2->Fill(mtp);
	      hiC2->Fill(cth);
	      for (ii=0; ii<ymin.size(); ii++) {
		if (yv > ymin[ii] && yv <= ymax[ii]) {
		  if (ninter <10) std::cout << " Loop " << ii << " Limit " << ymin[ii] << " " << ymax[ii] << " " << hiKE1[ii] << " " << hiKE2[ii] << "\n";
		  hiKE1[ii]->Fill(mtp);
		  hiKE2[ii]->Fill(mtp,wt);
		}
	      }
	    }
	  }
	}
      }
    }
    
    std::cout << ninter << " interactions seen in " << nentry << " trials\n";
    double sigma = atwt*10000.*log((double)(nentry)/(double)(nentry-ninter))/(rhol*6.023);
    double dsigma    = sigma/sqrt(double(max(ninter,1)));
    double sigmaEl   = sigma*((double)(elastic))/((double)(max(ninter,1)));
    double dsigmaEl  = sigmaEl/sqrt(double(max(elastic,1)));
    double sigmaInel = sigma*((double)(inelastic))/((double)(max(ninter,1)));
    double dsigmaInel= sigmaInel/sqrt(double(max(inelastic,1)));
    std::cout << "Total     " << sigma << " +- " << dsigma 
	      << " mb (" << ninter << " events)\n"
	      << "Elastic   " << sigmaEl<< " +- " << dsigmaEl
	      << " mb (" << ninter << " events)\n"
	      << "Inelastic " << sigmaInel << " +- " << dsigmaInel
	      << " mb (" << ninter << " events)\n";
  }

  gStyle->SetCanvasBorderMode(0); gStyle->SetCanvasColor(kWhite);
  gStyle->SetPadColor(kWhite);    gStyle->SetFrameBorderMode(0);
  gStyle->SetFrameBorderSize(1);  gStyle->SetFrameFillColor(0);
  gStyle->SetFrameFillStyle(0);   gStyle->SetFrameLineColor(1);
  gStyle->SetFrameLineStyle(1);   gStyle->SetFrameLineWidth(1);
  gStyle->SetOptLogy(1);          gStyle->SetTitleOffset(1.2,"Y");
    
  sprintf (title, "Reduced Transverse mass of %s (GeV)", fpart.c_str());
  hiK0->GetXaxis()->SetTitle(title);
  hiK1->GetXaxis()->SetTitle(title);
  hiK2->GetXaxis()->SetTitle(title);
  if (plot != 'N' && plot != 'n') {
    TCanvas *c1 = new TCanvas("c1","Reduced Transverse Mass",800,600); 
    c1->Divide(2,2);
    c1->cd(1); hiK1->Draw(); c1->cd(2); hiK2->Draw(); c1->cd(3); hiK0->Draw();
  }
  sprintf (title, "cos (#theta) of scattered %s", fpart.c_str());
  hiC0->GetXaxis()->SetTitle(title);
  hiC1->GetXaxis()->SetTitle(title);
  hiC2->GetXaxis()->SetTitle(title);
  if (plot != 'N' && plot != 'n') {
    TCanvas *c2 = new TCanvas("c2","cos#theta",800,600); c2->Divide(2,2);
    c2->cd(1); hiC1->Draw(); c2->cd(2); hiC2->Draw(); c2->cd(3); hiC0->Draw();
  }
  TCanvas *cc[30];
  TH1F    *hiKE0[30];
  for (ii=0; ii<ymin.size(); ii++) {
    double xbin = hiKE1[ii]->GetBinWidth(1);
    sprintf (title, "Reduced Transverse Mass of %s (GeV)", fpart.c_str());
    hiKE1[ii]->GetXaxis()->SetTitle(title);
    sprintf (title, "Events/%6.3f GeV", xbin);
    hiKE1[ii]->GetYaxis()->SetTitle(title);
    double xbin  = hiKE2[ii]->GetBinWidth(1);
    double dy    = (ymax[ii]-ymin[ii]);
    double yv    = 0.5*(ymin[ii]+ymax[ii]);
    double scale = sigmaInel/(((double)(max(inelastic,1)))*xbin*2.*pi*dy);
    std::cout << "Bin " << ii << " yv " << yv << " Bin " << xbin << " Scale " << scale << " " << title << "\n";
    sprintf (title, "Reduced Transverse mass of %s (GeV)", fpart.c_str());
    hiKE2[ii]->GetXaxis()->SetTitle(title);
    sprintf (title, "Events (scaled by #frac{1}{mT})/%6.3f GeV",xbin);
    hiKE2[ii]->GetYaxis()->SetTitle(title);
    sprintf (name, "KE0%s%s%sGeVy%4.2f", element, list, ene, yv);
    hiKE0[ii] = (TH1F*)hiKE2[ii]->Clone();
    hiKE0[ii]->SetName(name); hiKE0[ii]->Scale(scale);
    hiKE0[ii]->GetYaxis()->SetTitle("E#frac{d^{3}#sigma}{dp^{3}} (mb/GeV^{2})");
      
    if (plot != 'N' && plot != 'n') {
      sprintf(name, "Canvas%i", ii);
      sprintf (title, "p+%s at %s GeV (%s) (y = %5.2f)", element,ene,list,yv);
      cc[ii] = new TCanvas(name,title,800,600); cc[ii]->Divide(2,2);
      std::cout << "hiKE1: " << hiKE1[ii]->GetName() << " " << hiKE1[ii]->GetEntries() << " " << hiKE1[ii] << "\n";
      cc[ii]->cd(1); hiKE1[ii]->Draw();
      std::cout << "hiKE0: " << hiKE0[ii]->GetName() << " " << hiKE0[ii]->GetEntries() << " " << hiKE0[ii] << "\n";
      cc[ii]->cd(2); hiKE0[ii]->Draw();
      std::cout << "hiKE2: " << hiKE2[ii]->GetName() << " " << hiKE2[ii]->GetEntries() << " " << hiKE2[ii] << "\n";
      cc[ii]->cd(3); hiKE2[ii]->Draw(); 
    }
  }
    
  char ofile[40];
  sprintf (ofile, "%s%s%sGeV_1.root", element, list, ene);
  TFile f(ofile, "recreate");
  hiK0->Write(); hiK1->Write(); hiK2->Write();
  hiC0->Write(); hiC1->Write(); hiC2->Write();
  for (ii=0; ii<ymin.size(); ii++) {
    hiKE1[ii]->Write(); hiKE0[ii]->Write(); hiKE2[ii]->Write();
  }
  f.Close();
  std::cout << "o/p saved in file " << ofile << "\n";

  file->Close();
}

double rhoL(char element[2]) {

  double tmp=0;
  if      (element == "H")   tmp = 0.0708 * 800.;
  else if (element == "Be")  tmp = 1.848 * 80.;
  else if (element == "C")   tmp = 2.265 * 80.;
  else if (element == "Al")  tmp = 2.700 * 80.;
  else if (element == "Ti")  tmp = 4.530 * 40.;
  else if (element == "Fe")  tmp = 7.870 * 30.;
  else if (element == "Cu")  tmp = 8.960 * 30.;
  else if (element == "Nb")  tmp = 8.550 * 30.;
  else if (element == "Cd")  tmp = 8.630 * 30.;
  else if (element == "Sn")  tmp = 7.310 * 35.;
  else if (element == "Ta")  tmp = 16.65 * 20.;
  else if (element == "Au")  tmp = 18.85 * 20.;
  else if (element == "Pb")  tmp = 11.35 * 30.;
  else if (element == "U")   tmp = 18.95 * 20.;
  return tmp;
}

double atomicWt(char element[2]) {

  double tmp=0;
  if      (element == "H")   tmp = 1.00794;
  else if (element == "Be")  tmp = 9.0122;
  else if (element == "C")   tmp = 12.011;
  else if (element == "Al")  tmp = 26.98;
  else if (element == "Ti")  tmp = 47.88;
  else if (element == "Fe")  tmp = 55.85;
  else if (element == "Cu")  tmp = 63.546;
  else if (element == "Nb")  tmp = 92.906;
  else if (element == "Cd")  tmp = 112.41;
  else if (element == "Sn")  tmp = 118.69;
  else if (element == "Ta")  tmp = 180.9479;
  else if (element == "Au")  tmp = 196.97;
  else if (element == "Pb")  tmp = 207.19;
  else if (element == "U")   tmp = 238.03;
  return tmp;
}

std::vector<double> angleScan(int scan) {

  static double deg = 3.1415926/180.;
  std::vector<double> tmp;
  if (scan <= 1) {
    tmp.push_back(59.1*deg);
    tmp.push_back(89.0*deg);
    tmp.push_back(119.0*deg);
    tmp.push_back(159.6*deg);
  } else {
    tmp.push_back(10.1*deg);
    tmp.push_back(15.0*deg);
    tmp.push_back(19.8*deg);
    tmp.push_back(24.8*deg);
    tmp.push_back(29.5*deg);
    tmp.push_back(34.6*deg);
    tmp.push_back(39.6*deg);
    tmp.push_back(44.3*deg);
    tmp.push_back(49.3*deg);
    tmp.push_back(54.2*deg);
    tmp.push_back(59.1*deg);
    tmp.push_back(64.1*deg);
    tmp.push_back(69.1*deg);
    tmp.push_back(74.1*deg);
    tmp.push_back(79.1*deg);
    tmp.push_back(84.1*deg);
    tmp.push_back(89.0*deg);
    tmp.push_back(98.9*deg);
    tmp.push_back(108.9*deg);
    tmp.push_back(119.0*deg);
    tmp.push_back(129.1*deg);
    tmp.push_back(139.1*deg);
    tmp.push_back(149.3*deg);
    tmp.push_back(159.6*deg);
    tmp.push_back(161.4*deg);
    tmp.push_back(165.5*deg);
    tmp.push_back(169.5*deg);
    tmp.push_back(173.5*deg);
    tmp.push_back(177.0*deg);
  }
  std::cout << "Scan " << tmp.size() << " angular regions:\n";
  for (unsigned int i=0; i<tmp.size(); i++) {
    std::cout << tmp[i]/deg;
    if (i == tmp.size()-1) std::cout << " degrees\n";
    else                   std::cout << ", ";
  }
  return tmp;
}

std::vector<double> energyScan(char part[2]) {

  std::vector<double> tmp;
  if (part == "n" || part == "N") {
    tmp.push_back(0.01);
    tmp.push_back(0.03);
    tmp.push_back(0.05);
  }
  tmp.push_back(0.07);
  tmp.push_back(0.09);
  tmp.push_back(0.11);
  tmp.push_back(0.13);
  tmp.push_back(0.15);
  tmp.push_back(0.17);
  tmp.push_back(0.19);
  tmp.push_back(0.21);
  tmp.push_back(0.23);
  tmp.push_back(0.25);

  std::cout << "Scan " << tmp.size() << " Energy regions:\n";
  for (unsigned int i=0; i<tmp.size(); i++) {
    std::cout << tmp[i];
    if (i == tmp.size()-1) std::cout << " GeV\n";
    else                   std::cout << ", ";
  }
  return tmp;
}

std::vector<double> rapidityScan() {

  std::vector<double> tmp;
  tmp.push_back(0.60);
  tmp.push_back(0.80);
  tmp.push_back(1.00);
  tmp.push_back(1.20);
  tmp.push_back(1.40);
  tmp.push_back(1.60);
  tmp.push_back(1.80);
  tmp.push_back(2.00);
  tmp.push_back(2.20);
  tmp.push_back(2.40);
  tmp.push_back(2.60);
  tmp.push_back(2.80);
  tmp.push_back(3.00);

  std::cout << "Scan " << tmp.size() << " rapidity regions:\n";
  for (unsigned int i=0; i<tmp.size(); i++) {
    std::cout << tmp[i];
    if (i == tmp.size()-1) std::cout << ";\n";
    else                   std::cout << ", ";
  }
  return tmp;
}

int type(double mass) {

  double m  = (mass >=0 ? mass: -mass);
  int    tmp=0;
  if      (m < 0.01)   {tmp = 1;}
  else if (m < 1.00)   {if (mass < 0) tmp = 2; else tmp = 3;}
  else if (m < 115.00) {if (mass < 0) tmp = 4; else tmp = 5;}
  else if (m < 135.00) tmp = 6;
  else if (m < 140.00) {if (mass < 0) tmp = 7; else tmp = 8;}
  else if (m < 495.00) {if (mass < 0) tmp = 9; else tmp = 10;}
  else if (m < 500.00) tmp = 11;
  else if (m < 938.50) {if (mass < 0) tmp = 12; else tmp = 13;}
  else if (m < 940.00) tmp = 14;
  else if (m < 1850.0) {tmp = 15;}
  else                 {tmp = 16;}
  //  std::cout << "Mass " << mass << " type " << tmp << "\n";
  return tmp;
}
