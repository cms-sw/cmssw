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
#include "TH1I.h"
#include "TCanvas.h"
#include "TApplication.h"
#include "TRefArray.h"
#include "TStyle.h"
#include "TGraph.h"

static unsigned int antiproton=12, proton=13, neutron=14, heavy=15, ions=16;

void AnalyseH2TB(char element[6], char list[20], char ene[6], char part[4], int sav=0, int nMax=-1, bool debug=false) {

  char *g4ver = "9.2.ref01P";
  bool detail = true;

  int  energy = atoi(ene);
  char fname[120];
  sprintf (fname, "%s%s_%s_%sGeV.root", element, list, part, ene);
  char ofile[130];
  sprintf (ofile, "histo/histo_%s", fname);

  double rhol = rhoL(element);
  double atwt = atomicWt(element);
  std::vector<std::string> types = types();
  std::cout << fname << " rhoL " << rhol << " atomic weight " << atwt << "\n";
  
  TFile *fout = new TFile(ofile, "recreate");
  TH1F *hiKE0[20], *hiKE1[20], *hiKE2[20], *hiCT0[20], *hiCT1[20], *hiCT2[20];
  TH1I *hiMulti[20];
  TH1F *hiParticle[5][20], *hiTotalKE[20], *hiMomInclusive[20];
  TH1F *hiSumP[20], *hiPT2[20], *hiEP2[4], *baryon1, *baryon2;
  TH1F *hProton[2], *hNeutron[2], *hHeavy[2], *hIon[2], *hBaryon[2];;
  char name[80], title[180], ctype[20], ytitle[20];
  double xbin;

  for (unsigned int ii=0; ii<=(types.size()); ii++) {
    if      (ii == 0) sprintf (ctype, "All Particles");
    else              sprintf (ctype, "%s", types[ii-1].c_str());
    
    sprintf (title, "%s GeV %s on %s: Inclusive Mom Dist of %s (%s %s)", ene, part, element, ctype, g4ver, list);
    sprintf (name, "hiMomInclusive_%s%s%sGeV(%s)", element, list, ene, ctype);
    hiMomInclusive[ii] = new TH1F (name, title, 6000, 0., 300.);

    for (unsigned int jj=0; jj<5; jj++) {
      sprintf (title, "Particle %i : %s in %s at %s GeV (%s)",jj, ctype, element, ene, list);
      sprintf (name, "Particle%i_KE%s%s%sGeV(%s)",jj,element, list, ene, ctype);
      if (ii==ions) hiParticle[jj][ii] = new TH1F (name, title, 50000, 0., 10.);
      else          hiParticle[jj][ii] = new TH1F (name, title, 15500, 0., 310.);
      if (debug) std::cout << "hiParticle[" << jj << "][" << ii << "] = " << hiParticle[jj][ii] << " " <<  name << " Particle KE Energy  " << title << "\n"; 
    }
    if (debug) std::cout << "Ctype " << ctype << "\n";

    sprintf (title, "%s in %s at %s GeV (%s)", ctype, element, ene, list);
    sprintf (name, "KE0%s%s%sGeV(%s)", element, list, ene, ctype);
    hiKE0[ii] = new TH1F (name, title, 15000, 0., 300.);
    hiKE0[ii]->GetXaxis()->SetTitle("Kinetic Energy (GeV)");
    xbin = hiKE0[ii]->GetBinWidth(1);
    sprintf (ytitle, "Events/%6.3f GeV", xbin);
    hiKE0[ii]->GetYaxis()->SetTitle(ytitle);
    if (debug) std::cout << "hiKE0[" << ii << "] = " << hiKE0[ii] << " " <<  name << " KE Energy  " << title << "\n";

    sprintf (name, "CT0%s%s%sGeV(%s)", element, list, ene, ctype);
    hiCT0[ii] = new TH1F (name, title, 100, -1.0, 1.0.);
    hiCT0[ii]->GetXaxis()->SetTitle("cos (#theta)");
    xbin = hiCT0[ii]->GetBinWidth(1);
    sprintf (ytitle, "Events/%6.3f", xbin);
    hiCT0[ii]->GetYaxis()->SetTitle(ytitle);
    if (debug) std::cout << "hiCT0[" << ii << "] = " << hiCT0[ii] << " " <<  name << " cos(T#eta) " << title << "\n";

    sprintf (title, "%s (Elastic) in %s at %s GeV (%s)", ctype, element, ene, list);
    sprintf (name, "KE1%s%s%sGeV(%s)", element, list, ene, ctype);
    hiKE1[ii] = new TH1F (name, title, 15000, 0., 300.);
    hiKE1[ii]->GetXaxis()->SetTitle("Kinetic Energy (GeV)");
    xbin = hiKE1[ii]->GetBinWidth(1);
    sprintf (ytitle, "Events/%6.3f GeV", xbin);
    hiKE1[ii]->GetYaxis()->SetTitle(ytitle);
    if (debug) std::cout << "hiKE1[" << ii << "] = " << hiKE1[ii] << " " <<  name << " KE Energy  " << title << "\n";

    sprintf (name, "CT1%s%s%sGeV(%s)", element, list, ene, ctype);
    hiCT1[ii] = new TH1F (name, title, 100, -1.0, 1.0.);
    hiCT1[ii]->GetXaxis()->SetTitle("cos (#theta)");
    xbin = hiCT1[ii]->GetBinWidth(1);
    sprintf (ytitle, "Events/%6.3f", xbin);
    hiCT1[ii]->GetYaxis()->SetTitle(ytitle);
    if (debug) std::cout << "hiCT1[" << ii << "] = " << hiCT1[ii] << " " <<  name << " cos(T#eta) " << title << "\n";

    sprintf (title, "%s (InElastic) in %s at %s GeV (%s)", ctype, element, ene, list);
    sprintf (name, "KE2%s%s%sGeV(%s)", element, list, ene, ctype);
    hiKE2[ii] = new TH1F (name, title, 15000, 0., 300.);
    hiKE2[ii]->GetXaxis()->SetTitle("Kinetic Energy (GeV)");
    xbin = hiKE2[ii]->GetBinWidth(1);
    sprintf (ytitle, "Events/%6.3f GeV", xbin);
    hiKE2[ii]->GetYaxis()->SetTitle(ytitle);
    if (debug) std::cout << "hiKE2[" << ii << "] = " << hiKE2[ii] << " " <<  name << " KE Energy  " << title << "\n";

    sprintf (name, "CT2%s%s%sGeV(%s)", element, list, ene, ctype);
    hiCT2[ii] = new TH1F (name, title, 100, -1.0, 1.0.);
    hiCT2[ii]->GetXaxis()->SetTitle("cos (#theta)");
    xbin = hiCT2[ii]->GetBinWidth(1);
    sprintf (ytitle, "Events/%6.3f", xbin);
    hiCT2[ii]->GetYaxis()->SetTitle(ytitle);
    if (debug) std::cout << "hiCT2[" << ii << "] = " << hiCT2[ii] << " " <<  name << " cos(T#eta) " << title << "\n";

    sprintf (name, "PT2%s%s%sGeV(%s)", element, list, ene, ctype);
    hiPT2[ii] = new TH1F (name, title, 15000, 0.0, 3000.0.);
    hiPT2[ii]->GetXaxis()->SetTitle("p_T (GeV)");
    xbin = hiCT2[ii]->GetBinWidth(1);
    sprintf (ytitle, "Events/(%6.3f GeV)", xbin);
    hiPT2[ii]->GetYaxis()->SetTitle(ytitle);
    if (debug) std::cout << "hiPT2[" << ii << "] = " << hiPT2[ii] << " " <<  name << " pT " << title << "\n";

    sprintf (name, "Multi%s%s%sGeV(%s)", element, list, ene, ctype);
    sprintf (title,"%s multiplicity in %s at %s GeV (%s)", ctype, element, ene, list);
    hiMulti[ii] = new TH1I (name, title, 101, -1, 100);
    hiMulti[ii]->GetXaxis()->SetTitle("Multiplicity");
    if (debug) std::cout << "hiMulti[" << ii << "] = " << hiMulti[ii] << " " <<  name << " Multiplicity\n";

    sprintf (name, "TotalKE%s%s%sGeV(%s)", element, list, ene, ctype);
    sprintf (title,"%s (inelastic) in %s at %s GeV (%s)", ctype, element, ene, list);
    hiTotalKE[ii] = new TH1F (name, title, 15500, 0, 310);
    sprintf (title, "Total KE carried by %s", ctype);
    hiTotalKE[ii]->GetXaxis()->SetTitle(title);
    if (debug) std::cout << "hiTotalKE[" << ii << "] = " << hiTotalKE[ii] << " " <<  name << " " << title << "\n";

    sprintf(name, "hiSumMomentum%s%s%sGeV(%s)", element, list, ene, ctype);
    sprintf (title, "%s GeV %s on %s: Total Mom carried by %s (%s %s)", ene, part, element, ctype, g4ver, list);
    hiSumP[ii] = new TH1F (name, title, 6000, 0., 310.);
    if (debug) std::cout << "hiSumP[" << ii << "] = " << hiSumP[ii] << " " <<  name << " " << title << "\n";
  }
  hiEP2[0] = new TH1F ("sumPX", "Sum px", 2000., -100., 100.);
  hiEP2[0]->GetXaxis()->SetTitle("Momentum balance (x)");
  hiEP2[0]->GetYaxis()->SetTitle("Events");
  hiEP2[1] = new TH1F ("sumPY", "Sum py", 2000., -100., 100.);
  hiEP2[1]->GetXaxis()->SetTitle("Momentum balance (y)");
  hiEP2[1]->GetYaxis()->SetTitle("Events");
  hiEP2[2] = new TH1F ("sumPZ", "Sum pz", 2000., -100., 100.);
  hiEP2[2]->GetXaxis()->SetTitle("Momentum balance (z)");
  hiEP2[2]->GetYaxis()->SetTitle("Events");
  hiEP2[3] = new TH1F ("sumE",  "Sum E",  2000., -100., 100.);
  hiEP2[3]->GetXaxis()->SetTitle("Energy balance");
  hiEP2[3]->GetYaxis()->SetTitle("Events");

  if (detail) {
    for (int i=0; i<2; i++) {
      sprintf(title, "proton%i_%s%s%sGeV(%s)", i, element, list, ene, ctype);
      hProton[i] = new TH1F(title, title, 15500, 0., 310.);
      hProton[i]->GetXaxis()->SetTitle("Kinetic Energy (GeV)");

      sprintf(title, "neutron%i_%s%s%sGeV(%s)", i, element, list, ene, ctype);
      hNeutron[i] = new TH1F(title, title, 15500, 0., 310.);
      hNeutron[i]->GetXaxis()->SetTitle("Kinetic Energy (GeV)");

      sprintf(title, "heavy%i_%s%s%sGeV(%s)", i, element, list, ene, ctype);
      hHeavy[i] = new TH1F(title, title, 15500, 0., 310.);
      hHeavy[i]->GetXaxis()->SetTitle("Kinetic Energy (GeV)");

      sprintf(title, "ion%i_%s%s%sGeV(%s)", i, element, list, ene, ctype);
      hIon[i] = new TH1F(title, title, 50000, 0., 10.);
      hIon[i]->GetXaxis()->SetTitle("Kinetic Energy (GeV)");

      sprintf(title, "baryon%i_%s%s%sGeV(%s)", i, element, list, ene, ctype);
      hBaryon[i] = new TH1F(title, title, 15500, 0., 310.);
      hBaryon[i]->GetXaxis()->SetTitle("Kinetic Energy (GeV)");
    }

    sprintf(title, "baryonX_%s%s%sGeV", element, list, ene);
    baryon1 = new TH1F("baryon1", title, 15500, 0., 310.);
    sprintf(title, "baryonY_%s%s%sGeV", element, list, ene);
    baryon2 = new TH1F("baryon2", title, 15500, 0., 310.);
  }

  //  sprintf(fname, "out/%s", fname);
  std::cout << "Reading from " << fname << std::endl;

  TFile *file = new TFile(fname);
  TTree *tree = (TTree *) file->Get("T1");
  
  if (!tree) {
    std::cout << "Cannot find Tree T1 in file " << fname << "\n";
  } else {
    std::cout << "Tree T1 found with " << tree->GetEntries() << " entries\n";
    int nentry = tree->GetEntries();
    int ninter=0, elastic=0, inelastic=0;
    if (nMax > 0 && nMax < nentry) nentry = nMax;

    for (int i=0; i<nentry; i++) {
      if (i%1000 == 0) std::cout << "Start processing event " << i << "\n";
      std::vector<int>                     *nsec, *procids;
      std::vector<double>                  *px, *py, *pz, *mass;
      std::vector<std::string>             *procs;
      tree->SetBranchAddress("NumberSecondaries", &nsec);
      tree->SetBranchAddress("ProcessID",         &procids);
      tree->SetBranchAddress("SecondaryPx",       &px);
      tree->SetBranchAddress("SecondaryPy",       &py);
      tree->SetBranchAddress("SecondaryPz",       &pz);
      tree->SetBranchAddress("SecondaryMass",     &mass);
      tree->GetEntry(i);

      
      if ((*nsec).size() > 0) {

	if (debug) std::cout << "Entry " << i << " no. of secondaries " << (*nsec)[0] << "\n";
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
	
	std::list<double> pProton, pNeutron, pIon, pHeavy;
	int counter[20];
	double sumKE[20], sumPx[20], sumPy[20], sumPz[20];
	for(unsigned int nct=0; nct<=types.size(); nct++) {
	  counter[nct] = 0;   sumKE[nct] = 0;
	  sumPx[nct]   = 0.0; sumPy[nct] = 0.0; sumPz[nct] = 0.0;
	}

	double sumpx=0, sumpy=-energy, sumpz=0, sume=-energy;
	int num = (*nsec)[0];
	if (debug) std::cout << "Secondaries: " << num << "\n";
	std::vector<double> partEne(num,0.0); 
	std::vector<int>    partType(num,999);
	double              kBaryon=0;

	for (int k=0; k<(*nsec)[0]; k++) {
	  int type = type((*mass)[k]);
	  double m  = (((*mass)[k]) >=0 ? ((*mass)[k]): -((*mass)[k]));
	  m        /= 1000.;
	  double pl = ((*py)[k])/1000.;
	  double p1 = ((*px)[k])/1000.;
	  double p2 = ((*pz)[k])/1000.;
	  double pt = (p1*p1 + p2*p2);
	  double pp = (pt+pl*pl);
	  double ke = (sqrt (pp + m*m) - m);
	  pp        = sqrt (pp);
	  double cth= (pp == 0. ? -2. : (pl/pp));
	  pt        = sqrt(pt);

	  if      (type == proton)  pProton.push_back(ke);
	  else if (type == neutron) pNeutron.push_back(ke);
	  else if (type == heavy)   pHeavy.push_back(ke);
	  else if (type == ions)    pIon.push_back(ke);
	  if (type == proton || type == neutron || type == heavy) kBaryon += ke;

	  sumpx += p1;
	  sumpy += pl;
	  sumpz += p2;
	  sume  += (ke+m);
	  partEne[k] = ke; partType[k] = type;
	  hiMomInclusive[0]->Fill(pp);
	  hiMomInclusive[type]->Fill(pp);
	  if (debug) std::cout << "Entry " << i << " Secondary " << k << " Mass " << (*mass)[k] << " Type " << type << " Cth " << cth << " KE " << ke << "\n";
	  hiKE0[0]->Fill(ke);
	  hiCT0[0]->Fill(cth);
	  hiKE0[type]->Fill(ke);
	  hiCT0[type]->Fill(cth);
	  if (isItElastic) {
	    hiKE1[0]->Fill(ke);
	    hiCT1[0]->Fill(cth);
	    hiKE1[type]->Fill(ke);
	    hiCT1[type]->Fill(cth);
	  } else {
	    hiKE2[0]->Fill(ke);
	    hiCT2[0]->Fill(cth);
	    hiPT2[0]->Fill(pt);
	    hiKE2[type]->Fill(ke);
	    hiCT2[type]->Fill(cth);
	    hiPT2[type]->Fill(pt);
	    counter[0]    += 1;
	    counter[type] += 1;
	    sumKE[0]      += ke;
	    sumKE[type]   += ke;
	    sumPx[0]      += (*px)[k];
	    sumPy[0]      += (*py)[k];
	    sumPz[0]      += (*pz)[k];
	    sumPx[type]   += (*px)[k];
	    sumPy[type]   += (*py)[k];
	    sumPz[type]   += (*pz)[k];
	  }
	} // loop over particles
	
	if (debug) std::cout << "Entry " << i << "  :: Elastic (?) " << isItElastic << "\n";

	if( !isItElastic ) {
	  for (unsigned int nct=0; nct<=(types.size()); nct++) {
	    double sumP = std::sqrt(sumPx[nct]*sumPx[nct] + sumPy[nct]*sumPy[nct] + sumPz[nct]*sumPz[nct]);
	    if(debug) {
	      if(nct<1) sdt::cout <<  sumP << "   "; 
	      else      sdt::cout << types[nct-1] << " " <<  sumP << "   "; 
	    }
	    hiSumP[nct]->Fill( sumP );

	    hiMulti[nct]->Fill(counter[nct]);
	    hiTotalKE[nct]->Fill(sumKE[nct]);
	    if (debug) {
	      for (unsigned int nct=0; nct<=types.size(); nct++) 
		std::cout << "  [" << nct <<"]:" << counter[nct] << " KE " << sumKE[nct] << "\n";
	    }
	    hiEP2[0]->Fill(sumpx);
	    hiEP2[1]->Fill(sumpy);
	    hiEP2[2]->Fill(sumpz);
	    hiEP2[3]->Fill(sume);
	  }

	  if (detail) {
	    list<double>::iterator iter;
	    double kMaxB=0;
	    if (pProton.size() > 0) {
	      pProton.sort();
	      iter = pProton.end(); iter--;
	      double pMax = *iter;
	      kMaxB = pMax;
	      hProton[0]->Fill(pMax);
	      double pNext= 0;
	      if (pProton.size() > 1) {
		iter--; pNext = *iter;
		hProton[1]->Fill(pNext);
	      }
	      if (debug) std::cout << "Proton " << pProton.size() << " " << pMax << " " << pNext << "\n";
	    }

	    if (pNeutron.size() > 0) {
	      pNeutron.sort();
	      iter = pNeutron.end(); iter--;
	      double pMax = *iter;
	      if (pMax > kMaxB) kMaxB = pMax;
	      hNeutron[0]->Fill(pMax);
	      double pNext= 0;
	      if (pNeutron.size() > 1) {
		iter--; pNext = *iter;
		hNeutron[1]->Fill(pNext);
	      }
	      if (debug) std::cout << "Neutron " << pNeutron.size() << " " << pMax << " " << pNext << "\n";
	    }

	    if (pHeavy.size() > 0) {
	      pHeavy.sort();
	      iter = pHeavy.end(); iter--;
	      double pMax = *iter;
	      if (pMax > kMaxB) kMaxB = pMax;
	      hHeavy[0]->Fill(pMax);
	      double pNext= 0;
	      if (pHeavy.size() > 1) {
		iter--; pNext = *iter;
		hHeavy[1]->Fill(pNext);
	      }
	    }

	    if (kMaxB > 0) {
	      hBaryon[0]->Fill(kMaxB);
	      hBaryon[1]->Fill(kBaryon-kMaxB);
	      if (debug) std::cout << "Baryon " << kMaxB << " " << kBaryon-kMaxB << "\n";
	    }

	    if (pIon.size() > 0) {
	      pIon.sort();
	      iter = pIon.end(); iter--;
	      double pMax = *iter;
	      hIon[0]->Fill(pMax);
	      double pNext= 0;
	      if (pIon.size() > 1) {
		iter--; pNext = *iter;
		hIon[1]->Fill(pNext);
	      }
	    }
	  
	    for(int ipart=0; ipart<partEne.size(); ipart++){
	      for(int jpart=ipart+1; jpart<partEne.size(); jpart++){
		if(partEne[ipart] < partEne[jpart]){
		  double tempE = partEne[ipart];    int tempI       = partType[ipart];
		  partEne[ipart] = partEne[jpart];  partType[ipart] = partType[jpart];
		  partEne[jpart] = tempE;           partType[jpart] = tempI;
		}
	      }
	    }

	    int nPart[20]; for(unsigned int ii=0; ii<20; ii++) nPart[ii]=0;
	    int nbaryon=0; 
	    bool firstBaryon  = false;
	    bool secondBaryon = false;
	    bool first = true;
	    for(int unsigned ii=0; ii<partEne.size(); ii++){
	      nPart[partType[ii]] += 1;
	      if(partType[ii]==proton || partType[ii]==neutron){
		if (first)     baryon1->Fill(partEne[ii]);
		else           baryon2->Fill(partEne[ii]);
		first = false;
	      }
	      
	      if(partType[ii]==antiproton || partType[ii]==proton || partType[ii]==neutron || partType[ii]==heavy || partType[ii]==ions) {
		nbaryon++;
		if(nbaryon == 1) firstBaryon  = true;
		if(nbaryon == 2) secondBaryon = true;
	      }

	      unsigned int ip = partType[ii];
	      if(firstBaryon) {
		if (debug && partEne[ii]>200) std::cout << "nbaryon " << nbaryon << " ii " << ii << "  partType " << partType[ii] << " " << partEne[ii] << "\n";
		if(nPart[ip]==1) hiParticle[0][ip]->Fill(partEne[ii]);
	      }

	      firstBaryon = false;
	      if(secondBaryon){
		if (debug && partEne[ii]>200) std::cout << "nbaryon " << nbaryon << " ii " << ii << "  partType " << partType[ii] << " " << partEne[ii] << "\n";
		hiParticle[1][ip]->Fill(partEne[ii]);
	      }
	      secondBaryon = false;
	    } 
	  }

	} // ifInElastic
      }
    } // loop over entries

    gStyle->SetCanvasBorderMode(0); gStyle->SetCanvasColor(kWhite);
    gStyle->SetPadColor(kWhite);    gStyle->SetFrameBorderMode(0);
    gStyle->SetFrameBorderSize(1);  gStyle->SetFrameFillColor(0);
    gStyle->SetFrameFillStyle(0);   gStyle->SetFrameLineColor(1);
    gStyle->SetFrameLineStyle(1);   gStyle->SetFrameLineWidth(1);
    gStyle->SetOptLogy(1);          gStyle->SetTitleOffset(1.2,"Y");

    if (sav < 0) {
      TCanvas *cc1[20], *cc2[20], *cc3; 
      cc3 = new TCanvas("c_multiplicity", "c_multiplicity", 800, 800);
      TLegend *leg = new TLegend(0.5, 0.5, 0.8, 0.8);
      for (unsigned int iia=0; iia<=(types.size()); iia++) {
	if      (iia == 0) sprintf (ctype, "All Particles");
	else               sprintf (ctype, "%s", types[iia-1].c_str());

	sprintf (title, "%s in %s at %s GeV (%s)", ctype, element, ene, list);
	sprintf(name, "C-KE%i", iia);
	cc1[iia] = new TCanvas(name,title,800,600); cc1[iia]->Divide(2,2);
	cc1[iia]->cd(1); if (hiKE0[iia]->GetEntries() > 0) hiKE0[iia]->Draw(); 
	cc1[iia]->cd(3); if (hiKE1[iia]->GetEntries() > 0) hiKE1[iia]->Draw();
	cc1[iia]->cd(4); if (hiKE2[iia]->GetEntries() > 0) hiKE2[iia]->Draw(); 
	sprintf(name, "C-CT%i", iia);
	cc2[iia] = new TCanvas(name,title,800,600); cc2[iia]->Divide(2,2);
	cc2[iia]->cd(1); if (hiCT0[iia]->GetEntries() > 0) hiCT0[iia]->Draw(); 
	cc2[iia]->cd(3); if (hiCT1[iia]->GetEntries() > 0) hiCT1[iia]->Draw();
	cc2[iia]->cd(4); if (hiCT2[iia]->GetEntries() > 0) hiCT2[iia]->Draw(); 
	cc3->cd();
	hiMulti[iia]->SetLineColor(iia+1);
	if(iia>=9) hiMulti[iia]->SetLineColor(iia+2);
	if(iia==0) hiMulti[iia]->Draw();
	else       hiMulti[iia]->Draw("sames");
	leg->AddEntry(hiMulti[iia], title, "l");
      }
      cc3->cd();
      leg->Draw("same");

      if (detail) {
	TCanvas *cc4 = new TCanvas("c_Nucleon", "c_Nucleon", 800, 800);
	cc4->Divide(2,2);
	cc4->cd(1); hProton[0]->Draw();
	cc4->cd(2); hProton[1]->Draw();
	cc4->cd(3); hNeutron[0]->Draw();
	cc4->cd(4); hNeutron[1]->Draw();

	TCanvas *cc5 = new TCanvas("c_Ion", "c_Ion", 800, 800);
	cc5->Divide(2,2);
	cc5->cd(1); hHeavy[0]->Draw();
	cc5->cd(2); hHeavy[1]->Draw();
	cc5->cd(3); hIon[0]->Draw();
	cc5->cd(4); hIon[1]->Draw();
	
	TCanvas *cc6 = new TCanvas("c_Baryon", "c_Baryon", 800, 400);
	cc6->Divide(2,1);
	cc6->cd(1); hBaryon[0]->Draw();
	cc6->cd(2); hBaryon[1]->Draw();
      }

    } else {

      std::cout << "Writing histograms to " << ofile << "\n";
      fout->cd();
      fout->Write();
    }

    std::cout << ninter << " interactions seen in " << nentry << " trials\n"
	      << "Elastic/Inelastic " << elastic << "/" << inelastic << "\n";
    if( nentry-ninter != 0 ) {
      double sigma = atwt*10000.*log((double)(nentry)/(double)(nentry-ninter))/(rhol*6.023);
      double dsigma    = sigma/sqrt(double(ninter));
      double sigmaEl   = sigma*((double)(elastic))/((double)(ninter));
      double dsigmaEl  = sigmaEl/sqrt(double(max(1,elastic)));
      double sigmaInel = sigma*((double)(inelastic))/((double)(ninter));
      double dsigmaInel= sigmaInel/sqrt(double(max(1,inelastic)));
      std::cout << "Total     " << sigma << " +- " << dsigma 
		<< " mb (" << ninter << " events)\n"
		<< "Elastic   " << sigmaEl<< " +- " << dsigmaEl
		<< " mb (" << elastic << " events)\n"
		<< "Inelasric " << sigmaInel << " +- " << dsigmaInel
		<< " mb (" << inelastic << " events)\n";
    }

  } // if tree is found
} //end analysis()

double rhoL(char element[6]) {
  
  double tmp=0;
  if      (element == "Brass") tmp = 8.50 * 0.40;
  else if (element == "PbWO4") tmp = 8.28 * 0.30;
  else if (element == "Fe")    tmp = 7.87 * 0.30;
  else if (element == "H")     tmp = 0.0708 * 12.0;
  else if (element == "D")     tmp = 0.162  * 6.0;
  return tmp;
}

double atomicWt(char element[6]) {
  
  double tmp=0;
  if      (element == "Brass") tmp = 64.228;
  else if (element == "PbWO4") tmp = 455.036;
  else if (element == "Fe")    tmp = 55.85;
  else if (element == "H")     tmp = 1.0079;
  else if (element == "D")     tmp = 2.01;
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

std::vector<std::string> types() {

  std::vector<string> tmp;
  tmp.push_back("Photon/Neutrino");
  tmp.push_back("Electron");
  tmp.push_back("Positron");
  tmp.push_back("MuMinus");
  tmp.push_back("MuPlus");
  tmp.push_back("Pizero");
  tmp.push_back("Piminus");
  tmp.push_back("Piplus");
  tmp.push_back("Kminus");
  tmp.push_back("Kiplus");
  tmp.push_back("Kzero");
  tmp.push_back("AntiProton");
  tmp.push_back("Proton");
  tmp.push_back("Neutron/AntiNeutron");
  tmp.push_back("Heavy Hadrons");
  tmp.push_back("Ions");

  return tmp;
}

