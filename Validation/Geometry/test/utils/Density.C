#define Density_cxx
#include "Density.h"
#include <TH2.h>
#include <TStyle.h>
#include <TCanvas.h>

#include <cmath>

//#define DEBUG

//using namespace std;

void Density::Loop()
{
  //   In a ROOT session, you can do:
  //      Root > .L Density.C
  //      Root > Density t
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
  if (fChain == 0) return;
  
  Long64_t nentries = fChain->GetEntriesFast();
  
  cout << " Entries " << nentries << endl;
  
  Long64_t nbytes = 0, nb = 0;
  for (Long64_t jentry=0; jentry<nentries; jentry++) {
    // load tree variables
    Long64_t ientry = LoadTree(jentry);
    if (ientry < 0) break;
    nb = fChain->GetEntry(jentry);   nbytes += nb;
    //
    
    double pathSum     = 0.0; // x_i
    double normDensSum = 0.0; // rho_i x x_i
    //    cout << " Steps " << Nsteps << endl;
    
    // loop on steps
    for(Int_t iStep=0; iStep<Nsteps; iStep++) {
      //      cout << iStep << endl;
      // x_i
      double x_i = sqrt(
			(FinalX[iStep]-InitialX[iStep]) * (FinalX[iStep]-InitialX[iStep])
			+
			(FinalY[iStep]-InitialY[iStep]) * (FinalY[iStep]-InitialY[iStep])
			+
			(FinalZ[iStep]-InitialZ[iStep]) * (FinalZ[iStep]-InitialZ[iStep])
			);
      // rho_i
      double rho_i = MaterialDensity[iStep];
      /*
	cout << "################################" << endl;
	cout << "\t x_i = "   << x_i   << " mm"    << endl;
	cout << "\t rho_i = " << rho_i << " g/cm3" << endl;
	cout << "################################" << endl;
      */
      pathSum     += x_i;
      normDensSum += (rho_i * x_i);
      //
    } // step loop
    
    if(Nsteps!=0) {
      // average density: Sum(x_i x rho_i) / Sum(x_i)
      double averageDensity = normDensSum / pathSum;
      /*
	cout << "################################"         << endl;
	cout << "\t eta = "              << ParticleEta    << endl;
	cout << "\t Sum(x_i) = "         << pathSum        << " mm"         << endl;
	cout << "\t Sum(x_i x rho_i) = " << normDensSum    << " mm x g/cm3" << endl;
	cout << "\t <rho> = "            << averageDensity << " g/cm3"      << endl;
	cout << "################################"         << endl;
      */
      prof_density_vs_eta->Fill(fabs(ParticleEta),averageDensity);
      //
    }
    
  } // event loop
  
  // Draw plots
  MakePlots("Gigi");
  //
  
}


void Density::MakePlots(TString suffix) { 
  //
  TGaxis::SetMaxDigits(3);
  gStyle->SetOptStat(0);
  gStyle->SetOptFit(0);
  gStyle->SetOptLogy(0);
  //
  //
  // canvas
  TCanvas can_Gigi("can_Gigi","can_Gigi",1300,800);
  //  can_Gigi.Range(0,0,25,25);
  can_Gigi.SetFillColor(kWhite);
  //
  
  // Draw
  can_Gigi.cd();
  prof_density_vs_eta->SetMarkerColor(kBlue);
  prof_density_vs_eta->SetMarkerStyle(20);
  prof_density_vs_eta->Draw("E1");
  //  
  
  // Store
  can_Gigi.Update();
  can_Gigi.SaveAs( Form("%s/AverageDensity_%s.eps",  theDirName.Data(), suffix.Data() ) );
  can_Gigi.SaveAs( Form("%s/AverageDensity_%s.gif",  theDirName.Data(), suffix.Data() ) );
  // 
}

void Density::helpfulCommands() {
  cout << "########################################" << endl;
  cout << "a.Loop()"                                 << endl;
  cout << "########################################" << endl;
}
