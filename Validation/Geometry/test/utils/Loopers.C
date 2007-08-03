#define Loopers_cxx
#include "Loopers.h"
#include <TH2.h>
#include <TStyle.h>
#include <TCanvas.h>

#include <cmath>

//#define DEBUG

//using namespace std;

void Loopers::Loop()
{
  //   In a ROOT session, you can do:
  //      Root > .L Loopers.C
  //      Root > Loopers t
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
    
    double phiSum   = 0.0; // phi_i
    double phi_init = 0.0;
    double phi_last = 0.0;
    unsigned int n_loops = 0;
    
    // loop on steps
    //    cout << " Steps " << Nsteps << endl;
    for(Int_t iStep=0; iStep<Nsteps; iStep++) {
      // middle-point
      double xStep = (FinalX[iStep]+InitialX[iStep])/2;
      double yStep = (FinalY[iStep]+InitialY[iStep])/2;
      double zStep = (FinalZ[iStep]+InitialZ[iStep])/2;
      //
      double phi_i = atan2(yStep,xStep); // [-pi,+pi]
      double deltaPhi = (phi_last > 0 && phi_i < 0) ? (phi_i-phi_last)+(2*pi) : phi_i-phi_last; // it works only if phi is growing anti-clockwise
      phiSum += deltaPhi;
      //
      /*
	cout << "################################" << endl;
	cout << "\t step " << iStep                << endl;
	cout << "\t phi_i = " << phi_i             << endl;
	cout << "\t DeltaPhi = " << deltaPhi       << endl;
	cout << "\t PhiSum = " << phiSum           << endl;
	cout << "################################" << endl;
      */
      phi_last = phi_i;
    } // step loop
    
    double phiLoops = phiSum/(2*pi);
    phiLoops += (float)n_loops;
    
    if(Nsteps!=0) {
      /*
	cout << "################################"      << endl;
	cout << "\t Sum(phi_i) = "  << phiSum           << endl;
      	cout << "\t Loops = "       << phiLoops         << endl;
	cout << "\t Energy Init = " << InitialE[0]      << endl;
	cout << "\t Energy End = "  << FinalE[Nsteps-1] << endl;
	cout << "\t Last particle = " << ParticleStepID[Nsteps-1] << endl;
	cout << "\t Last step = "   << ParticleStepInteraction[Nsteps-1] << endl;
	cout << "################################"      << endl;
      */
      hist_loops->Fill(phiLoops);
      hist_energy_init->Fill(InitialE[0]);
      hist_energy_end->Fill(FinalE[Nsteps-1]);
      //
    }
    
  } // event loop
  
  // Draw plots
  MakePlots("Gigi");
  //
  
}


void Loopers::MakePlots(TString suffix) { 
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
  hist_loops->SetLineColor(kBlue);
  hist_loops->SetFillColor(kWhite);
  hist_loops->Draw();
  //  
  
  // Store
  can_Gigi.Update();
  can_Gigi.SaveAs( Form("%s/Loopers_Loops_%s.eps",  theDirName.Data(), suffix.Data() ) );
  can_Gigi.SaveAs( Form("%s/Loopers_Loops_%s.gif",  theDirName.Data(), suffix.Data() ) );
  // 
  
  // Draw
  can_Gigi.cd();
  hist_energy_init->SetLineColor(kBlue);
  hist_energy_init->SetFillColor(kWhite);
  hist_energy_init->Draw();
  //  
  
  // Store
  can_Gigi.Update();
  can_Gigi.SaveAs( Form("%s/Loopers_Energy_Init_%s.eps",  theDirName.Data(), suffix.Data() ) );
  can_Gigi.SaveAs( Form("%s/Loopers_Energy_Init_%s.gif",  theDirName.Data(), suffix.Data() ) );
  // 

  // Draw
  can_Gigi.cd();
  hist_energy_end->SetLineColor(kBlue);
  hist_energy_end->SetFillColor(kWhite);
  hist_energy_end->Draw();
  //  
  
  // Store
  can_Gigi.Update();
  can_Gigi.SaveAs( Form("%s/Loopers_Energy_End_%s.eps",  theDirName.Data(), suffix.Data() ) );
  can_Gigi.SaveAs( Form("%s/Loopers_Energy_End_%s.gif",  theDirName.Data(), suffix.Data() ) );
  // 
}

void Loopers::helpfulCommands() {
  cout << "########################################" << endl;
  cout << "a.Loop()"                                 << endl;
  cout << "########################################" << endl;
}
