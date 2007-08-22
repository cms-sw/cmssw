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
  
  theLogFile << " Entries " << nentries << endl;
  
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
    // Density
    double pathSum     = 0.0; // x_i
    double normDensSum = 0.0; // rho_i x x_i
    
    // loop on steps
    //    theLogFile << " Steps " << Nsteps << endl;
    theLogFile << "Event " << jentry+1 << " Steps = " << Nsteps << endl;
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
      // Density
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
      pathSum     += x_i;
      normDensSum += (rho_i * x_i);
      /*
	theLogFile << "################################" << endl;
	theLogFile << "\t step " << iStep                << endl;
	theLogFile << "\t phi_i = " << phi_i             << endl;
	theLogFile << "\t DeltaPhi = " << deltaPhi       << endl;
	theLogFile << "\t PhiSum = " << phiSum           << endl;
	theLogFile << "\t x_i = "   << x_i   << " mm"    << endl;
	theLogFile << "\t rho_i = " << rho_i << " g/cm3" << endl;
	theLogFile << "################################" << endl;
      */
      phi_last = phi_i;
    } // step loop
    
    double phiTurns = phiSum/(2*pi);
    phiTurns += (float)n_loops;
    
    if(Nsteps!=0) {
      // average density: Sum(x_i x rho_i) / Sum(x_i)
      double averageDensity = normDensSum / pathSum;
      //
      if(
	 (
	  ParticleStepFinalPt[Nsteps-1] > 10 // The particle must disappear with pT=0 GeV
	  &&
	  FinalZ[Nsteps-1] < 2500 // only if not at the end of the Tracker (along z)
	  ) 
	 ) { 
	theLogFile << "################################"                                      << endl;
	theLogFile << "Event " << jentry+1                                                    << endl;
	theLogFile << "\t Steps = " << Nsteps                                                 << endl;
	theLogFile << "\t Sum(phi_i) = "  << phiSum                                           << endl;
	theLogFile << "\t Turns = "       << phiTurns                                         << endl;
	theLogFile << "\t Energy Init = " << InitialE[0]                                      << endl;
	theLogFile << "\t Energy End = "  << FinalE[Nsteps-1]                                 << endl;
	theLogFile << "\t Final Z = "  << FinalZ[Nsteps-1]                                    << endl;
	theLogFile << "\t Particle Mass = " << ParticleMass                                   << endl;
	theLogFile << "\t pT Init = " << ParticleStepInitialPt[0]                             << endl;
	theLogFile << "\t pT End = "  << ParticleStepFinalPt[Nsteps-1]                        << endl;
	theLogFile << "\t Last particle = " << ParticleStepID[Nsteps-1]                       << endl;
	theLogFile << "\t Final Pre Interaction  = " << ParticleStepPreInteraction[Nsteps-1]  << endl;
	theLogFile << "\t Final Post Interaction = " << ParticleStepPostInteraction[Nsteps-1] << endl;
	theLogFile << "\t Sum(x_i) = "         << pathSum        << " mm"                     << endl;
	theLogFile << "\t Sum(x_i x rho_i) = " << normDensSum    << " mm x g/cm3"             << endl;
	theLogFile << "\t <rho> = "            << averageDensity << " g/cm3"                  << endl;
	theLogFile << "\t path length per turn = " << pathSum / phiTurns << " mm"             << endl;
	theLogFile << "################################"                                      << endl;
      }
      // else { // The particle must disappear with pT=0 GeV
      hist_loops->Fill(phiTurns);
      hist_energy_init->Fill(InitialE[0]);
      hist_energy_end->Fill(FinalE[Nsteps-1]);
      hist_density->Fill(averageDensity);
      hist_density_vs_loops->Fill(phiTurns,averageDensity);
      hist_pT_init->Fill(ParticleStepInitialPt[0]);
      hist_pT_end->Fill(ParticleStepFinalPt[Nsteps-1]);
      hist_energyLossPerTurn->Fill( (InitialE[0]-FinalE[Nsteps-1]) / phiTurns ); // Energy = Kinetics + Mass (Final Energy=Mass)
      hist_trackLength->Fill(pathSum);
      hist_trackLengthPerTurn->Fill(pathSum / phiTurns);
      hist_lastInteraction->Fill(ParticleStepPostInteraction[Nsteps-1]);
      //      } // sanity check final pT=0
      
    } // steps!=0
    
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
  // Print
  theLogFile << endl << "--------"     << endl;
  theLogFile << hist_loops->GetTitle() << endl;
  theLogFile << "\t Mean = " << hist_loops->GetMean()  << endl;
  theLogFile << "\t  RMS = " << hist_loops->GetRMS()   << endl;
  //
  // Store
  can_Gigi.Update();
  can_Gigi.SaveAs( Form("%s/Loopers_Turns_%s.eps",  theDirName.Data(), suffix.Data() ) );
  can_Gigi.SaveAs( Form("%s/Loopers_Turns_%s.gif",  theDirName.Data(), suffix.Data() ) );
  // 
  
  // Draw
  can_Gigi.cd();
  hist_energy_init->SetLineColor(kBlue);
  hist_energy_init->SetFillColor(kWhite);
  hist_energy_init->Draw();
  //  
  // Print
  theLogFile << endl << "--------"     << endl;
  theLogFile << hist_energy_init->GetTitle() << endl;
  theLogFile << "\t Mean = " << hist_energy_init->GetMean()  << endl;
  theLogFile << "\t  RMS = " << hist_energy_init->GetRMS()   << endl;
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
  // Print
  theLogFile << endl << "--------"     << endl;
  theLogFile << hist_energy_end->GetTitle() << endl;
  theLogFile << "\t Mean = " << hist_energy_end->GetMean()  << endl;
  theLogFile << "\t  RMS = " << hist_energy_end->GetRMS()   << endl;
  //
  // Store
  can_Gigi.Update();
  can_Gigi.SaveAs( Form("%s/Loopers_Energy_End_%s.eps",  theDirName.Data(), suffix.Data() ) );
  can_Gigi.SaveAs( Form("%s/Loopers_Energy_End_%s.gif",  theDirName.Data(), suffix.Data() ) );
  // 

  // Draw
  can_Gigi.cd();
  hist_density->SetLineColor(kBlue);
  hist_density->SetFillColor(kWhite);
  hist_density->Draw();
  //  
  // Print
  theLogFile << endl << "--------"     << endl;
  theLogFile << hist_density->GetTitle() << endl;
  theLogFile << "\t Mean = " << hist_density->GetMean()  << endl;
  theLogFile << "\t  RMS = " << hist_density->GetRMS()   << endl;
  //
  // Store
  can_Gigi.Update();
  can_Gigi.SaveAs( Form("%s/Loopers_Density_%s.eps",  theDirName.Data(), suffix.Data() ) );
  can_Gigi.SaveAs( Form("%s/Loopers_Density_%s.gif",  theDirName.Data(), suffix.Data() ) );
  // 

  // Draw
  can_Gigi.cd();
  hist_density_vs_loops->SetLineColor(kBlue);
  hist_density_vs_loops->SetFillColor(kBlue);
  hist_density_vs_loops->Draw("BOX");
  //  
  // Print
  theLogFile << endl << "--------"     << endl;
  theLogFile << hist_density_vs_loops->GetTitle() << endl;
  theLogFile << "\t Mean = " << hist_density_vs_loops->GetMean()  << endl;
  theLogFile << "\t  RMS = " << hist_density_vs_loops->GetRMS()   << endl;
  //
  // Store
  can_Gigi.Update();
  can_Gigi.SaveAs( Form("%s/Loopers_Density_vs_Turns_%s.eps",  theDirName.Data(), suffix.Data() ) );
  can_Gigi.SaveAs( Form("%s/Loopers_Density_vs_Turns_%s.gif",  theDirName.Data(), suffix.Data() ) );
  // 

  // Draw
  can_Gigi.cd();
  hist_pT_init->SetLineColor(kBlue);
  hist_pT_init->SetFillColor(kWhite);
  hist_pT_init->Draw();
  //  
  // Print
  theLogFile << endl << "--------"     << endl;
  theLogFile << hist_pT_init->GetTitle() << endl;
  theLogFile << "\t Mean = " << hist_pT_init->GetMean()  << endl;
  theLogFile << "\t  RMS = " << hist_pT_init->GetRMS()   << endl;
  //
  // Store
  can_Gigi.Update();
  can_Gigi.SaveAs( Form("%s/Loopers_pT_Init_%s.eps",  theDirName.Data(), suffix.Data() ) );
  can_Gigi.SaveAs( Form("%s/Loopers_pT_Init_%s.gif",  theDirName.Data(), suffix.Data() ) );
  // 

  // Draw
  can_Gigi.cd();
  hist_pT_end->SetLineColor(kBlue);
  hist_pT_end->SetFillColor(kWhite);
  hist_pT_end->Draw();
  //  
  // Print
  theLogFile << endl << "--------"     << endl;
  theLogFile << hist_pT_end->GetTitle() << endl;
  theLogFile << "\t Mean = " << hist_pT_end->GetMean()  << endl;
  theLogFile << "\t  RMS = " << hist_pT_end->GetRMS()   << endl;
  //
  // Store
  can_Gigi.Update();
  can_Gigi.SaveAs( Form("%s/Loopers_pT_End_%s.eps",  theDirName.Data(), suffix.Data() ) );
  can_Gigi.SaveAs( Form("%s/Loopers_pT_End_%s.gif",  theDirName.Data(), suffix.Data() ) );
  // 

  // Draw
  can_Gigi.cd();
  hist_energyLossPerTurn->SetLineColor(kBlue);
  hist_energyLossPerTurn->SetFillColor(kWhite);
  hist_energyLossPerTurn->Draw();
  //  
  // Print
  theLogFile << endl << "--------"     << endl;
  theLogFile << hist_energyLossPerTurn->GetTitle() << endl;
  theLogFile << "\t Mean = " << hist_energyLossPerTurn->GetMean()  << endl;
  theLogFile << "\t  RMS = " << hist_energyLossPerTurn->GetRMS()   << endl;
  //
  // Store
  can_Gigi.Update();
  can_Gigi.SaveAs( Form("%s/Loopers_EnergyLossPerTurn_%s.eps",  theDirName.Data(), suffix.Data() ) );
  can_Gigi.SaveAs( Form("%s/Loopers_EnergyLossPerTurn_%s.gif",  theDirName.Data(), suffix.Data() ) );
  // 
  
  // Draw
  can_Gigi.cd();
  hist_trackLength->SetLineColor(kBlue);
  hist_trackLength->SetFillColor(kWhite);
  hist_trackLength->Draw();
  //  
  // Print
  theLogFile << endl << "--------"     << endl;
  theLogFile << hist_trackLength->GetTitle() << endl;
  theLogFile << "\t Mean = " << hist_trackLength->GetMean()  << endl;
  theLogFile << "\t  RMS = " << hist_trackLength->GetRMS()   << endl;
  //
  // Store
  can_Gigi.Update();
  can_Gigi.SaveAs( Form("%s/Loopers_trackLength_%s.eps",  theDirName.Data(), suffix.Data() ) );
  can_Gigi.SaveAs( Form("%s/Loopers_trackLength_%s.gif",  theDirName.Data(), suffix.Data() ) );
  // 
  
  // Draw
  can_Gigi.cd();
  hist_trackLengthPerTurn->SetLineColor(kBlue);
  hist_trackLengthPerTurn->SetFillColor(kWhite);
  hist_trackLengthPerTurn->Draw();
  //  
  // Print
  theLogFile << endl << "--------"     << endl;
  theLogFile << hist_trackLengthPerTurn->GetTitle() << endl;
  theLogFile << "\t Mean = " << hist_trackLengthPerTurn->GetMean()  << endl;
  theLogFile << "\t  RMS = " << hist_trackLengthPerTurn->GetRMS()   << endl;
  //
  // Store
  can_Gigi.Update();
  can_Gigi.SaveAs( Form("%s/Loopers_trackLengthPerTurn_%s.eps",  theDirName.Data(), suffix.Data() ) );
  can_Gigi.SaveAs( Form("%s/Loopers_trackLengthPerTurn_%s.gif",  theDirName.Data(), suffix.Data() ) );
  // 
  
  // Prepare Axes Labels
  hist_lastInteraction->GetXaxis()->SetBinLabel( 1,"Not Defined");
  hist_lastInteraction->GetXaxis()->SetBinLabel( 2,"Transportation");
  hist_lastInteraction->GetXaxis()->SetBinLabel( 3,"Electromagnetic");
  hist_lastInteraction->GetXaxis()->SetBinLabel( 4,"Optical");
  hist_lastInteraction->GetXaxis()->SetBinLabel( 5,"Hadronic");
  hist_lastInteraction->GetXaxis()->SetBinLabel( 6,"Photolepton Hadron");
  hist_lastInteraction->GetXaxis()->SetBinLabel( 7,"Decay");
  hist_lastInteraction->GetXaxis()->SetBinLabel( 8,"General");
  hist_lastInteraction->GetXaxis()->SetBinLabel( 9,"Parameterisation");
  hist_lastInteraction->GetXaxis()->SetBinLabel(10,"User Defined");
  hist_lastInteraction->GetXaxis()->SetBinLabel(11,"Other");
  hist_lastInteraction->GetXaxis()->SetTitle("");
  
  // Draw
  can_Gigi.cd();
  hist_lastInteraction->SetLineColor(kBlue);
  hist_lastInteraction->SetFillColor(kWhite);
  hist_lastInteraction->Draw();
  //  
  // Print
  theLogFile << endl << "--------"     << endl;
  theLogFile << hist_lastInteraction->GetTitle() << endl;
  theLogFile << "\t Mean = " << hist_lastInteraction->GetMean()  << endl;
  theLogFile << "\t  RMS = " << hist_lastInteraction->GetRMS()   << endl;
  //
  // Store
  can_Gigi.Update();
  can_Gigi.SaveAs( Form("%s/Loopers_lastInteraction_%s.eps",  theDirName.Data(), suffix.Data() ) );
  can_Gigi.SaveAs( Form("%s/Loopers_lastInteraction_%s.gif",  theDirName.Data(), suffix.Data() ) );
  // 
  
}

void Loopers::helpfulCommands() {
  cout << "########################################" << endl;
  cout << "a.Loop()"                                 << endl;
  cout << "########################################" << endl;
}
