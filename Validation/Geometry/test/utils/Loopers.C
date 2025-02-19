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
    
    double phiSum   = 1e-11; // phi_i [protection against division by zero]
    double phi_init = 0.0;
    double phi_last = 0.0;
    unsigned int n_loops = 0;
    // Density
    double pathSum     = 1e-11; // x_i
    double normDensSum = 0.0; // rho_i x x_i
    double pathSum_211  = 0.0; // x_i
    double pathSum_13   = 0.0; // x_i
    double pathSum_11   = 0.0; // x_i
    double pathSum_2212 = 0.0; // x_i
    // Lambda0
    double nuclSum     = 1e-11; // x_i / l0_i
    // Hadronic Interactions
    unsigned int iHadronicInteractions = 0;
    double energyLossHad = 0.0;
    // Secondaries
    int    iParticle    = -1;
    double particleMass = -1;
    
    // loop on steps
    //    theLogFile << " Steps " << Nsteps << endl;
    theLogFile << "Event " << jentry+1 << " Steps = " << Nsteps << endl;
    Int_t lastStep = 0;
    for(Int_t iStep=0; iStep<Nsteps; iStep++) {
      
      // x_i
      double x_i = sqrt(
			(FinalX[iStep]-InitialX[iStep]) * (FinalX[iStep]-InitialX[iStep])
			+
			(FinalY[iStep]-InitialY[iStep]) * (FinalY[iStep]-InitialY[iStep])
			+
			(FinalZ[iStep]-InitialZ[iStep]) * (FinalZ[iStep]-InitialZ[iStep])
			);
      
      // hadronic interactions
      if(ParticleStepPostInteraction[iStep] == 4) {
	iHadronicInteractions++;
	energyLossHad += (InitialE[iStep] - FinalE[iStep]);
      }
      
      // find emission
      if(iStep==0) actualParticleID = ParticleStepID[iStep];
      if(iStep!=0) {
	if(FinalZ[iStep-1] != InitialZ[iStep]) {
	  theLogFile << "\t\t EMISSION at " << iStep
		     << " from " << actualParticleID << " of a " << ParticleStepID[iStep] << endl;
	}
      }
      
      // find decay
      if(actualParticleID != ParticleStepID[iStep]) {
	theLogFile << "\t\t DECAY at " << iStep
		   << " from " << actualParticleID << " to " << ParticleStepID[iStep] << endl;
	actualParticleID = ParticleStepID[iStep];	
	if (fabs(ParticleStepID[iStep]) == 211) { // pi+-
	  iParticle = 1;
	}
	if (fabs(ParticleStepID[iStep]) == 13) {  // mu+-
	  iParticle = 2;
	}
	if (fabs(ParticleStepID[iStep]) == 11) {  // e+-
	  iParticle = 3;
	}
	if (fabs(ParticleStepID[iStep]) == 2212) { // p+-
	  iParticle = 4;
	}	
	hist_productionEnergy_vs_secondaryParticle->Fill((float)iParticle,(InitialE[iStep]-InitialM[iStep]));
      }
      
      // path sum
      if (fabs(ParticleStepID[iStep]) == 211) { // pi+-
	pathSum_211+=x_i;
      }
      if (fabs(ParticleStepID[iStep]) == 13) {  // mu+-
	pathSum_13+=x_i;
      }
      if (fabs(ParticleStepID[iStep]) == 11) {  // e+-
	pathSum_11+=x_i;
      }
      if (fabs(ParticleStepID[iStep]) == 2212) { // p+-
	pathSum_2212+=x_i;
      }
      
      
      // select secondary/decay particles: only pi/mu/e/p
      if( fabs(ParticleStepID[iStep])!=211
	  &&
	  fabs(ParticleStepID[iStep])!=13
	  &&
	  fabs(ParticleStepID[iStep])!=11
	  &&
	  fabs(ParticleStepID[iStep])!=2212
	  ) {
	//	theLogFile << "Skip particle " << ParticleStepID[iStep] << " step " << iStep << endl;
	continue;
      } else {
	// go on
      }
      
      // energy threshold
      if( ( ParticleStepID[0] == ParticleID )
	  &&
	  (
	   ( (FinalE[iStep]-InitialM[iStep]) < Ekin_threshold ) // lower energy
	   ||
	   ( ParticleStepPostInteraction[iStep] == 6 ) // decay
	   )
	  ) {
	if(lastStep == 0) {
	  theLogFile << "Set final step to " << iStep << " final E " << FinalE[iStep] << " MeV" << endl;
	  lastStep = iStep;
	}
	continue;
      }
      
      
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
      // rho_i
      double rho_i = MaterialDensity[iStep];
      pathSum     += x_i;
      normDensSum += (rho_i * x_i);
      // Lambda0
      double l0_i = MaterialLambda0[iStep];
      nuclSum += (x_i / l0_i);      
      
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
    
    theLogFile << "  Last Step = " << lastStep << endl;
    if(lastStep!=0) {
      // average density: Sum(x_i x rho_i) / Sum(x_i)
      double averageDensity = normDensSum / pathSum;
      double averageLambda0 = pathSum / nuclSum;
      double time = (pathSum/1000.) / c; // time: [s] ; pathSum: [mm]->[m] ; c: [m/s]
      double bunchCrossings = (time * 1E9) / bx; // time: [s]-->[ns] ; bx: [ns]
      //
      if(
	 (
	  ParticleStepFinalPt[lastStep] > 10 // The particle must disappear with pT=0 GeV
	  &&
	  FinalZ[lastStep] < 2500 // only if not at the end of the Tracker (along z)
	  )
	 ||
	 (
	  ParticleStepPostInteraction[lastStep] == 0
	  )
	 ) {
	theLogFile << "################################"                                      << endl;
	theLogFile << "Event " << jentry+1                                                    << endl;
	theLogFile << "\t Steps = " << Nsteps                                                 << endl;
	theLogFile << "\t Last Step = " << lastStep                                           << endl;
	theLogFile << "\t Sum(phi_i) = "  << phiSum                                           << endl;
	theLogFile << "\t Turns = "       << phiTurns                                         << endl;
	theLogFile << "\t Energy Init = " << InitialE[0]                                      << endl;
	theLogFile << "\t Energy End = "  << FinalE[lastStep]                                 << endl;
	theLogFile << "\t Final Z = "  << FinalZ[lastStep]                                    << endl;
	theLogFile << "\t Particle Mass = " << InitialM[0]                                    << endl;
	theLogFile << "\t pT Init = " << ParticleStepInitialPt[0]                             << endl;
	theLogFile << "\t pT End = "  << ParticleStepFinalPt[lastStep]                        << endl;
	theLogFile << "\t Last particle = " << ParticleStepID[lastStep]                       << endl;
	theLogFile << "\t Final Pre Interaction  = " << ParticleStepPreInteraction[lastStep]  << endl;
	theLogFile << "\t Final Post Interaction = " << ParticleStepPostInteraction[lastStep] << endl;
	theLogFile << "\t Sum(x_i) = "         << pathSum        << " mm"                     << endl;
	theLogFile << "\t Sum(x_i x rho_i) = " << normDensSum    << " mm x g/cm3"             << endl;
	theLogFile << "\t <rho> = "            << averageDensity << " g/cm3"                  << endl;
	theLogFile << "\t <lambda0> = "        << averageLambda0 << " mm"                     << endl;
	theLogFile << "\t path length per turn = " << pathSum / phiTurns << " mm"             << endl;
	theLogFile << "\t time spent = " << time << " s"                                      << endl;
	theLogFile << "\t bunch crossings = " << bunchCrossings                               << endl;
	theLogFile << "################################"                                      << endl;
      }
      // else { // The particle must disappear with pT=0 GeV
      hist_loops->Fill(phiTurns);
      hist_energy_init->Fill(InitialE[0]);
      hist_energy_end->Fill(FinalE[lastStep]);
      hist_energy_beforeend->Fill(InitialE[lastStep]);
      hist_density->Fill(averageDensity);
      hist_lambda0->Fill(averageLambda0);
      hist_density_vs_loops->Fill(phiTurns,averageDensity);
      hist_pT_init->Fill(ParticleStepInitialPt[0]);
      hist_pT_end->Fill(ParticleStepFinalPt[lastStep]);
      hist_energyLossPerTurn->Fill( (InitialE[0]-FinalE[lastStep]) / phiTurns ); // Energy = Kinetics + Mass (Final Energy=Mass)
      hist_trackLength->Fill(pathSum);
      hist_trackLengthPerTurn->Fill(pathSum / phiTurns);
      hist_lastInteraction->Fill(ParticleStepPostInteraction[lastStep]);
      hist_bx->Fill(bunchCrossings);
      hist_bx_finer->Fill(bunchCrossings);
      hist_energybeforeend_vs_lastInteraction->Fill(ParticleStepPostInteraction[lastStep],InitialE[lastStep]);
      hist_trackLength_vs_lastInteraction->Fill(ParticleStepPostInteraction[lastStep],pathSum);
      hist_hadronicInteractions->Fill((float)iHadronicInteractions);
      hist_hadronicInteractions_vs_lastInteraction->Fill(ParticleStepPostInteraction[lastStep],(float)iHadronicInteractions);
      hist_energyLossHadronicInteractions->Fill(energyLossHad);

      // Secondaries
      // cut away zeroes
      //      if(pathSum_211  == 0) pathSum_211  = -10.;
      //      if(pathSum_13   == 0) pathSum_13   = -10.;
      if( !(pathSum_11   > 0) ) pathSum_11   = -10000.;
      if( !(pathSum_2212 > 0) ) pathSum_2212 = -10000.;
      //
      if(pathSum_13 > 0){
	hist_bx_vs_secondaryParticle->Fill(1,-10.);
	hist_bx_vs_secondaryParticle->Fill(2,((pathSum_211+pathSum_13)/1000)/c*1E9/bx);
      } else {
	hist_bx_vs_secondaryParticle->Fill(1,(pathSum_211/1000)/c*1E9/bx);
	hist_bx_vs_secondaryParticle->Fill(2,-10.);
      }
      hist_bx_vs_secondaryParticle->Fill(3,(pathSum_11/1000)/c*1E9/bx);
      hist_bx_vs_secondaryParticle->Fill(4,(pathSum_2212/1000)/c*1E9/bx);
      //      } // sanity check final pT=0
      
    } // steps!=0
    
  } // event loop
  
  // Draw plots
  MakePlots("Gigi");
  //
  
}


void Loopers::MakePlots(TString suffix) { 
  //
  rootStyle();
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
  hist_energy_beforeend->SetLineColor(kBlue);
  hist_energy_beforeend->SetFillColor(kWhite);
  hist_energy_beforeend->Draw();
  //  
  // Print
  theLogFile << endl << "--------"     << endl;
  theLogFile << hist_energy_beforeend->GetTitle() << endl;
  theLogFile << "\t Mean = " << hist_energy_beforeend->GetMean()  << endl;
  theLogFile << "\t  RMS = " << hist_energy_beforeend->GetRMS()   << endl;
  //
  // Store
  can_Gigi.Update();
  can_Gigi.SaveAs( Form("%s/Loopers_Energy_BeforeEnd_%s.eps",  theDirName.Data(), suffix.Data() ) );
  can_Gigi.SaveAs( Form("%s/Loopers_Energy_BeforeEnd_%s.gif",  theDirName.Data(), suffix.Data() ) );
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
  hist_lambda0->SetLineColor(kBlue);
  hist_lambda0->SetFillColor(kWhite);
  hist_lambda0->Draw();
  //  
  // Print
  theLogFile << endl << "--------"     << endl;
  theLogFile << hist_lambda0->GetTitle() << endl;
  theLogFile << "\t Mean = " << hist_lambda0->GetMean()  << endl;
  theLogFile << "\t  RMS = " << hist_lambda0->GetRMS()   << endl;
  //
  // Store
  can_Gigi.Update();
  can_Gigi.SaveAs( Form("%s/Loopers_Lambda0_%s.eps",  theDirName.Data(), suffix.Data() ) );
  can_Gigi.SaveAs( Form("%s/Loopers_Lambda0_%s.gif",  theDirName.Data(), suffix.Data() ) );
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
  for(unsigned int iBin = 0; iBin<= hist_lastInteraction->GetNbinsX(); iBin++)
    theLogFile << "\t\t" << hist_lastInteraction->GetXaxis()->GetBinLabel(iBin) << " : "
	       << hist_lastInteraction->GetBinContent(iBin) / hist_lastInteraction->GetEntries()
	       << std::endl;
  //
  // Store
  can_Gigi.Update();
  can_Gigi.SaveAs( Form("%s/Loopers_lastInteraction_%s.eps",  theDirName.Data(), suffix.Data() ) );
  can_Gigi.SaveAs( Form("%s/Loopers_lastInteraction_%s.gif",  theDirName.Data(), suffix.Data() ) );
  // 
  
  // Draw
  can_Gigi.cd();
  hist_bx->SetLineColor(kBlue);
  hist_bx->SetFillColor(kWhite);
  hist_bx->Draw();
  //  
  // Print
  theLogFile << endl << "--------"     << endl;
  theLogFile << hist_bx->GetTitle() << endl;
  theLogFile << "\t Mean = " << hist_bx->GetMean()  << endl;
  theLogFile << "\t  RMS = " << hist_bx->GetRMS()   << endl;
  for(unsigned int iBin = 0; iBin<= hist_bx->GetNbinsX(); iBin++) {
    theLogFile << "\t\t bx " << hist_bx->GetBinCenter(iBin) << " : " << hist_bx->GetBinContent(iBin)
	       << " integral > " << hist_bx->GetBinCenter(iBin)
	       << " : " << hist_bx->Integral(iBin+1,hist_bx->GetNbinsX()+1) / hist_bx->Integral()
	       << std::endl;
  }
  //
  // Store
  can_Gigi.Update();
  can_Gigi.SaveAs( Form("%s/Loopers_bx_%s.eps",  theDirName.Data(), suffix.Data() ) );
  can_Gigi.SaveAs( Form("%s/Loopers_bx_%s.gif",  theDirName.Data(), suffix.Data() ) );
  // 
  
  // Draw
  can_Gigi.cd();
  hist_bx_finer->SetLineColor(kBlue);
  hist_bx_finer->SetFillColor(kWhite);
  hist_bx_finer->Draw();
  //  
  // Print
  theLogFile << endl << "--------"     << endl;
  theLogFile << hist_bx_finer->GetTitle() << endl;
  theLogFile << "\t Mean = " << hist_bx_finer->GetMean()  << endl;
  theLogFile << "\t  RMS = " << hist_bx_finer->GetRMS()   << endl;
  for(unsigned int iBin = 0; iBin<= hist_bx_finer->GetNbinsX(); iBin++) {
    theLogFile << "\t\t bx_finer " << hist_bx_finer->GetBinCenter(iBin) << " : " << hist_bx_finer->GetBinContent(iBin)
	       << " integral > " << hist_bx_finer->GetBinCenter(iBin)
	       << " : " << hist_bx_finer->Integral(iBin+1,hist_bx_finer->GetNbinsX()+1) / hist_bx_finer->Integral()
	       << std::endl;
  }
  //
  // Store
  can_Gigi.Update();
  can_Gigi.SaveAs( Form("%s/Loopers_bx_finer_%s.eps",  theDirName.Data(), suffix.Data() ) );
  can_Gigi.SaveAs( Form("%s/Loopers_bx_finer_%s.gif",  theDirName.Data(), suffix.Data() ) );
  // 
  
  // Prepare Axes Labels
  hist_energybeforeend_vs_lastInteraction->GetXaxis()->SetBinLabel( 1,"Not Defined");
  hist_energybeforeend_vs_lastInteraction->GetXaxis()->SetBinLabel( 2,"Transportation");
  hist_energybeforeend_vs_lastInteraction->GetXaxis()->SetBinLabel( 3,"Electromagnetic");
  hist_energybeforeend_vs_lastInteraction->GetXaxis()->SetBinLabel( 4,"Optical");
  hist_energybeforeend_vs_lastInteraction->GetXaxis()->SetBinLabel( 5,"Hadronic");
  hist_energybeforeend_vs_lastInteraction->GetXaxis()->SetBinLabel( 6,"Photolepton Hadron");
  hist_energybeforeend_vs_lastInteraction->GetXaxis()->SetBinLabel( 7,"Decay");
  hist_energybeforeend_vs_lastInteraction->GetXaxis()->SetBinLabel( 8,"General");
  hist_energybeforeend_vs_lastInteraction->GetXaxis()->SetBinLabel( 9,"Parameterisation");
  hist_energybeforeend_vs_lastInteraction->GetXaxis()->SetBinLabel(10,"User Defined");
  hist_energybeforeend_vs_lastInteraction->GetXaxis()->SetBinLabel(11,"Other");
  hist_energybeforeend_vs_lastInteraction->GetXaxis()->SetTitle("");
  // Draw
  can_Gigi.cd();
  hist_energybeforeend_vs_lastInteraction->SetLineColor(kBlue);
  hist_energybeforeend_vs_lastInteraction->SetFillColor(kBlue);
  hist_energybeforeend_vs_lastInteraction->Draw("BOX");
  //  
  // Print
  theLogFile << endl << "--------"     << endl;
  theLogFile << hist_energybeforeend_vs_lastInteraction->GetTitle() << endl;
  theLogFile << "\t Mean = " << hist_energybeforeend_vs_lastInteraction->GetMean()  << endl;
  theLogFile << "\t  RMS = " << hist_energybeforeend_vs_lastInteraction->GetRMS()   << endl;
  //
  // Store
  can_Gigi.Update();
  can_Gigi.SaveAs( Form("%s/Loopers_EnergyBeforeEnd_vs_lastInteraction_%s.eps",  theDirName.Data(), suffix.Data() ) );
  can_Gigi.SaveAs( Form("%s/Loopers_EnergyBeforeEnd_vs_lastInteraction_%s.gif",  theDirName.Data(), suffix.Data() ) );
  // 

  // Prepare Axes Labels
  hist_trackLength_vs_lastInteraction->GetXaxis()->SetBinLabel( 1,"Not Defined");
  hist_trackLength_vs_lastInteraction->GetXaxis()->SetBinLabel( 2,"Transportation");
  hist_trackLength_vs_lastInteraction->GetXaxis()->SetBinLabel( 3,"Electromagnetic");
  hist_trackLength_vs_lastInteraction->GetXaxis()->SetBinLabel( 4,"Optical");
  hist_trackLength_vs_lastInteraction->GetXaxis()->SetBinLabel( 5,"Hadronic");
  hist_trackLength_vs_lastInteraction->GetXaxis()->SetBinLabel( 6,"Photolepton Hadron");
  hist_trackLength_vs_lastInteraction->GetXaxis()->SetBinLabel( 7,"Decay");
  hist_trackLength_vs_lastInteraction->GetXaxis()->SetBinLabel( 8,"General");
  hist_trackLength_vs_lastInteraction->GetXaxis()->SetBinLabel( 9,"Parameterisation");
  hist_trackLength_vs_lastInteraction->GetXaxis()->SetBinLabel(10,"User Defined");
  hist_trackLength_vs_lastInteraction->GetXaxis()->SetBinLabel(11,"Other");
  hist_trackLength_vs_lastInteraction->GetXaxis()->SetTitle("");
  // Draw
  can_Gigi.cd();
  hist_trackLength_vs_lastInteraction->SetLineColor(kBlue);
  hist_trackLength_vs_lastInteraction->SetFillColor(kBlue);
  hist_trackLength_vs_lastInteraction->Draw("BOX");
  //  
  // Print
  theLogFile << endl << "--------"     << endl;
  theLogFile << hist_trackLength_vs_lastInteraction->GetTitle() << endl;
  theLogFile << "\t Mean = " << hist_trackLength_vs_lastInteraction->GetMean()  << endl;
  theLogFile << "\t  RMS = " << hist_trackLength_vs_lastInteraction->GetRMS()   << endl;
  //
  // Store
  can_Gigi.Update();
  can_Gigi.SaveAs( Form("%s/Loopers_trackLength_vs_lastInteraction_%s.eps",  theDirName.Data(), suffix.Data() ) );
  can_Gigi.SaveAs( Form("%s/Loopers_trackLength_vs_lastInteraction_%s.gif",  theDirName.Data(), suffix.Data() ) );
  // 

  // Draw
  can_Gigi.cd();
  hist_hadronicInteractions->SetLineColor(kBlue);
  hist_hadronicInteractions->SetFillColor(kWhite);
  hist_hadronicInteractions->Draw();
  //  
  // Print
  theLogFile << endl << "--------"     << endl;
  theLogFile << hist_hadronicInteractions->GetTitle() << endl;
  theLogFile << "\t Mean = " << hist_hadronicInteractions->GetMean()  << endl;
  theLogFile << "\t  RMS = " << hist_hadronicInteractions->GetRMS()   << endl;
  //
  // Store
  can_Gigi.Update();
  can_Gigi.SaveAs( Form("%s/Loopers_HadronicInteractions_%s.eps",  theDirName.Data(), suffix.Data() ) );
  can_Gigi.SaveAs( Form("%s/Loopers_HadronicInteractions_%s.gif",  theDirName.Data(), suffix.Data() ) );
  // 
  
  // Prepare Axes Labels
  hist_hadronicInteractions_vs_lastInteraction->GetXaxis()->SetBinLabel( 1,"Not Defined");
  hist_hadronicInteractions_vs_lastInteraction->GetXaxis()->SetBinLabel( 2,"Transportation");
  hist_hadronicInteractions_vs_lastInteraction->GetXaxis()->SetBinLabel( 3,"Electromagnetic");
  hist_hadronicInteractions_vs_lastInteraction->GetXaxis()->SetBinLabel( 4,"Optical");
  hist_hadronicInteractions_vs_lastInteraction->GetXaxis()->SetBinLabel( 5,"Hadronic");
  hist_hadronicInteractions_vs_lastInteraction->GetXaxis()->SetBinLabel( 6,"Photolepton Hadron");
  hist_hadronicInteractions_vs_lastInteraction->GetXaxis()->SetBinLabel( 7,"Decay");
  hist_hadronicInteractions_vs_lastInteraction->GetXaxis()->SetBinLabel( 8,"General");
  hist_hadronicInteractions_vs_lastInteraction->GetXaxis()->SetBinLabel( 9,"Parameterisation");
  hist_hadronicInteractions_vs_lastInteraction->GetXaxis()->SetBinLabel(10,"User Defined");
  hist_hadronicInteractions_vs_lastInteraction->GetXaxis()->SetBinLabel(11,"Other");
  hist_hadronicInteractions_vs_lastInteraction->GetXaxis()->SetTitle("");
  // Draw
  can_Gigi.cd();
  hist_hadronicInteractions_vs_lastInteraction->SetLineColor(kBlue);
  hist_hadronicInteractions_vs_lastInteraction->SetFillColor(kBlue);
  hist_hadronicInteractions_vs_lastInteraction->Draw("BOX");
  //  
  // Print
  theLogFile << endl << "--------"     << endl;
  theLogFile << hist_hadronicInteractions_vs_lastInteraction->GetTitle() << endl;
  //  theLogFile << "\t Mean = " << hist_hadronicInteractions_vs_lastInteraction->GetMean()  << endl;
  //  theLogFile << "\t  RMS = " << hist_hadronicInteractions_vs_lastInteraction->GetRMS()   << endl;
  //
  // Store
  can_Gigi.Update();
  can_Gigi.SaveAs( Form("%s/Loopers_HadronicInteractions_vs_lastInteraction_%s.eps",  theDirName.Data(), suffix.Data() ) );
  can_Gigi.SaveAs( Form("%s/Loopers_Hadronicinteractions_vs_lastInteraction_%s.gif",  theDirName.Data(), suffix.Data() ) );
  // 

  // Draw
  can_Gigi.cd();
  hist_energyLossHadronicInteractions->SetLineColor(kBlue);
  hist_energyLossHadronicInteractions->SetFillColor(kWhite);
  hist_energyLossHadronicInteractions->Draw();
  //  
  // Print
  theLogFile << endl << "--------"     << endl;
  theLogFile << hist_energyLossHadronicInteractions->GetTitle() << endl;
  theLogFile << "\t Mean = " << hist_energyLossHadronicInteractions->GetMean()  << endl;
  theLogFile << "\t  RMS = " << hist_energyLossHadronicInteractions->GetRMS()   << endl;
  //
  // Store
  can_Gigi.Update();
  can_Gigi.SaveAs( Form("%s/Loopers_EnergyLossHadronicInteractions_%s.eps",  theDirName.Data(), suffix.Data() ) );
  can_Gigi.SaveAs( Form("%s/Loopers_EnergyLossHadronicInteractions_%s.gif",  theDirName.Data(), suffix.Data() ) );
  // 

  // Prepare Axes Labels
  hist_productionEnergy_vs_secondaryParticle->GetXaxis()->SetBinLabel( 1,"#pi^{#pm}");
  hist_productionEnergy_vs_secondaryParticle->GetXaxis()->SetBinLabel( 2,"#mu^{#pm}");
  hist_productionEnergy_vs_secondaryParticle->GetXaxis()->SetBinLabel( 3,"e^{#pm}");
  hist_productionEnergy_vs_secondaryParticle->GetXaxis()->SetBinLabel( 4,"p/#bar{p}");
  // Draw
  can_Gigi.cd();
  hist_productionEnergy_vs_secondaryParticle->SetLineColor(kBlue);
  hist_productionEnergy_vs_secondaryParticle->SetFillColor(kBlue);
  hist_productionEnergy_vs_secondaryParticle->Draw("BOX");
  //  
  // Print
  theLogFile << endl << "--------"     << endl;
  theLogFile << hist_productionEnergy_vs_secondaryParticle->GetTitle() << endl;
  //  theLogFile << "\t Mean = " << hist_productionEnergy_vs_secondaryParticle->GetMean()  << endl;
  //  theLogFile << "\t  RMS = " << hist_productionEnergy_vs_secondaryParticle->GetRMS()   << endl;
  //
  // Store
  can_Gigi.Update();
  can_Gigi.SaveAs( Form("%s/Loopers_ProductionEnergy_vs_SecondaryParticle_%s.eps",  theDirName.Data(), suffix.Data() ) );
  can_Gigi.SaveAs( Form("%s/Loopers_ProductionEnergy_vs_SecondaryParticle_%s.gif",  theDirName.Data(), suffix.Data() ) );
  //
  
  // Draw
  can_Gigi.cd();
  hist_productionEnergy_vs_secondaryParticle->SetLineColor(kBlack);
  hist_productionEnergy_vs_secondaryParticle->SetMarkerColor(kBlue);
  hist_productionEnergy_vs_secondaryParticle->SetMarkerStyle(20);
  hist_productionEnergy_vs_secondaryParticle->Draw("AXIS");
  hist_productionEnergy_vs_secondaryParticle->ProfileX()->Draw("E1,SAME");
  // Store
  can_Gigi.Update();
  can_Gigi.SaveAs( Form("%s/Loopers_ProductionEnergy_vs_SecondaryParticle_profile_%s.eps",  theDirName.Data(), suffix.Data() ) );
  can_Gigi.SaveAs( Form("%s/Loopers_ProductionEnergy_vs_SecondaryParticle_profile_%s.gif",  theDirName.Data(), suffix.Data() ) );
  //
  
  // Prepare Axes Labels
  hist_bx_vs_secondaryParticle->GetXaxis()->SetBinLabel( 1,"#pi^{#pm}");
  hist_bx_vs_secondaryParticle->GetXaxis()->SetBinLabel( 2,"#mu^{#pm}");
  hist_bx_vs_secondaryParticle->GetXaxis()->SetBinLabel( 3,"e^{#pm}");
  hist_bx_vs_secondaryParticle->GetXaxis()->SetBinLabel( 4,"p/#bar{p}");
  // Draw
  can_Gigi.cd();
  hist_bx_vs_secondaryParticle->GetYaxis()->SetRangeUser(0.,5.);
  hist_bx_vs_secondaryParticle->SetLineColor(kBlue);
  hist_bx_vs_secondaryParticle->SetFillColor(kBlue);
  hist_bx_vs_secondaryParticle->Draw("BOX");
  //  
  // Print
  theLogFile << endl << "--------"     << endl;
  theLogFile << hist_bx_vs_secondaryParticle->GetTitle() << endl;
  //  theLogFile << "\t Mean = " << hist_bx_vs_secondaryParticle->GetMean()  << endl;
  //  theLogFile << "\t  RMS = " << hist_bx_vs_secondaryParticle->GetRMS()   << endl;
  //

  // Store
  can_Gigi.Update();
  can_Gigi.SaveAs( Form("%s/Loopers_bx_vs_SecondaryParticle_%s.eps",  theDirName.Data(), suffix.Data() ) );
  can_Gigi.SaveAs( Form("%s/Loopers_bx_vs_SecondaryParticle_%s.gif",  theDirName.Data(), suffix.Data() ) );
  //
  // Draw
  can_Gigi.cd();
  hist_bx_vs_secondaryParticle->SetLineColor(kBlack);
  hist_bx_vs_secondaryParticle->SetMarkerColor(kBlue);
  hist_bx_vs_secondaryParticle->SetMarkerStyle(20);
  hist_bx_vs_secondaryParticle->Draw("AXIS");
  hist_bx_vs_secondaryParticle->ProfileX()->Draw("E1,SAME");
  // Store
  can_Gigi.Update();
  can_Gigi.SaveAs( Form("%s/Loopers_bx_vs_SecondaryParticle_profile_%s.eps",  theDirName.Data(), suffix.Data() ) );
  can_Gigi.SaveAs( Form("%s/Loopers_bx_vs_SecondaryParticle_profile_%s.gif",  theDirName.Data(), suffix.Data() ) );
  //

  // only 1/2 pi/mu
  hist_bx_vs_secondaryParticle->GetXaxis()->SetRangeUser(0.5,1.5);
  hist_bx_vs_secondaryParticle->Draw("BOX");
  // Store
  can_Gigi.Update();
  can_Gigi.SaveAs( Form("%s/Loopers_bx_vs_SecondaryParticle_pimu_%s.eps",  theDirName.Data(), suffix.Data() ) );
  can_Gigi.SaveAs( Form("%s/Loopers_bx_vs_SecondaryParticle_pimu_%s.gif",  theDirName.Data(), suffix.Data() ) );
  //
  // only 4 proton
  hist_bx_vs_secondaryParticle->GetXaxis()->SetRangeUser(3.5,4.5);
  hist_bx_vs_secondaryParticle->Draw("BOX");
  // Store
  can_Gigi.Update();
  can_Gigi.SaveAs( Form("%s/Loopers_bx_vs_SecondaryParticle_p_%s.eps",  theDirName.Data(), suffix.Data() ) );
  can_Gigi.SaveAs( Form("%s/Loopers_bx_vs_SecondaryParticle_p_%s.gif",  theDirName.Data(), suffix.Data() ) );
  //
  // only 4 proton
  hist_bx_vs_secondaryParticle->GetXaxis()->SetRangeUser(3.5,4.5);
  hist_bx_vs_secondaryParticle->ProjectionY(" ",4,4)->GetXaxis()->SetRangeUser(0.,5.); // bx axis
  hist_bx_vs_secondaryParticle->ProjectionY(" ",4,4)->Draw();
  // Store
  can_Gigi.Update();
  can_Gigi.SaveAs( Form("%s/Loopers_bx_vs_SecondaryParticle_projection_p_%s.eps",  theDirName.Data(), suffix.Data() ) );
  can_Gigi.SaveAs( Form("%s/Loopers_bx_vs_SecondaryParticle_projection_p_%s.gif",  theDirName.Data(), suffix.Data() ) );
  //
  
  
}

void Loopers::rootStyle() {
  // rrStyle
  TStyle* rrStyle = new TStyle("rootStyle","rootStyle");
  TGaxis::SetMaxDigits(3);          // to avoid too much decimal digits
  rrStyle->SetOptStat(0000);        // general statistics
  rrStyle->SetOptFit(1111);         // fit statistics
  rrStyle->SetOptLogy(0);           // logscale
  rrStyle->SetCanvasColor(kWhite);  // white canvas
  rrStyle->SetHistFillColor(34);    // histo: blue gray filling
  rrStyle->SetFuncColor(146);       // function: dark red line
  //
  rrStyle->SetLabelSize(0.04,"x,y,z");
  rrStyle->SetTitleSize(0.05,"x,y,z");
  rrStyle->SetTitleOffset(0.8,"x,y,z");
  rrStyle->SetTitleFontSize(0.06);
  //
  rrStyle->SetHistLineWidth(1);
  //
  rrStyle->SetPaintTextFormat("g");
  //
  rrStyle->SetTitleBorderSize(0);
  rrStyle->SetTitleFillColor(0);
  rrStyle->SetTitleFont(12,"pad");
  rrStyle->SetTitleFontSize(0.04);
  rrStyle->SetTitleX(0.075);
  rrStyle->SetTitleY(0.950);
  //
  rrStyle->SetLegendBorderSize(0); // no legend border
  rrStyle->SetFillColor(0); // all the filling colors are white
  //
  
  // ROOT macro
  gROOT->SetBatch();
  gROOT->SetStyle("rootStyle");
  //
}

void Loopers::helpfulCommands() {
  cout << "########################################" << endl;
  cout << "a.Loop()"                                 << endl;
  cout << "########################################" << endl;
}
