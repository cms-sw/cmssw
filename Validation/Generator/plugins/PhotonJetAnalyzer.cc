#include "Validation/Generator/plugins/PhotonJetAnalyzer.h"

using namespace reco;
using namespace std;

PhotonJetAnalyzer::PhotonJetAnalyzer(const edm::ParameterSet& iConfig)
{
}

PhotonJetAnalyzer::~PhotonJetAnalyzer()
{
}

void
PhotonJetAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
   using namespace edm;

   // ***************************************************************************
   // ***** Collect Generated Particle data *************************************

   // Get Gen Particle Collection


   basicStruct mcHardPhoton;
   mcHardPhoton.pt = -1.0;
   int mcHardPhotonMomID = 0;
   bool dtctbleDrctPhtnFnd = false;

   /*
   Handle<GenParticleCollection> genParticles;
   iEvent.getByLabel("genParticles", genParticles);
   */
   Handle<CandidateCollection> genParticles;
   iEvent.getByLabel( "genParticleCandidates", genParticles );
   if (!genParticles.isValid()) {
     cout << "No " << "genParticleCandidates" << " found!" << endl;
   }

   for(size_t i = 0; (i < genParticles->size()) && (!dtctbleDrctPhtnFnd); ++ i) {


     const Candidate & p = (*genParticles)[ i ];

     int id     = p.pdgId();
     int status = p.status();
     double et  = p.et();
     double eta = p.eta();
     double phi = p.phi(); 

     // Find gen photon, with a minimum et of 20 GeV.
     // Detectable particles have status of 1.
     if ( (id == 22) && (status == 1) && (et > mcHardPhoton.pt) && (et > 20) ) {

       const Candidate * mom = p.mother();
       int motherID = mom->pdgId();

       if (motherID < 25) {
	 dtctbleDrctPhtnFnd = true;
	 mcHardPhoton.et  = et; // et and pt are same for photon
	 mcHardPhoton.phi = phi;
	 mcHardPhoton.eta = eta;
	 mcHardPhotonMomID = motherID;
       }

     } // End of IF a photon 
   }
   /*
   cout << "Photon = " << mcHardPhoton.et << "\t" << mcHardPhoton.eta << "\t" << mcHardPhoton.phi << endl;
   */
   // ***** End Collect Generated Pharticle data ********************************
   // ***************************************************************************



   // ***************************************************************************                                            
   // ***** Collect Gen Jet data ************************************************                                            

   // get gen jet collection                              
   Handle<GenJetCollection> jetsgen;
   iEvent.getByLabel("iterativeCone5GenJetsPt10", jetsgen);
   if (!jetsgen.isValid()) {
     cout << "No " << "iterativeCone5GenJetsPt10" << " found!" << endl;
   }

   basicStruct hiGenJet;
   hiGenJet.pt = -1.0;

   int hiGenJetCount = 0;

   if (jetsgen.isValid() ) {
     for (size_t j = 0; j < jetsgen->size(); j++) {
       double pt  = (*jetsgen)[j].pt();
       double eta = (*jetsgen)[j].eta();
       double phi = (*jetsgen)[j].phi();

       // Count the number of gen jets above 10 GeV                                                                            
       if (pt > 10) {
	 hiGenJetCount++;
       }

       // Only care about storing jet with reasonable pt, and opposite phi
       if ( (pt<10) || (fabs(phi - mcHardPhoton.phi) < 0.8*3.14159) || (fabs(phi - mcHardPhoton.phi) > 1.2*3.14159) ) continue;

       // Record this jet if it's gives the highest pt
       if (pt > hiGenJet.pt) {
	 hiGenJet.pt  = pt;
	 hiGenJet.eta = eta;
	 hiGenJet.phi = phi;
       }
     }
   }
   /*
   cout << "Found " << hiGenJetCount << " gen jets." << endl;
   cout << "Gen Jet        :\t" << hiGenJet.pt << "\t" << hiGenJet.eta << "\t" << hiGenJet.phi << endl;
   */
   // ***** END Collect Gen Jet data ********************************************
   // ***************************************************************************



   // ***************************************************************************
   // ***** Fill Histograms *****************************************************
   if ( (mcHardPhoton.et > 0.0) && (fabs(mcHardPhoton.eta) < 2.5) ) {
     cout << "Filling histograms...";
     hGenPhtnHardEt-> Fill(mcHardPhoton.et);
     hGenPhtnHardEta->Fill(mcHardPhoton.eta);
     hGenPhtnHardPhi->Fill(mcHardPhoton.phi);
     hGenPhtnHardMom->Fill(mcHardPhotonMomID);

     // Highest pt gen jet                                                                                                      
     if (hiGenJet.pt > 0.0) {
       hHIGenJetPt->  Fill(hiGenJet.pt);
       hHIGenJetEta-> Fill(hiGenJet.eta);
       hHIGenJetPhi-> Fill(hiGenJet.phi);
       hGenPhtnHardDrJet->Fill(calcDeltaR(mcHardPhoton.eta, mcHardPhoton.phi, hiGenJet.eta, hiGenJet.phi));
     }
     hHIGenJetCnt-> Fill(hiGenJetCount);
     cout << "...filled." << endl;
   }
   // ***** END Fill Histograms *************************************************
   // ***************************************************************************
}

void 
PhotonJetAnalyzer::beginJob()
{

  // Plotting variables
  const float MAX_E     = 800.0;
  const float MIN_E     =   0.0;
  const float MAX_ETA   =   3.29867223;
  const float MAX_PHI   =   3.29867223;
  const int HI_NUM_BINS =  80;
  const int ME_NUM_BINS =  42;
  //  const int LO_NUM_BINS =  25;

  // Simple Histograms
  hGenPhtnHardEt  = new TH1F("genPhtnEt" , "Highest Et Gen Photon: Et " , HI_NUM_BINS,    MIN_E, MAX_E);
  hGenPhtnHardEta = new TH1F("genPhtnEta", "Highest Et Gen Photon: #eta", ME_NUM_BINS, -MAX_ETA, MAX_ETA);
  hGenPhtnHardPhi = new TH1F("genPhtnPhi", "Highest Et Gen Photon: #phi", ME_NUM_BINS, -MAX_PHI, MAX_PHI);
  hGenPhtnHardMom = new TH1F("genPhtnMom", "Highest Et Gen Photon: Mother ID", 40, -0.5, 39.5);
  hGenPhtnHardDrJet = new TH1F("genPhtnDrJet", "Highest Et Gen Photon: #DeltaR to Gen Jet", ME_NUM_BINS, 0.0, 5.0);

  hHIGenJetPt  = new TH1F("genJetPt" , "Highest Pt Gen Jet: Pt "  , HI_NUM_BINS,      MIN_E,    MAX_E);
  hHIGenJetEta = new TH1F("genJetEta", "Highest Pt Gen Jet: #eta ", ME_NUM_BINS, -2*MAX_ETA,  2*MAX_ETA);
  hHIGenJetPhi = new TH1F("genJetPhi", "Highest Pt Gen Jet: #phi ", ME_NUM_BINS,   -MAX_PHI,  MAX_PHI);
  hHIGenJetCnt = new TH1F("genJetCnt", "Number of Gen Jets (pt > 20 GeV)", 10, -0.5, 9.5);

}

void 
PhotonJetAnalyzer::endJob() {

  TFile *outputRootFile = new TFile("photonJetoutput.root","RECREATE");
  outputRootFile->cd();

  hGenPhtnHardEt->Write();
  hGenPhtnHardEta->Write();
  hGenPhtnHardPhi->Write();
  hGenPhtnHardMom->Write();
  hGenPhtnHardDrJet->Write();

  hHIGenJetPt->Write();
  hHIGenJetEta->Write();
  hHIGenJetPhi->Write();
  hHIGenJetCnt->Write();

  // Write histograms and ntuples
  outputRootFile->Write();
  outputRootFile->Close();

}

// **********************
// Method for Calculating the delta-r between two things
double PhotonJetAnalyzer::calcDeltaR(double eta1, double phi1, double eta2, double phi2) {

  double deltaEta = eta1 - eta2;
  double deltaPhi = calcDeltaPhi(phi1, phi2);

  double deltaR = deltaEta*deltaEta + deltaPhi*deltaPhi;
  deltaR = sqrt(deltaR);

  return deltaR;
} // End of calcDeltaR
// **********************

// **********************
// Method for Calculating the delta-phi
double PhotonJetAnalyzer::calcDeltaPhi(double phi1, double phi2) {

  double deltaPhi = phi1 - phi2;

  if (deltaPhi < 0) deltaPhi = -deltaPhi;

  if (deltaPhi > 3.1415926) {
    deltaPhi = 2 * 3.1415926 - deltaPhi;
  }

  return deltaPhi;
} // End of calcDeltaPhi
// **********************

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(PhotonJetAnalyzer);
