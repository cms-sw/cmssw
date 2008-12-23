#ifndef H4MU_ANALYZER
#define H4MU_ANALYZER

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "HepMC/WeightContainer.h"
#include "HepMC/GenEvent.h"
#include "HepMC/GenParticle.h"

#include "SimDataFormats/HepMCProduct/interface/HepMCProduct.h"

#include "TH1D.h"
#include "TFile.h"

//
// class decleration
//

class NewAnalyzer : public edm::EDAnalyzer {
   public:
      explicit NewAnalyzer(const edm::ParameterSet&);
      ~NewAnalyzer();


   private:
      virtual void beginJob(const edm::EventSetup&) ;
      virtual void analyze(const edm::Event&, const edm::EventSetup&);
      virtual void endJob() ;

      // ----------member data ---------------------------
      
  std::string outputFilename;
  TH1D* weight_histo; int event;
  TH1F* Z_invmass_histo; TH1F* Z0J_invmass_histo; TH1F* Z1J_invmass_histo; TH1F*  Z2J_invmass_histo; TH1F* Z3J_invmass_histo; TH1F*  Z4J_invmass_histo; 
  TH1F* Z1JJ1Eta_histo; TH1F* Z2JJ1Eta_histo;  TH1F* Z3JJ1Eta_histo;  TH1F* Z4JJ1Eta_histo; TH1F* ZEta_histo;
  TH1F* Z2JJ2Eta_histo; TH1F*  Z3JJ2Eta_histo; TH1F*  Z4JJ2Eta_histo; TH1F* Jetmult_histo;
  TH1F* Pt_histo; TH1F* J1Pt_histo; TH1F* J2Pt_histo; TH1F* EJDelR_histo; 
  TH1F* E1Pt_histo; TH1F* E2Pt_histo; TH1F* ZPz_histo; TH1F* ZPt_histo; TH1F* Z2JJDelPhi_histo; TH1F* Z3JJDelPhi_histo; TH1F* Z4JJDelPhi_histo;
  TH1F* JDelR_histo; TH1F* JDelPhi_histo; TH1F* Z1JJ1Phi_histo; TH1F*  Z2JJ1Phi_histo; TH1F*  Z3JJ1Phi_histo; TH1F*  Z4JJ1Phi_histo; 
  TH1F* Z2JJ2Phi_histo; TH1F*  Z3JJ2Phi_histo; TH1F* Z4JJ2Phi_histo; TH1F* Z2JJDelR_histo; TH1F* Z3JJDelR_histo; TH1F* Z4JJDelR_histo; TH1F* ZRap_histo;
  TH1F* JetPt1J; TH1F* JetPt2J; TH1F* JetPt3J; TH1F* JetPt4J; 
  TH1F* ZPt1J_histo; TH1F* ZPt2J_histo; TH1F* ZPt3J_histo; TH1F* ZPt4J_histo; TH1F* ZPt0J_histo; 
  TH1F* Z1JJ1Pt_histo; TH1F* Z2JJ1Pt_histo; TH1F* Z2JJ2Pt_histo; TH1F* Z3JJ1Pt_histo; TH1F* Z3JJ2Pt_histo;
  TH1F* Z4JJ1Pt_histo; TH1F* Z4JJ2Pt_histo; TH1F* J1Eta_histo; TH1F* J1Phi_histo; 
};

#endif
