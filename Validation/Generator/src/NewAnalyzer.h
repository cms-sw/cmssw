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
  TH1F* EE_invmass_histo; TH1F* J1Eta_histo; TH1F* J2Eta_histo;
  TH1F* Pt_histo; TH1F* J1Pt_histo; TH1F* J2Pt_histo; TH1F* EJDelR_histo; 
  TH1F* E1Pt_histo; TH1F* E2Pt_histo; TH1F* ZPz_histo; TH1F* ZPt_histo;
  TH1F* JDelR_histo; TH1F* JDelPhi_histo; TH1F* J1Phi_histo; TH1F* J2Phi_histo;
};

#endif
