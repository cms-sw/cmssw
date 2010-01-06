#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "HepMC/GenEvent.h"
#include "HepMC/GenParticle.h"
#include "HepMC/WeightContainer.h"

#include "TH1D.h"
#include "TFile.h"


class TauAnalyzer : public edm::EDAnalyzer {
 public:
  explicit TauAnalyzer(const edm::ParameterSet&);
  ~TauAnalyzer();
  
 private:
  virtual void beginJob() ;
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;
  
 private:
  std::string outputFilename;
  TH1D* weight_histo;
  TH1D* invmass_histo;
  TH1D* pT1_histo;
  TH1D* pT2_histo;
  TH1D* h_mcRtau;
  TH1D* h_mcRleptonic;
  
  int eventCounter;
  int mcHadronicTauCounter;
  int mcVisibleTauCounter;
  int mcTauPtCutCounter;
};
