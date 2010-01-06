#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "HepMC/GenEvent.h"
#include "HepMC/GenParticle.h"
#include "HepMC/WeightContainer.h"

#include "TH1D.h"
#include "TFile.h"


class H4muAnalyzer : public edm::EDAnalyzer {
 public:
  explicit H4muAnalyzer(const edm::ParameterSet&);
  ~H4muAnalyzer();
  
 private:
  virtual void beginJob() ;
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;
  
 private:
  std::string outputFilename;
  TH1D* weight_histo;
  TH1D* invmass_histo;
  TH1D* Z_invmass_histo;
  
};
