#include <memory>
#include <vector>
#include <fstream>
#include <string>
#include <Math/VectorUtil.h>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "Utilities/General/interface/envUtil.h"

#include "DataFormats/PatCandidates/interface/Electron.h"
#include "DataFormats/PatCandidates/interface/Muon.h"
#include "DataFormats/PatCandidates/interface/Tau.h"
#include "DataFormats/PatCandidates/interface/Jet.h"
#include "DataFormats/PatCandidates/interface/MET.h"

#include "AnalysisDataFormats/TopObjects/interface/TtGenEvent.h"

#include "TH1.h"
#include "TF1.h"
#include "TFile.h"
#include "TTree.h"

namespace res{
  typedef reco::PixelMatchGsfElectron electronType;
  typedef reco::Muon muonType;
  typedef reco::CaloJet jetType;
  typedef reco::CaloMET metType;
}

class ResolutionCreator : public edm::EDAnalyzer {

 public:

  explicit ResolutionCreator(const edm::ParameterSet&);
  ~ResolutionCreator();
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  
 private:

  TFile *outfile;
  TF1 *fResEtEtaBin[10][20][20];
  TF1 *fResEtaBin[10][20];
  TH1F *hResEtEtaBin[10][20][20];
  TH1F *hResEtaBin[10][20];
  TH1F *hEtaBins;

  std::string objectType_, labelName_;
  std::vector<double> etabinVals_, eTbinVals_;
  double minDR_;
  int etnrbins, etanrbins;
  int nrFilled;
};
