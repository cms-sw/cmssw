// -*- C++ -*-
//
// Package:
// Class:   PFJetBenchmarkAnalyzer.cc
//
/**\class PFJetBenchmarkAnalyzer PFJetBenchmarkAnalyzer.cc

 Description: <one line class summary>

 Implementation:


*/
//
// Original Author:  Michel Della Negra
//         Created:  Wed Jan 23 10:11:13 CET 2008
// Extensions by Joanna Weng
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/JetReco/interface/BasicJetCollection.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/JetReco/interface/GenJet.h"
#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/JetReco/interface/PFJet.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "RecoParticleFlow/Benchmark/interface/PFJetBenchmark.h"
using namespace edm;
using namespace reco;
using namespace std;

//
// class decleration

class PFJetBenchmarkAnalyzer : public edm::one::EDAnalyzer<> {
public:
  typedef dqm::legacy::DQMStore DQMStore;
  typedef dqm::legacy::MonitorElement MonitorElement;

  explicit PFJetBenchmarkAnalyzer(const edm::ParameterSet &);
  ~PFJetBenchmarkAnalyzer() override;

private:
  void beginJob() override;
  void analyze(const edm::Event &, const edm::EventSetup &) override;
  void endJob() override;
  // ----------member data ---------------------------
  edm::EDGetTokenT<reco::GenJetCollection> sGenJetAlgo_tok_;
  edm::EDGetTokenT<reco::PFJetCollection> sJetAlgo_tok_;

  // neuhaus - comment
  PFJetBenchmark PFJetBenchmark_;
  string outjetfilename;
  bool pfjBenchmarkDebug;
  bool plotAgainstReco;
  bool onlyTwoJets;
  double deltaRMax = 0.1;
  string benchmarkLabel_;
  double recPt;
  double maxEta;
  dqm::legacy::DQMStore *dbe_;
};
/// PFJet Benchmark

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
PFJetBenchmarkAnalyzer::PFJetBenchmarkAnalyzer(const edm::ParameterSet &iConfig)

{
  // now do what ever initialization is needed
  sGenJetAlgo_tok_ = consumes<reco::GenJetCollection>(iConfig.getParameter<edm::InputTag>("InputTruthLabel"));
  sJetAlgo_tok_ = consumes<reco::PFJetCollection>(iConfig.getParameter<edm::InputTag>("InputRecoLabel"));
  outjetfilename = iConfig.getUntrackedParameter<string>("OutputFile");
  pfjBenchmarkDebug = iConfig.getParameter<bool>("pfjBenchmarkDebug");
  plotAgainstReco = iConfig.getParameter<bool>("PlotAgainstRecoQuantities");
  onlyTwoJets = iConfig.getParameter<bool>("OnlyTwoJets");
  deltaRMax = iConfig.getParameter<double>("deltaRMax");
  benchmarkLabel_ = iConfig.getParameter<string>("BenchmarkLabel");
  recPt = iConfig.getParameter<double>("recPt");
  maxEta = iConfig.getParameter<double>("maxEta");

  dbe_ = edm::Service<DQMStore>().operator->();

  PFJetBenchmark_.setup(
      outjetfilename, pfjBenchmarkDebug, plotAgainstReco, onlyTwoJets, deltaRMax, benchmarkLabel_, recPt, maxEta, dbe_);
}

PFJetBenchmarkAnalyzer::~PFJetBenchmarkAnalyzer() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
}

//
// member functions
//

// ------------ method called to for each event  ------------
void PFJetBenchmarkAnalyzer::analyze(const edm::Event &iEvent, const edm::EventSetup &iSetup) {
  // get gen jet collection
  Handle<GenJetCollection> genjets;
  bool isGen = iEvent.getByToken(sGenJetAlgo_tok_, genjets);
  if (!isGen) {
    std::cout << "Warning : no Gen jets in input !" << std::endl;
    return;
  }

  // get rec PFJet collection
  Handle<PFJetCollection> pfjets;
  bool isReco = iEvent.getByToken(sJetAlgo_tok_, pfjets);
  if (!isReco) {
    std::cout << "Warning : no PF jets in input !" << std::endl;
    return;
  }
  // Analyse (no "z" in "analyse" : we are in Europe, dammit!)
  PFJetBenchmark_.process(*pfjets, *genjets);
}

// ------------ method called once each job just before starting event loop
// ------------
void PFJetBenchmarkAnalyzer::beginJob() {}

// ------------ method called once each job just after ending the event loop
// ------------
void PFJetBenchmarkAnalyzer::endJob() {
  //  PFJetBenchmark_.save();
  PFJetBenchmark_.write();
}

// define this as a plug-in
DEFINE_FWK_MODULE(PFJetBenchmarkAnalyzer);
