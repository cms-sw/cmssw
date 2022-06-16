// -*- C++ -*-
//
// Package:
// Class:   PFMETBenchmarkAnalyzer.cc
//
/**\class PFMETBenchmarkAnalyzer PFMETBenchmarkAnalyzer.cc

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
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/HepMCCandidate/interface/GenParticleFwd.h"
#include "DataFormats/METReco/interface/CaloMET.h"
#include "DataFormats/METReco/interface/CaloMETCollection.h"
#include "DataFormats/METReco/interface/GenMET.h"
#include "DataFormats/METReco/interface/MET.h"
#include "DataFormats/METReco/interface/PFMET.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "RecoParticleFlow/Benchmark/interface/PFMETBenchmark.h"
using namespace edm;
using namespace reco;
using namespace std;

//
// class decleration

class PFMETBenchmarkAnalyzer : public edm::one::EDAnalyzer<> {
public:
  typedef dqm::legacy::DQMStore DQMStore;
  typedef dqm::legacy::MonitorElement MonitorElement;

  explicit PFMETBenchmarkAnalyzer(const edm::ParameterSet &);
  ~PFMETBenchmarkAnalyzer() override;

private:
  void beginJob() override;
  void analyze(const edm::Event &, const edm::EventSetup &) override;
  void endJob() override;
  // ----------member data ---------------------------
  edm::EDGetTokenT<reco::GenParticleCollection> sInputTruthLabel_tok_;
  edm::EDGetTokenT<reco::PFMETCollection> sInputRecoLabel_tok_;
  edm::EDGetTokenT<reco::CaloMETCollection> sInputCaloLabel_tok_;
  edm::EDGetTokenT<reco::METCollection> sInputTCLabel_tok_;

  // neuhaus - comment
  PFMETBenchmark PFMETBenchmark_;
  string OutputFileName;
  bool pfmBenchmarkDebug;
  bool xplotAgainstReco;
  string xbenchmarkLabel_;
  dqm::legacy::DQMStore *xdbe_;
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
PFMETBenchmarkAnalyzer::PFMETBenchmarkAnalyzer(const edm::ParameterSet &iConfig)

{
  // now do what ever initialization is needed
  sInputTruthLabel_tok_ = consumes<reco::GenParticleCollection>(iConfig.getParameter<InputTag>("InputTruthLabel"));
  sInputRecoLabel_tok_ = consumes<reco::PFMETCollection>(iConfig.getParameter<InputTag>("InputRecoLabel"));
  sInputCaloLabel_tok_ = consumes<reco::CaloMETCollection>(iConfig.getParameter<InputTag>("InputCaloLabel"));
  sInputTCLabel_tok_ = consumes<reco::METCollection>(iConfig.getParameter<InputTag>("InputTCLabel"));
  OutputFileName = iConfig.getUntrackedParameter<string>("OutputFile");
  pfmBenchmarkDebug = iConfig.getParameter<bool>("pfjBenchmarkDebug");
  xplotAgainstReco = iConfig.getParameter<bool>("PlotAgainstRecoQuantities");
  xbenchmarkLabel_ = iConfig.getParameter<string>("BenchmarkLabel");
  xdbe_ = edm::Service<DQMStore>().operator->();

  PFMETBenchmark_.setup(OutputFileName, pfmBenchmarkDebug, xplotAgainstReco, xbenchmarkLabel_, xdbe_);
}

PFMETBenchmarkAnalyzer::~PFMETBenchmarkAnalyzer() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
}

//
// member functions
//

// ------------ method called to for each event  ------------
void PFMETBenchmarkAnalyzer::analyze(const edm::Event &iEvent, const edm::EventSetup &iSetup) {
  // get gen jet collection
  Handle<GenParticleCollection> genparticles;
  bool isGen = iEvent.getByToken(sInputTruthLabel_tok_, genparticles);
  if (!isGen) {
    std::cout << "Warning : no Gen Particles in input !" << std::endl;
    return;
  }

  // get rec PFMet collection
  Handle<PFMETCollection> pfmets;
  bool isReco = iEvent.getByToken(sInputRecoLabel_tok_, pfmets);
  if (!isReco) {
    std::cout << "Warning : no PF MET in input !" << std::endl;
    return;
  }

  // get rec TCMet collection
  Handle<METCollection> tcmets;
  bool isTC = iEvent.getByToken(sInputTCLabel_tok_, tcmets);
  if (!isTC) {
    std::cout << "Warning : no TC MET in input !" << std::endl;
    return;
  }

  Handle<CaloMETCollection> calomets;
  bool isCalo = iEvent.getByToken(sInputCaloLabel_tok_, calomets);
  if (!isCalo) {
    std::cout << "Warning : no Calo MET in input !" << std::endl;
    return;
  }

  // Analyse (no "z" in "analyse" : we are in Europe, dammit!)
  PFMETBenchmark_.process(*pfmets, *genparticles, *calomets, *tcmets);
}

// ------------ method called once each job just before starting event loop
// ------------
void PFMETBenchmarkAnalyzer::beginJob() {}

// ------------ method called once each job just after ending the event loop
// ------------
void PFMETBenchmarkAnalyzer::endJob() {
  //  PFMETBenchmark_.save();
  PFMETBenchmark_.analyse();
  PFMETBenchmark_.write();
}

// define this as a plug-in
DEFINE_FWK_MODULE(PFMETBenchmarkAnalyzer);
