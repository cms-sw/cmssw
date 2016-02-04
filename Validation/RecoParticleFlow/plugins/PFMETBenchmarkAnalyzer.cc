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
// $Id: PFMETBenchmarkAnalyzer.cc,v 1.4 2010/02/20 21:02:45 wmtan Exp $
// Extensions by Joanna Weng
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/METReco/interface/MET.h"
#include "DataFormats/METReco/interface/CaloMETCollection.h"
#include "DataFormats/METReco/interface/CaloMET.h"
#include "DataFormats/METReco/interface/PFMET.h"
#include "DataFormats/METReco/interface/GenMET.h"
#include "DataFormats/HepMCCandidate/interface/GenParticleFwd.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "RecoParticleFlow/Benchmark/interface/PFMETBenchmark.h"
#include "FWCore/ServiceRegistry/interface/Service.h" 
#include "FWCore/Utilities/interface/InputTag.h"
using namespace edm;
using namespace reco;
using namespace std;

//
// class decleration

 
class PFMETBenchmarkAnalyzer : public edm::EDAnalyzer {
public:
  explicit PFMETBenchmarkAnalyzer(const edm::ParameterSet&);
  ~PFMETBenchmarkAnalyzer();


private:
  virtual void beginJob() ;
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;
  // ----------member data ---------------------------

};
/// PFJet Benchmark

//neuhaus - comment
PFMETBenchmark PFMETBenchmark_;
InputTag sInputTruthLabel;
InputTag sInputRecoLabel;
InputTag sInputCaloLabel;
InputTag sInputTCLabel;
string OutputFileName;
bool pfmBenchmarkDebug;
bool xplotAgainstReco;
string xbenchmarkLabel_;
DQMStore * xdbe_;
//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
PFMETBenchmarkAnalyzer::PFMETBenchmarkAnalyzer(const edm::ParameterSet& iConfig)

{
  //now do what ever initialization is needed
  sInputTruthLabel = 
    iConfig.getParameter<InputTag>("InputTruthLabel");
  sInputRecoLabel = 
    iConfig.getParameter<InputTag>("InputRecoLabel");
  sInputCaloLabel = 
    iConfig.getParameter<InputTag>("InputCaloLabel");
  sInputTCLabel = 
    iConfig.getParameter<InputTag>("InputTCLabel");
  OutputFileName = 
    iConfig.getUntrackedParameter<string>("OutputFile");
  pfmBenchmarkDebug = 
    iConfig.getParameter<bool>("pfjBenchmarkDebug");
  xplotAgainstReco = 
    iConfig.getParameter<bool>("PlotAgainstRecoQuantities");
  xbenchmarkLabel_  = 
    iConfig.getParameter<string>("BenchmarkLabel"); 
  xdbe_ = edm::Service<DQMStore>().operator->();

  PFMETBenchmark_.setup(
			OutputFileName, 
			pfmBenchmarkDebug,
			xplotAgainstReco,
			xbenchmarkLabel_, 
			xdbe_);
}


PFMETBenchmarkAnalyzer::~PFMETBenchmarkAnalyzer()
{
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
}


//
// member functions
//

// ------------ method called to for each event  ------------
void
PFMETBenchmarkAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
 // get gen jet collection
  Handle<GenParticleCollection> genparticles;
  bool isGen = iEvent.getByLabel(sInputTruthLabel, genparticles);
  if (!isGen) { 
    std::cout << "Warning : no Gen Particles in input !" << std::endl;
    return;
  }

  // get rec PFMet collection
  Handle<PFMETCollection> pfmets;
  bool isReco = iEvent.getByLabel(sInputRecoLabel, pfmets);   
  if (!isReco) { 
    std::cout << "Warning : no PF MET in input !" << std::endl;
    return;
  }

  // get rec TCMet collection
  Handle<METCollection> tcmets;
  bool isTC = iEvent.getByLabel(sInputTCLabel, tcmets);   
  if (!isTC) { 
    std::cout << "Warning : no TC MET in input !" << std::endl;
    return;
  }

  Handle<CaloMETCollection> calomets;
  bool isCalo = iEvent.getByLabel(sInputCaloLabel, calomets);   
  if (!isCalo) { 
    std::cout << "Warning : no Calo MET in input !" << std::endl;
    return;
  }

  // Analyse (no "z" in "analyse" : we are in Europe, dammit!) 
  PFMETBenchmark_.process(*pfmets, *genparticles, *calomets, *tcmets);
}


// ------------ method called once each job just before starting event loop  ------------
void 
PFMETBenchmarkAnalyzer::beginJob()
{

}

// ------------ method called once each job just after ending the event loop  ------------
void 
PFMETBenchmarkAnalyzer::endJob() {
//  PFMETBenchmark_.save();
  PFMETBenchmark_.analyse();
  PFMETBenchmark_.write();
}

//define this as a plug-in
DEFINE_FWK_MODULE(PFMETBenchmarkAnalyzer);
