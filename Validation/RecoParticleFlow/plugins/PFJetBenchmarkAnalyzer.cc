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
// $Id: PFJetBenchmarkAnalyzer.cc,v 1.3 2010/02/20 21:02:43 wmtan Exp $
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
#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/JetReco/interface/CaloJetCollection.h"
#include "DataFormats/JetReco/interface/BasicJetCollection.h"
#include "DataFormats/JetReco/interface/PFJet.h"
#include "DataFormats/JetReco/interface/GenJet.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "RecoParticleFlow/Benchmark/interface/PFJetBenchmark.h"
#include "FWCore/ServiceRegistry/interface/Service.h" 
#include "FWCore/Utilities/interface/InputTag.h"
using namespace edm;
using namespace reco;
using namespace std;

//
// class decleration

 
class PFJetBenchmarkAnalyzer : public edm::EDAnalyzer {
public:
  explicit PFJetBenchmarkAnalyzer(const edm::ParameterSet&);
  ~PFJetBenchmarkAnalyzer();


private:
  virtual void beginJob() ;
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;
  // ----------member data ---------------------------

};
/// PFJet Benchmark

//neuhaus - comment
PFJetBenchmark PFJetBenchmark_;
InputTag sGenJetAlgo;
InputTag sJetAlgo;
string outjetfilename;
bool pfjBenchmarkDebug;
bool plotAgainstReco;
bool onlyTwoJets;
double deltaRMax=0.1;
string benchmarkLabel_;
double recPt;
double maxEta;
DQMStore * dbe_;
//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
PFJetBenchmarkAnalyzer::PFJetBenchmarkAnalyzer(const edm::ParameterSet& iConfig)

{
  //now do what ever initialization is needed
  sGenJetAlgo = 
    iConfig.getParameter<InputTag>("InputTruthLabel");
  sJetAlgo = 
    iConfig.getParameter<InputTag>("InputRecoLabel");
  outjetfilename = 
    iConfig.getUntrackedParameter<string>("OutputFile");
  pfjBenchmarkDebug = 
    iConfig.getParameter<bool>("pfjBenchmarkDebug");
  plotAgainstReco = 
    iConfig.getParameter<bool>("PlotAgainstRecoQuantities");
  onlyTwoJets = 
    iConfig.getParameter<bool>("OnlyTwoJets");
  deltaRMax = 
    iConfig.getParameter<double>("deltaRMax");	  
  benchmarkLabel_  = 
    iConfig.getParameter<string>("BenchmarkLabel"); 
  recPt  = 
    iConfig.getParameter<double>("recPt"); 
  maxEta = 
    iConfig.getParameter<double>("maxEta"); 
  
  dbe_ = edm::Service<DQMStore>().operator->();

  PFJetBenchmark_.setup(
			outjetfilename, 
			pfjBenchmarkDebug,
			plotAgainstReco,
			onlyTwoJets,
			deltaRMax,
			benchmarkLabel_, 
			recPt, 
			maxEta, 
			dbe_);
}


PFJetBenchmarkAnalyzer::~PFJetBenchmarkAnalyzer()
{
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
}


//
// member functions
//

// ------------ method called to for each event  ------------
void
PFJetBenchmarkAnalyzer::analyze(const edm::Event& iEvent, 
				const edm::EventSetup& iSetup)
{
 // get gen jet collection
  Handle<GenJetCollection> genjets;
  bool isGen = iEvent.getByLabel(sGenJetAlgo, genjets);
  if (!isGen) { 
    std::cout << "Warning : no Gen jets in input !" << std::endl;
    return;
  }

  // get rec PFJet collection
  Handle<PFJetCollection> pfjets;
  bool isReco = iEvent.getByLabel(sJetAlgo, pfjets);   
  if (!isReco) { 
    std::cout << "Warning : no PF jets in input !" << std::endl;
    return;
  }
  // Analyse (no "z" in "analyse" : we are in Europe, dammit!) 
  PFJetBenchmark_.process(*pfjets, *genjets);
}


// ------------ method called once each job just before starting event loop  ------------
void 
PFJetBenchmarkAnalyzer::beginJob()
{

}

// ------------ method called once each job just after ending the event loop  ------------
void 
PFJetBenchmarkAnalyzer::endJob() {
//  PFJetBenchmark_.save();
  PFJetBenchmark_.write();
}

//define this as a plug-in
DEFINE_FWK_MODULE(PFJetBenchmarkAnalyzer);
