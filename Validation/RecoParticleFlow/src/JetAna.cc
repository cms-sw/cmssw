// -*- C++ -*-
//
// Package:    JetAna
// Class:      JetAna
// 
/**\class JetAna JetAna.cc Work/JetAna/src/JetAna.cc

 Description: <one line class summary>

 Implementation:

module pfJetAnalyzer = JetAna 
 { 
   string InputTruthLabel = 'iterativeCone5GenJets'
   string InputRecoLabel  = 'iterativeCone5PFJets'
   untracked string OutputFile = 'PFJetTester_data.root'
   bool PlotAgainstRecoQuantities = true
   string BenchmarkLabel = 'ParticleFlow'
   double recPt = 22 # -1 means no cut
   double maxEta = 3 # -1 means no cut

   bool pfjBenchmarkDebug = 0
   double deltaRMax = 0.1
 }

*/
//
// Original Author:  Michel Della Negra
//         Created:  Wed Jan 23 10:11:13 CET 2008
// $Id$
//
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


using namespace edm;
using namespace reco;
using namespace std;

//
// class decleration

 
class JetAna : public edm::EDAnalyzer {
public:
  explicit JetAna(const edm::ParameterSet&);
  ~JetAna();


private:
  virtual void beginJob(const edm::EventSetup&) ;
  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void endJob() ;
  // ----------member data ---------------------------

};
/// PFJet Benchmark

//neuhaus - comment
PFJetBenchmark PFJetBenchmark_;
string sGenJetAlgo;
string sJetAlgo;
string outjetfilename;
bool pfjBenchmarkDebug;
bool PlotAgainstReco;
double deltaRMax=0.1;
string benchmarkLabel_;
double recPt;
double maxEta;
//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
JetAna::JetAna(const edm::ParameterSet& iConfig)

{
  //now do what ever initialization is needed
  sGenJetAlgo = 
    iConfig.getParameter<string>("InputTruthLabel");
  sJetAlgo = 
    iConfig.getParameter<string>("InputRecoLabel");
  outjetfilename = 
    iConfig.getUntrackedParameter<string>("OutputFile");
  pfjBenchmarkDebug = 
    iConfig.getParameter<bool>("pfjBenchmarkDebug");
  PlotAgainstReco = 
    iConfig.getParameter<bool>("PlotAgainstRecoQuantities");
  deltaRMax = 
    iConfig.getParameter<double>("deltaRMax");	  
  benchmarkLabel_  = 
    iConfig.getParameter<string>("BenchmarkLabel"); 
  recPt  = 
    iConfig.getParameter<double>("recPt"); 
  maxEta = 
    iConfig.getParameter<double>("maxEta"); 
  
  PFJetBenchmark_.setup(
			outjetfilename, 
			pfjBenchmarkDebug,
			PlotAgainstReco,
			deltaRMax,
			benchmarkLabel_, 
			recPt, 
			maxEta);
}


JetAna::~JetAna()
{
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
}


//
// member functions
//

// ------------ method called to for each event  ------------
void
JetAna::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
 // get gen jet collection
  Handle<GenJetCollection> genjets;
  iEvent.getByLabel(sGenJetAlgo.data(), genjets);

  // get rec PFJet collection
  Handle<PFJetCollection> pfjets;
  iEvent.getByLabel(sJetAlgo.data(), pfjets);   
  PFJetBenchmark_.process(*pfjets, *genjets);
}


// ------------ method called once each job just before starting event loop  ------------
void 
JetAna::beginJob(const edm::EventSetup&)
{

}

// ------------ method called once each job just after ending the event loop  ------------
void 
JetAna::endJob() {
  PFJetBenchmark_.save();
}

//define this as a plug-in
DEFINE_FWK_MODULE(JetAna);
