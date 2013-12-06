#include "Validation/RecoParticleFlow/plugins/GenericBenchmarkAnalyzer.h"
// author: Mike Schmitt, University of Florida
// first version 11/7/2007
// extension: Leo Neuhaus & Joanna Weng 09.2008
// Performs matching and basic resolution plots of 2 candidate
// (or candidate based) collections

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"


#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/Candidate/interface/Candidate.h"

#include "DQMServices/Core/interface/DQMStore.h"

#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"

#include <vector>
#include <ostream>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <memory>

using namespace reco;
using namespace edm;
using namespace std;

GenericBenchmarkAnalyzer::GenericBenchmarkAnalyzer(const edm::ParameterSet& iConfig)
{

  inputTruthLabel_             = iConfig.getParameter<edm::InputTag>("InputTruthLabel");
  inputRecoLabel_              = iConfig.getParameter<edm::InputTag>("InputRecoLabel");
  outputFile_                  = iConfig.getUntrackedParameter<std::string>("OutputFile");
  benchmarkLabel_              = iConfig.getParameter<std::string>("BenchmarkLabel"); 
  startFromGen_                = iConfig.getParameter<bool>("StartFromGen");
  plotAgainstRecoQuantities_   = iConfig.getParameter<bool>("PlotAgainstRecoQuantities");
  onlyTwoJets_                 = iConfig.getParameter<bool>("OnlyTwoJets");
  recPt_cut                    = iConfig.getParameter<double>("recPt");
  minEta_cut                   = iConfig.getParameter<double>("minEta");
  maxEta_cut                   = iConfig.getParameter<double>("maxEta");
  deltaR_cut                   = iConfig.getParameter<double>("deltaRMax");

  minDeltaEt_                   = iConfig.getParameter<double>("minDeltaEt");
  maxDeltaEt_                   = iConfig.getParameter<double>("maxDeltaEt");
  minDeltaPhi_                  = iConfig.getParameter<double>("minDeltaPhi");
  maxDeltaPhi_                  = iConfig.getParameter<double>("maxDeltaPhi");
  doMetPlots_                   = iConfig.getParameter<bool>("doMetPlots");

  if (outputFile_.size() > 0)
    edm::LogInfo("OutputInfo") << " ParticleFLow Task histograms will be saved to '" << outputFile_.c_str()<< "'";
  else edm::LogInfo("OutputInfo") << " ParticleFlow Task histograms will NOT be saved";

  myTruth_ = consumes< edm::View<reco::Candidate> >(inputTruthLabel_);
  myReco_ = consumes< edm::View<reco::Candidate> >(inputRecoLabel_);

}

GenericBenchmarkAnalyzer::~GenericBenchmarkAnalyzer() { }

void 
GenericBenchmarkAnalyzer::beginJob()
{

  // get ahold of back-end interface
  dbe_ = edm::Service<DQMStore>().operator->();
  
  if (dbe_) {
    //dbe_->setVerbose(1);
    //string path = "PFTask/Benchmarks/" + benchmarkLabel_ + "/";
    std::string path = "ParticleFlow/" + benchmarkLabel_ + "/" ;
    if (plotAgainstRecoQuantities_) path += "Reco"; else path += "Gen";
    dbe_->setCurrentFolder(path.c_str());
    setup(dbe_, plotAgainstRecoQuantities_, minDeltaEt_, maxDeltaEt_, minDeltaPhi_, maxDeltaPhi_, doMetPlots_);

  }

}

void 
GenericBenchmarkAnalyzer::analyze(const edm::Event& iEvent, 
				  const edm::EventSetup& iSetup)
{
  
  // Typedefs to use views
  typedef edm::View<reco::Candidate> candidateCollection ;
  typedef edm::View<reco::Candidate> candidateCollection ;
  
  const candidateCollection *truth_candidates;
  const candidateCollection *reco_candidates;
 
  // ==========================================================
  // Retrieve!
  // ==========================================================

  { 
    // Get Truth Candidates (GenCandidates, GenJets, etc.)
    Handle<candidateCollection> truth_hnd;
    bool isGen = iEvent.getByToken(myTruth_, truth_hnd);   

    if ( !isGen ) { 
      std::cout << "Warning : no Gen jets in input !" << std::endl;
      return;
    }

    truth_candidates = truth_hnd.product();

    // Get Reco Candidates (PFlow, CaloJet, etc.)
    Handle<candidateCollection> reco_hnd;
    bool isReco = iEvent.getByToken(myReco_, reco_hnd);
    if ( !isReco ) { 
      std::cout << "Warning : no Reco jets in input !" << std::endl;
      return; 
    }
    reco_candidates = reco_hnd.product();

    // no longer needed with template-ized Benchmark
    //const PFCandidateCollection *pf_candidates = reco_hnd.product();
    //static CandidateCollection reco_storage = algo_->makeCandidateCollection(pf_candidates);
    //reco_candidates = &reco_storage;

  }
  if (!truth_candidates || !reco_candidates) {

    edm::LogInfo("OutputInfo") << " failed to retrieve data required by ParticleFlow Task";
    edm::LogInfo("OutputInfo") << " ParticleFlow Task cannot continue...!";
    return;

  }

  // ==========================================================
  // Analyze!
  // ==========================================================

  fill(reco_candidates,truth_candidates,
      startFromGen_, plotAgainstRecoQuantities_, 
      onlyTwoJets_, recPt_cut,  minEta_cut, maxEta_cut, deltaR_cut);
}

void GenericBenchmarkAnalyzer::endJob() 
{

  // Store the DAQ Histograms
  if (outputFile_.size() != 0)
    dbe_->save(outputFile_);
}
