#include "Validation/RecoParticleFlow/interface/PFBenchmarkAnalyzer.h"
// author: Mike Schmitt, University of Florida
// first version 11/7/2007

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "DataFormats/HepMCCandidate/interface/GenParticleCandidateFwd.h"
#include "DataFormats/HepMCCandidate/interface/GenParticleCandidate.h"
#include "DataFormats/METReco/interface/GenMET.h"
#include "DataFormats/METReco/interface/GenMETCollection.h"
#include "DataFormats/METReco/interface/CaloMET.h"
#include "DataFormats/METReco/interface/CaloMETCollection.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlock.h"
#include "DataFormats/ParticleFlowReco/interface/PFBlockElement.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/Common/interface/RefToBase.h"

#include "RecoParticleFlow/Benchmark/interface/PFBenchmarkAlgo.h"

#include <vector>
#include <ostream>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <memory>

using namespace edm;
using namespace std;

PFBenchmarkAnalyzer::PFBenchmarkAnalyzer(const edm::ParameterSet& iConfig)
{

  inputTruthLabel_             = iConfig.getParameter<std::string>("InputTruthLabel");
  inputRecoLabel_              = iConfig.getParameter<std::string>("InputRecoLabel");
  outputFile_                  = iConfig.getUntrackedParameter<std::string>("OutputFile");
  benchmarkLabel_              = iConfig.getParameter<std::string>("BenchmarkLabel"); 
  plotAgainstRecoQuantities_   = iConfig.getParameter<bool>("PlotAgainstRecoQuantities");

  if (outputFile_.size() > 0)
    edm::LogInfo("OutputInfo") << " ParticleFLow Task histograms will be saved to '" << outputFile_.c_str() << "'";
  else edm::LogInfo("OutputInfo") << " ParticleFlow Task histograms will NOT be saved";

  // initialize the PFlow benchmarks algorithm
  algo_ = new PFBenchmarkAlgo();

}

PFBenchmarkAnalyzer::~PFBenchmarkAnalyzer() { delete algo_; }

void PFBenchmarkAnalyzer::beginJob(const edm::EventSetup& iSetup)
{

  // get ahold of back-end interface
  dbe_ = edm::Service<DaqMonitorBEInterface>().operator->();
  
  if (dbe_) {

    string path = "PFTask/Benchmarks/" + benchmarkLabel_ + "/";
    if (plotAgainstRecoQuantities_) path += "Reco"; else path += "Gen";
    dbe_->setCurrentFolder(path.c_str());

    // delta et quantities
    me["DeltaEt"]                = dbe_->book1D("DeltaEt","DeltaEt",1000,-100,100);
    me["DeltaEtvsEt"]            = dbe_->book2D("DeltaEtvsEt","DeltaEtvsEt",1000,0,1000,1000,-100,100);
    me["DeltaEtOverEtvsEt"]      = dbe_->book2D("DeltaEtOverEtvsEt","DeltaEtOverEtvsEt",1000,0,1000,100,-1,1);
    me["DeltaEtvsEta"]           = dbe_->book2D("DeltaEtvsEta","DeltaEtvsEta",200,-5,5,1000,-100,100);
    me["DeltaEtOverEtvsEta"]     = dbe_->book2D("DeltaEtOverEtvsEta","DeltaEtOverEtvsEta",200,-5,5,100,-1,1);
    me["DeltaEtvsPhi"]           = dbe_->book2D("DeltaEtvsPhi","DeltaEtvsPhi",200,-M_PI,M_PI,1000,-100,100);
    me["DeltaEtOverEtvsPhi"]     = dbe_->book2D("DeltaEtOverEtvsPhi","DeltaEtOverEtvsPhi",200,-M_PI,M_PI,100,-1,1);
    me["DeltaEtvsDeltaR"]        = dbe_->book2D("DeltaEtvsDeltaR","DeltaEtvsDeltaR",100,0,1,1000,-100,100);
    me["DeltaEtOverEtvsDeltaR"]  = dbe_->book2D("DeltaEtOverEtvsDeltaR","DeltaEtOverEtvsDeltaR",100,0,1,100,-1,1);

    // delta eta quantities
    me["DeltaEta"]               = dbe_->book1D("DeltaEta","DeltaEta",100,-3,3);
    me["DeltaEtavsEt"]           = dbe_->book2D("DeltaEtavsEt","DeltaEtavsEt",1000,0,1000,100,-3,3);
    me["DeltaEtaOverEtavsEt"]    = dbe_->book2D("DeltaEtaOverEtavsEt","DeltaEtaOverEtavsEt",1000,0,1000,100,-1,1);
    me["DeltaEtavsEta"]          = dbe_->book2D("DeltaEtavsEta","DeltaEtavsEta",200,-5,5,100,-3,3);
    me["DeltaEtaOverEtavsEta"]   = dbe_->book2D("DeltaEtaOverEtavsEta","DeltaEtaOverEtvsEta",200,-5,5,100,-1,1);
    me["DeltaEtavsPhi"]          = dbe_->book2D("DeltaEtavsPhi","DeltaEtavsPhi",200,-M_PI,M_PI,200,-M_PI,M_PI);
    me["DeltaEtaOverEtavsPhi"]   = dbe_->book2D("DeltaEtaOverEtavsPhi","DeltaEtaOverEtavsPhi",200,-M_PI,M_PI,100,-1,1);

    // delta phi quantities
    me["DeltaPhi"]             = dbe_->book1D("DeltaPhi","DeltaPhi",100,-M_PI_2,M_PI_2);
    me["DeltaPhivsEt"]         = dbe_->book2D("DeltaPhivsEt","DeltaPhivsEt",1000,0,1000,100,-M_PI_2,M_PI_2);
    me["DeltaPhiOverPhivsEt"]  = dbe_->book2D("DeltaPhiOverPhivsEt","DeltaPhiOverPhivsEt",1000,0,1000,100,-1,1);
    me["DeltaPhivsEta"]        = dbe_->book2D("DeltaPhivsEta","DeltaPhivsEta",200,-5,5,100,-M_PI_2,M_PI_2);
    me["DeltaPhiOverPhivsEta"] = dbe_->book2D("DeltaPhiOverPhivsEta","DeltaPhiOverPhivsEta",200,-5,5,100,-1,1);
    me["DeltaPhivsPhi"]        = dbe_->book2D("DeltaPhivsPhi","DeltaPhivsPhi",200,-M_PI,M_PI,200,-M_PI,M_PI);
    me["DeltaPhiOverPhivsPhi"] = dbe_->book2D("DeltaPhiOverPhivsPhi","DeltaPhiOverPhivsPhi",200,-M_PI,M_PI,100,-1,1);

    // delta R quantities
    me["DeltaR"]               = dbe_->book1D("DeltaR","DeltaR",100,0,1);
    me["DeltaRvsEt"]           = dbe_->book2D("DeltaRvsEt","DeltaRvsEt",1000,0,1000,100,0,1);
    me["DeltaRvsEta"]          = dbe_->book2D("DeltaRvsEta","DeltaRvsEta",200,-5,5,100,0,1);
    me["DeltaRvsPhi"]          = dbe_->book2D("DeltaRvsPhi","DeltaRvsPhi",200,-M_PI,M_PI,100,0,1);

    // number of truth particles found within given cone radius of reco
    me["NumMatchesVsDeltaR"]  = dbe_->book2D("NumInConeVsConeSize","NumInConeVsConeSize",100,0,1,25,0,25);

  }

}

void PFBenchmarkAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  
  // data to retrieve from the event
  const CandidateCollection *truth_candidates;
  // hack for now, assume reco = PFCandidate
  //const CandidateCollection *reco_candidates;
  const PFCandidateCollection *reco_candidates;

  // ==========================================================
  // Retrieve!
  // ==========================================================

  { 

    // Get Truth Candidates (GenCandidates, GenJets, etc.)
    Handle<CandidateCollection> truth_hnd;
    iEvent.getByLabel(inputTruthLabel_, truth_hnd);
    truth_candidates = truth_hnd.product();

    // Get Reco Candidates (PFlow, CaloJet, etc.)
    Handle<PFCandidateCollection> reco_hnd;
    iEvent.getByLabel(inputRecoLabel_, reco_hnd);
    reco_candidates = reco_hnd.product();

    /* needs debugging...
    Handle<View<Candidate> > truth_hnd;
    iEvent.getByLabel(inputTruthLabel_, truth_hnd);
    static CandidateCollection truth_cc(100);
    truth_cc.clear(); truth_cc.reserve(truth_hnd->size());
    copy(truth_hnd->begin(), truth_hnd->end(), truth_cc.begin());
    truth_candidates = &truth_cc;

    // Get Reco Candidates (PFlow, CaloJet, etc.)
    Handle<View<Candidate> > reco_hnd;
    iEvent.getByLabel(inputTruthLabel_, reco_hnd);
    static CandidateCollection reco_cc(100);
    reco_cc.clear(); reco_cc.reserve(reco_hnd->size());
    copy(reco_hnd->begin(), reco_hnd->end(), reco_cc.begin());
    reco_candidates = &reco_cc;
    */

  }

  if (!truth_candidates || !reco_candidates) {

    edm::LogInfo("OutputInfo") << " failed to retrieve data required by ParticleFlow Task";
    edm::LogInfo("OutputInfo") << " ParticleFlow Task cannot continue...!";
    return;

  }

  // ==========================================================
  // Analyze!
  // ==========================================================

  // loop over reco particles
  PFCandidateCollection::const_iterator reco;
  for (reco = reco_candidates->begin(); reco != reco_candidates->end(); reco++) {

    // generate histograms comparing the reco and truth candidate (truth = closest in delta-R)
    const Candidate *particle = &(*reco);
    const Candidate *gen_particle = algo_->matchByDeltaR(particle,truth_candidates);

    // get the quantities to place on the denominator and/or divide by
    double et, eta, phi;
    if (plotAgainstRecoQuantities_) { et = particle->et(); eta = particle->eta(); phi = particle->phi(); }
    else { et = gen_particle->et(); eta = gen_particle->eta(); phi = gen_particle->phi(); }

    // get the delta quantities
    double deltaEt = algo_->deltaEt(particle,gen_particle);
    double deltaR = algo_->deltaR(particle,gen_particle);
    double deltaEta = algo_->deltaEta(particle,gen_particle);
    double deltaPhi = algo_->deltaPhi(particle,gen_particle);

    // fill histograms
    me["DeltaEt"]->Fill(deltaEt);
    me["DeltaEtvsEt"]->Fill(et,deltaEt);
    me["DeltaEtOverEtvsEt"]->Fill(et,deltaEt/et);
    me["DeltaEtvsEta"]->Fill(eta,deltaEt);
    me["DeltaEtOverEtvsEta"]->Fill(eta,deltaEt/et);
    me["DeltaEtvsPhi"]->Fill(phi,deltaEt);
    me["DeltaEtOverEtvsPhi"]->Fill(phi,deltaEt/et);
    me["DeltaEtvsDeltaR"]->Fill(deltaR,deltaEt);
    me["DeltaEtOverEtvsDeltaR"]->Fill(deltaR,deltaEt/et);
 
    me["DeltaEta"]->Fill(deltaEta);
    me["DeltaEtavsEt"]->Fill(et,deltaEta/eta);
    me["DeltaEtaOverEtavsEt"]->Fill(et,deltaEta/eta);
    me["DeltaEtavsEta"]->Fill(eta,deltaEta);
    me["DeltaEtaOverEtavsEta"]->Fill(eta,deltaEta/eta);
    me["DeltaEtavsPhi"]->Fill(phi,deltaEta);
    me["DeltaEtaOverEtavsPhi"]->Fill(phi,deltaEta/eta);

    me["DeltaPhi"]->Fill(deltaPhi);
    me["DeltaPhivsEt"]->Fill(et,deltaPhi);
    me["DeltaPhiOverPhivsEt"]->Fill(et,deltaPhi/phi);
    me["DeltaPhivsEta"]->Fill(eta,deltaPhi);
    me["DeltaPhiOverPhivsEta"]->Fill(eta,deltaPhi/phi);
    me["DeltaPhivsPhi"]->Fill(phi,deltaPhi);
    me["DeltaPhiOverPhivsPhi"]->Fill(phi,deltaPhi/phi);

    me["DeltaR"]->Fill(deltaR);
    me["DeltaRvsEt"]->Fill(et,deltaR);
    me["DeltaRvsEta"]->Fill(eta,deltaR);
    me["DeltaRvsPhi"]->Fill(phi,deltaR);

    /* still a work in progress....
    // find all truth candidate matches within the given cone
    CandidateCollection match_candidates = algo_->findAllInCone(particle,truth_candidates,MAXCONE);

    // variables for filling out the 'NumMatches' histogram
    int nmatches = match_candidates.size();
    static int nbins = me["NumMatchesVsDeltaR"]->getNbinsX();
    static double mincone = 0, maxcone = 1;
    static double binwidth = (maxcone - mincone) / (2 * nbins);
    int lastbin = 1;

    // loop over matching candidates to find the delta-R's
    CandidateCollection::iterator match;
    for (match = match_candidates.begin(); match != match_candidates.end(); match++) {

      // calculate this match's delta-R
      const Candidate *match_particle = &(*match);
      double deltaR = algo_->deltaR(particle,match_particle);

      // identify the bin number associated with this delta-R
      int upperbin = (int)ceil(deltaR / binwidth) + 1;

      // fill the histogram
      for (int bin = lastbin; bin < upperbin; bin++)
        me["NumMatchesVsDeltaR"]->

      // adjust the variables for the next pass
      nmatches--; lastbin = bin;

    }*/

  }

}

void PFBenchmarkAnalyzer::endJob() 
{

  // Store the DAQ Histograms
  if (outputFile_.size() > 0 && dbe_)
    dbe_->save(outputFile_);

}
