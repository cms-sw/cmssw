#ifndef GENERICBENCHMARKANALYZER_H
#define GENERICBENCHMARKANALYZER_H

// author: Mike Schmitt (The University of Florida)
// date: 11/7/2007
// extension: Leo Neuhaus & Joanna Weng 09.2008

#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"

#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "RecoParticleFlow/Benchmark/interface/GenericBenchmark.h"

#include <map>

class GenericBenchmarkAnalyzer : public edm::EDAnalyzer, public GenericBenchmark {
public:
  explicit GenericBenchmarkAnalyzer(const edm::ParameterSet &);
  ~GenericBenchmarkAnalyzer() override;

  void analyze(const edm::Event &, const edm::EventSetup &) override;
  void beginJob() override;
  void endJob() override;

private:
  // Inputs from Configuration File
  std::string outputFile_;
  edm::EDGetTokenT<edm::View<reco::Candidate>> myTruth_;
  edm::EDGetTokenT<edm::View<reco::Candidate>> myReco_;
  edm::InputTag inputTruthLabel_;
  edm::InputTag inputRecoLabel_;
  std::string benchmarkLabel_;
  bool startFromGen_;
  bool plotAgainstRecoQuantities_;
  bool onlyTwoJets_;
  double recPt_cut;
  double minEta_cut;
  double maxEta_cut;
  double deltaR_cut;
  float minDeltaEt_;
  float maxDeltaEt_;
  float minDeltaPhi_;
  float maxDeltaPhi_;
  bool doMetPlots_;
};

#endif  // GENERICBENCHMARKANALYZER_H
