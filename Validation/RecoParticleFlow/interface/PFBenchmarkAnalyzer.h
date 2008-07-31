#ifndef PFBENCHMARKANALYZER_H
#define PFBENCHMARKANALYZER_H

// author: Mike Schmitt (The University of Florida)
// date: 11/7/2007

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "RecoParticleFlow/Benchmark/interface/PFBenchmarkAna.h"

#include <string>
#include <map>

class PFBenchmarkAnalyzer: public edm::EDAnalyzer, public PFBenchmarkAna {
public:

  explicit PFBenchmarkAnalyzer(const edm::ParameterSet&);
  virtual ~PFBenchmarkAnalyzer();

  virtual void analyze(const edm::Event&, const edm::EventSetup&);
  virtual void beginJob(const edm::EventSetup&) ;
  virtual void endJob() ;

 private:

  // Inputs from Configuration File
  std::string outputFile_;
  std::string inputTruthLabel_;
  std::string inputRecoLabel_;
  std::string benchmarkLabel_;
  bool plotAgainstRecoQuantities_;
  double recPt_cut;
  double maxEta_cut;
  double deltaR_cut;
};

#endif // PFBENCHMARKANALYZER_H
