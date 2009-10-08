#ifndef __Validation_RecoParticleFlow_BenchmarkAnalyzer__
#define __Validation_RecoParticleFlow_BenchmarkAnalyzer__

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/ParameterSet/interface/InputTag.h"

#include "RecoParticleFlow/Benchmark/interface/Benchmark.h"

/// abtract base class for benchmark analyzers
class BenchmarkAnalyzer: public edm::EDAnalyzer {
public:

  BenchmarkAnalyzer();
  explicit BenchmarkAnalyzer(const edm::ParameterSet&);
  virtual ~BenchmarkAnalyzer() {}

/*   virtual void analyze(const edm::Event&, const edm::EventSetup&) = 0; */
/*   virtual void beginJob() = 0; */
/*   virtual void endJob() = 0; */

 protected:

  /// name of the output root file
  std::string outputFile_;
  
  /// input collection
  edm::InputTag inputLabel_;

  /// benchmark label
  std::string benchmarkLabel_;

};

#endif 
