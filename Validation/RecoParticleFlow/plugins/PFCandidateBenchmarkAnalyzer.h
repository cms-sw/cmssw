#ifndef __Validation_RecoParticleFlow_PFCandidateBenchmarkAnalyzer__
#define __Validation_RecoParticleFlow_PFCandidateBenchmarkAnalyzer__


#include "Validation/RecoParticleFlow/plugins/BenchmarkAnalyzer.h"
#include "RecoParticleFlow/Benchmark/interface/PFCandidateBenchmark.h"


class TH1F; 

class PFCandidateBenchmarkAnalyzer: public BenchmarkAnalyzer, public PFCandidateBenchmark {
 public:
  
  PFCandidateBenchmarkAnalyzer(const edm::ParameterSet& parameterSet);

  void analyze(const edm::Event&, const edm::EventSetup&);
  void beginJob() ;
  void endJob();

/*  private: */
/*   TH1F *particleId_; */
};

#endif 
