#include <cuda_runtime.h>

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/stream/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "Test/Wrapper_uses_KernelFunction/interface/wrapper.h"

namespace Plugin_uses_Wrapper_uses_KernelFunction {

class Analyzer : public edm::stream::EDAnalyzer<> {
public:
  Analyzer(edm::ParameterSet const& config)
  { }

  ~Analyzer() override
  { }

  void analyze(edm::Event const& event, edm::EventSetup const& setup) override {
    Wrapper_uses_KernelFunction::wrapper();
  }
};

}

#include "FWCore/Framework/interface/MakerMacros.h"
using Plugin_uses_Wrapper_uses_KernelFunctionAnalyzer = Plugin_uses_Wrapper_uses_KernelFunction::Analyzer;
DEFINE_FWK_MODULE(Plugin_uses_Wrapper_uses_KernelFunctionAnalyzer);
