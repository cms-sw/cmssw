#include <cuda_runtime.h>

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/stream/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "Test/WrapperKernelFunction/interface/wrapper.h"

namespace Plugin_uses_WrapperKernelFunction {

class Analyzer : public edm::stream::EDAnalyzer<> {
public:
  Analyzer(edm::ParameterSet const& config)
  { }

  ~Analyzer() override
  { }

  void analyze(edm::Event const& event, edm::EventSetup const& setup) override {
    WrapperKernelFunction::wrapper();
  }
};

}

#include "FWCore/Framework/interface/MakerMacros.h"
using Plugin_uses_WrapperKernelFunctionAnalyzer = Plugin_uses_WrapperKernelFunction::Analyzer;
DEFINE_FWK_MODULE(Plugin_uses_WrapperKernelFunctionAnalyzer);
