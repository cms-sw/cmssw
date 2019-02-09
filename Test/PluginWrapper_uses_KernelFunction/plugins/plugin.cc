#include <cuda_runtime.h>

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/stream/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "wrapper.h"

namespace PluginWrapper_uses_KernelFunction {

class Analyzer : public edm::stream::EDAnalyzer<> {
public:
  Analyzer(edm::ParameterSet const& config)
  { }

  ~Analyzer() override
  { }

  void analyze(edm::Event const& event, edm::EventSetup const& setup) override {
    PluginWrapper_uses_KernelFunction::wrapper();
  }
};

}

#include "FWCore/Framework/interface/MakerMacros.h"
using PluginWrapper_uses_KernelFunctionAnalyzer = PluginWrapper_uses_KernelFunction::Analyzer;
DEFINE_FWK_MODULE(PluginWrapper_uses_KernelFunctionAnalyzer);
