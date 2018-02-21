#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <iostream>

namespace l1tVertexFinder {

  class VertexAnalyzer : public edm::EDAnalyzer {

  public:
    explicit VertexAnalyzer(const edm::ParameterSet&);
    ~VertexAnalyzer();

  private:
    void beginJob() override;
    void analyze(const edm::Event& evt, const edm::EventSetup& setup);
    void endJob() override;
  };

  VertexAnalyzer::VertexAnalyzer(const edm::ParameterSet&) {};

  void VertexAnalyzer::beginJob() {};
  void VertexAnalyzer::analyze(const edm::Event& evt, const edm::EventSetup& setup)
  {
    std::cout << "processing " << std::endl;
  }
  void VertexAnalyzer::endJob() {};

  VertexAnalyzer::~VertexAnalyzer() {};
}

using namespace l1tVertexFinder;

//define this as a plug-in
DEFINE_FWK_MODULE(VertexAnalyzer);
