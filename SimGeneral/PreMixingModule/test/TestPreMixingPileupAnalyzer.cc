#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/global/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "SimDataFormats/PileupSummaryInfo/interface/PileupSummaryInfo.h"

#include <algorithm>
#include <vector>

class TestPreMixingPileupAnalyzer : public edm::global::EDAnalyzer<> {
public:
public:
  explicit TestPreMixingPileupAnalyzer(edm::ParameterSet const& iConfig);
  void analyze(edm::StreamID, edm::Event const& iEvent, edm::EventSetup const& iSetup) const override;

private:
  edm::EDGetTokenT<std::vector<PileupSummaryInfo>> inputToken_;
  std::vector<unsigned int> allowedPileups_;
};

TestPreMixingPileupAnalyzer::TestPreMixingPileupAnalyzer(edm::ParameterSet const& iConfig)
    : inputToken_(consumes<std::vector<PileupSummaryInfo>>(iConfig.getUntrackedParameter<edm::InputTag>("src"))),
      allowedPileups_(iConfig.getUntrackedParameter<std::vector<unsigned int>>("allowedPileups")) {}

void TestPreMixingPileupAnalyzer::analyze(edm::StreamID,
                                          edm::Event const& iEvent,
                                          edm::EventSetup const& iSetup) const {
  edm::Handle<std::vector<PileupSummaryInfo>> h;
  iEvent.getByToken(inputToken_, h);
  const auto& summary = *h;

  auto it = std::find_if(summary.begin(), summary.end(), [](const auto& s) { return s.getBunchCrossing() == 0; });
  if (it == summary.end()) {
    throw cms::Exception("LogicError") << "Did not find PileupSummaryInfo in bunch crossing 0";
  }

  float trueNumInteractions = it->getTrueNumInteractions();
  auto pubin = static_cast<unsigned int>(trueNumInteractions);
  auto it2 = std::find(allowedPileups_.begin(), allowedPileups_.end(), pubin);
  if (it2 == allowedPileups_.end()) {
    cms::Exception ex{"LogicError"};
    ex << "Got event with true number of interactions " << trueNumInteractions << ", pileup bin is thus " << pubin
       << ", that is not in the list of allowed values:";
    for (auto v : allowedPileups_) {
      ex << " " << v;
    }
    throw ex;
  }
}

DEFINE_FWK_MODULE(TestPreMixingPileupAnalyzer);
