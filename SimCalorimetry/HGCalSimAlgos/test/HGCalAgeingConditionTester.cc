#include <cstdio>
#include <chrono>
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/StreamID.h"
#include "FWCore/Framework/interface/ESWatcher.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "CondFormats/HGCalObjects/interface/HGCalAgeingCondsHost.h"
#include "CondFormats/DataRecord/interface/HGCalAgeingCondsRcd.h"

class HGCalAgeingConditionTester : public edm::one::EDAnalyzer<> {
public:
  explicit HGCalAgeingConditionTester(const edm::ParameterSet&);
  static void fillDescriptions(edm::ConfigurationDescriptions&);

private:
  void analyze(const edm::Event&, const edm::EventSetup&) override;

  edm::ESWatcher<HGCalAgeingCondsRcd> cfgWatcher_;
  edm::ESGetToken<hgcal::HGCalAgeingCondsHost, HGCalAgeingCondsRcd> ageingParamsTkn_;
};

//
HGCalAgeingConditionTester::HGCalAgeingConditionTester(const edm::ParameterSet& iConfig)
    : ageingParamsTkn_(esConsumes()) {}

//
void HGCalAgeingConditionTester::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  // if the cfg didn't change there's nothing else to do
  if (!cfgWatcher_.check(iSetup))
    return;

  //test dense index token
  auto const& ageingParams = iSetup.getData(ageingParamsTkn_);
  printf("Retrieved %d ageing parameters info\n", ageingParams.view().metadata().size());
}

//
void HGCalAgeingConditionTester::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::ESInputTag>("ageingParams", edm::ESInputTag(""))->setComment("Aging parameters SoA");
  descriptions.addWithDefaultLabel(desc);
}

// define this as a plug-in
DEFINE_FWK_MODULE(HGCalAgeingConditionTester);
